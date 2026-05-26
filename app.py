
import os
import sys
import pickle
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from copy import deepcopy

import numpy as np
import pandas as pd
import yfinance as yf

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl

import streamlit as st

warnings.filterwarnings("ignore")


# --- CONFIGURATION (Adjust for deployment) ---
# For deployment, assume models/data are relative to app.py
DRIVE_ROOT     = "." # Assumes nse_predictor2 content is in the same dir as app.py
CHECKPOINT_DIR = f"{DRIVE_ROOT}/nse_predictor2/checkpoints/universal"
ADAPTER_DIR    = f"{DRIVE_ROOT}/nse_predictor2/checkpoints/adapters" # New: for hybrid model
DATA_CACHE_DIR = f"{DRIVE_ROOT}/nse_predictor2/data_cache"

# Model Hyperparameters (must match trained model)
SEQ_LEN        = 60
PRED_LEN       = 2
BATCH_SIZE     = 256 # Not directly used for single inference, but part of model config
MAX_EPOCHS     = 60
LR             = 1e-3
D_MODEL        = 128
N_HEADS        = 4
N_LAYERS       = 3
DROPOUT        = 0.1
GRAD_CLIP      = 1.0
ADAPTER_RANK   = 8 # New: for hybrid model adapter bottleneck dimension

# Feature columns (must match training)
FEATURE_COLS = [
    "ret_1","ret_5","ret_20",
    "log_ret_1","log_ret_5",
    "sma_10","sma_20","sma_50","sma_200",
    "ema_12","ema_26",
    "macd","macd_signal","macd_hist",
    "rsi_14",
    "bb_upper","bb_lower","bb_pct",
    "atr_14",
    "vol_ratio","log_volume",
    "hl_ratio","oc_ratio",
    "day_of_week","month",
]
TARGET_COLS = ["log_ret_1"]

# Make StockScaler available in __main__ for pickle compatibility
import __main__

# --- Custom Unpickler for handling __main__ class references ---
class CustomUnpickler(pickle.Unpickler):
    """Handle unpickling when classes were saved in a different __main__ context."""
    def find_class(self, module, name):
        if module == '__main__' and name == 'StockScaler':
            return StockScaler
        if module == '__main__' and name == 'UniversalStockTransformer':
            return UniversalStockTransformer
        if module == '__main__' and name == 'StockPredictorLit':
            return StockPredictorLit
        return super().find_class(module, name)


# --- StockScaler Class (from universal_model.py) ---
class StockScaler:
    """Fits Z-score per feature on train split; transforms all splits."""
    def __init__(self):
        self.means = {}
        self.stds  = {}

    def fit(self, df: pd.DataFrame, cols: list[str]):
        for c in cols:
            self.means[c] = df[c].mean()
            self.stds[c]  = df[c].std() + 1e-9

    def transform(self, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        df = df.copy()
        for c in cols:
            df[c] = (df[c] - self.means[c]) / self.stds[c]
        return df

    def inverse_transform_targets(self, arr: np.ndarray,
                                  target_cols: list[str]) -> np.ndarray:
        """Inverse-transform targets. Handles both 1D and 2D arrays."""
        out = arr.copy()
        if arr.ndim == 1:  # (PRED_LEN,) for log returns
            out = out * self.stds[target_cols[0]] + self.means[target_cols[0]]
        else:  # (N, PRED_LEN) for legacy 2D
            for i, c in enumerate(target_cols):
                out[:, i] = out[:, i] * self.stds[c] + self.means[c]
        return out

__main__.StockScaler = StockScaler # Register for pickle


# --- Feature Engineering (from universal_model.py) ---
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators as input features.
    """
    df = df.copy()
    c = df["Close"]

    df["ret_1"]  = c.pct_change(1)
    df["ret_5"]  = c.pct_change(5)
    df["ret_20"] = c.pct_change(20)

    df["sma_10"] = c.rolling(10).mean()
    df["sma_20"] = c.rolling(20).mean()
    df["sma_50"] = c.rolling(50).mean()
    df["sma_200"]= c.rolling(200).mean()

    df["ema_12"] = c.ewm(span=12, adjust=False).mean()
    df["ema_26"] = c.ewm(span=26, adjust=False).mean()

    df["macd"]        = df["ema_12"] - df["ema_26"]
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]

    delta  = c.diff()
    gain   = delta.clip(lower=0).rolling(14).mean()
    loss   = (-delta.clip(upper=0)).rolling(14).mean()
    rs     = gain / (loss + 1e-9)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    bb_mid        = c.rolling(20).mean()
    bb_std        = c.rolling(20).std()
    df["bb_upper"] = bb_mid + 2 * bb_std
    df["bb_lower"] = bb_mid - 2 * bb_std
    df["bb_pct"]   = (c - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-9)

    high, low = df["High"], df["Low"]
    tr = pd.concat([high - low,
                    (high - c.shift()).abs(),
                    (low  - c.shift()).abs()], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean()

    df["vol_ma20"]  = df["Volume"].rolling(20).mean()
    df["vol_ratio"] = df["Volume"] / (df["vol_ma20"] + 1e-9)

    df["hl_ratio"]  = (high - low) / (c + 1e-9)
    df["oc_ratio"]  = (df["Open"] - c) / (c + 1e-9)

    df["day_of_week"] = df.index.dayofweek.astype(float) / 4.0
    df["month"]       = (df.index.month - 1).astype(float) / 11.0

    df["log_ret_1"]  = np.log(c / c.shift(1) + 1e-9)
    df["log_ret_5"]  = np.log(c / c.shift(5) + 1e-9)
    df["log_volume"] = np.log(df["Volume"] + 1e-9)

    df.dropna(inplace=True)
    return df


# --- Model Architecture (from universal_model.py) ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.drop(x + self.pe[:, :x.size(1)])


class UniversalStockTransformer(nn.Module):
    def __init__(self, n_features: int, n_stocks: int,
                d_model: int = 128, n_heads: int = 4,
                n_layers: int = 3, dropout: float = 0.1,
                seq_len: int = 60, pred_len: int = 2):
        super().__init__()
        self.d_model   = d_model
        self.pred_len  = pred_len
        self.seq_len   = seq_len

        self.stock_emb = nn.Embedding(n_stocks, d_model)
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=seq_len + 10, dropout=dropout)

        causal_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
        self.register_buffer("causal_mask", causal_mask)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers,
                                            enable_nested_tensor=False)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, pred_len),
        )

    def forward(self, x: torch.Tensor, stock_ids: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        h = self.input_proj(x)
        se = self.stock_emb(stock_ids).unsqueeze(1)
        h  = h + se.expand(-1, T, -1)
        h = self.pos_enc(h)
        mask = self.causal_mask[:T, :T] # if T < self.seq_len else self.causal_mask # Mask shape needs to adapt to current T
        # Dynamically adjust mask size if needed
        if T < self.causal_mask.shape[0]:
            mask = self.causal_mask[:T, :T]
        else:
            # If T is larger or equal, we might need a new mask or assume it's pre-computed for max_len
            # For this context, assuming T <= max_len that causal_mask is built for.
            # If T > self.causal_mask.shape[0], this would raise an error, but unlikely given PositionalEncoding max_len.
            mask = self.causal_mask

        h = self.encoder(h, mask=mask)
        last = h[:, -1, :]
        out = self.head(last)
        return out.view(B, self.pred_len)

__main__.UniversalStockTransformer = UniversalStockTransformer # Register for pickle


# --- Lightning Module (from universal_model.py) ---
class StockPredictorLit(pl.LightningModule):
    def __init__(self, model: nn.Module, lr: float = 1e-3):
        super().__init__()
        self.model = model
        self.lr    = lr
        self.loss  = nn.HuberLoss(delta=0.5)
        self.save_hyperparameters(ignore=["model"])

    def _step(self, batch, name: str):
        x, sids, y = batch
        pred = self.model(x, sids)
        loss = self.loss(pred, y.squeeze(-1))
        self.log(f"{name}_loss", loss)
        return loss

    def training_step(self, batch, _):   return self._step(batch, "train")
    def validation_step(self, batch, _): return self._step(batch, "val")
    def test_step(self, batch, _):       return self._step(batch, "test")

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=MAX_EPOCHS, eta_min=1e-6)
        return [opt], [{"scheduler": sched, "interval": "epoch"}]

__main__.StockPredictorLit = StockPredictorLit # Register for pickle


# --- Hybrid Model Components (from hybrid_model.py) ---
def _flatten_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure single-level column headers regardless of yfinance version."""
    if isinstance(df.columns, pd.MultiIndex):
        level0 = df.columns.get_level_values(0)
        if "Close" in level0:
            df.columns = level0
        else:
            df.columns = df.columns.get_level_values(1)
    return df


class StockAdapter(nn.Module):
    def __init__(self, base_model: UniversalStockTransformer,
                 rank: int = ADAPTER_RANK):
        super().__init__()
        self.base = base_model
        self.rank = rank

        # Freeze ALL base parameters
        for p in self.base.parameters():
            p.requires_grad = False

        d = self.base.d_model

        # Small bottleneck MLP — only these weights are trained
        self.adapter_mlp = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, rank),
            nn.GELU(),
            nn.Linear(rank, d),
        )

        # Stock-specific output head
        # Output: (B, PRED_LEN)  — flat log returns, same as universal
        self.head_override = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(d, PRED_LEN),   # — PRED_LEN not PRED_LEN*2
        )

    def forward(self, x: torch.Tensor, stock_ids: torch.Tensor) -> torch.Tensor:
        """
        x:         (B, SEQ_LEN, n_features)
        stock_ids: (B,)
        returns:   (B, PRED_LEN)   — flat log return predictions
        """
        B, T, _ = x.shape

        # -- Base encoder forward (identical to universal) --
        h = self.base.input_proj(x)                        # (B, T, D)
        se = self.base.stock_emb(stock_ids).unsqueeze(1)   # (B, 1, D)
        h = h + se.expand(-1, T, -1)
        h = self.base.pos_enc(h)

        # Causal mask — must match universal's registered buffer
        mask = self.base.causal_mask[:T, :T] # if T < self.base.seq_len else self.base.causal_mask
        # Dynamically adjust mask size if needed
        if T < self.base.causal_mask.shape[0]:
            mask = self.base.causal_mask[:T, :T]
        else:
            mask = self.base.causal_mask

        h = self.base.encoder(h, mask=mask)                # (B, T, D)

        last = h[:, -1, :]                                 # (B, D)

        # -- Adapter residual + stock-specific head --
        last = last + self.adapter_mlp(last)               # residual
        return self.head_override(last)                    # (B, PRED_LEN)

    def trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]

    def n_trainable(self):
        return sum(p.numel() for p in self.trainable_params())


def _load_base_model(base_ckpt_path: str, n_features: int,
                     n_stocks: int) -> UniversalStockTransformer:
    """
    Load base model from checkpoint. Handles two formats:
      1. PyTorch Lightning .ckpt  (produced by Trainer during main training)
      2. Plain torch.save dict with 'model_state_dict' key
         (produced by incremental_update in universal_model.py)
    Freezes all weights after loading.
    """
    model = UniversalStockTransformer(
        n_features=n_features, n_stocks=n_stocks,
        d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
        dropout=DROPOUT, seq_len=SEQ_LEN, pred_len=PRED_LEN,
    )

    raw = torch.load(base_ckpt_path, map_location="cpu")

    if "model_state_dict" in raw:
        # Format written by incremental_update: plain state dict
        model.load_state_dict(raw["model_state_dict"])
    else:
        # Format written by pl.Trainer ModelCheckpoint: Lightning ckpt
        # Need to instantiate the Lit module with the correct model class
        lit = StockPredictorLit.load_from_checkpoint(
            base_ckpt_path, model=model)
        model = lit.model

    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


# --- Prediction Function (from universal_model.py) ---
@st.cache_resource
def load_model_resources():
    """Load stock_id_map, scalers, and the trained model (universal and hybrid)."""
    STOCK_ID_MAP_PATH = f"{DRIVE_ROOT}/nse_predictor2/stock_id_map.pkl"
    SCALERS_PATH = f"{DRIVE_ROOT}/nse_predictor2/scalers.pkl"
    CHECKPOINT_PATH = f"{CHECKPOINT_DIR}/last.ckpt"
    ADAPTER_REGISTRY_PATH = f"{ADAPTER_DIR}/registry.pkl"

    errors = []
    if not os.path.exists(STOCK_ID_MAP_PATH):
        errors.append(f"stock_id_map.pkl not found at {STOCK_ID_MAP_PATH}.")
    if not os.path.exists(SCALERS_PATH):
        errors.append(f"scalers.pkl not found at {SCALERS_PATH}.")
    if not os.path.exists(CHECKPOINT_PATH):
        errors.append(f"Universal model checkpoint not found at {CHECKPOINT_PATH}.")

    if errors:
        for err in errors: st.error(err)
        st.stop()

    with open(STOCK_ID_MAP_PATH, "rb") as f:
        stock_id_map = pickle.load(f)
    with open(SCALERS_PATH, "rb") as f:
        scalers = CustomUnpickler(f).load() # Use CustomUnpickler

    # Probe feature count from a dummy run or stored config
    sample_sym = next(iter(stock_id_map.keys()))
    # Need to load a sample parquet to infer n_features
    # This part assumes data_cache has at least one parquet file matching a symbol
    sample_meta_path = f"{DATA_CACHE_DIR}/meta.pkl"
    if os.path.exists(sample_meta_path):
        with open(sample_meta_path, "rb") as f:
            meta = pickle.load(f)
        if sample_sym in meta:
            sample_df_path = meta[sample_sym]["path"].replace(DRIVE_ROOT + "/nse_predictor2", f"{DRIVE_ROOT}/nse_predictor2") # Adjust path for deployment
            if os.path.exists(sample_df_path):
                sample_df = pd.read_parquet(sample_df_path)
                n_features_val = len([c for c in FEATURE_COLS if c in sample_df.columns])
            else:
                st.warning(f"Could not find sample data at {sample_df_path} for feature count. Falling back to default FEATURE_COLS length.")
                n_features_val = len(FEATURE_COLS)
        else:
            st.warning(f"Sample symbol '{sample_sym}' not in meta data. Falling back to default FEATURE_COLS length.")
            n_features_val = len(FEATURE_COLS)
    else:
        st.warning(f"Meta data not found at {sample_meta_path}. Falling back to default FEATURE_COLS length.")
        n_features_val = len(FEATURE_COLS)

    n_stocks_val   = max(stock_id_map.values()) + 1 if stock_id_map else 0

    # Load Universal Model for Free Mode
    universal_model = UniversalStockTransformer(
        n_features=n_features_val, n_stocks=n_stocks_val,
        d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
        dropout=DROPOUT, seq_len=SEQ_LEN, pred_len=PRED_LEN,
    )
    lit_universal = StockPredictorLit.load_from_checkpoint(CHECKPOINT_PATH, model=universal_model)
    lit_universal.eval()

    # Load Base Model for Hybrid Mode (used by adapters)
    base_model_hybrid = _load_base_model(CHECKPOINT_PATH, n_features_val, n_stocks_val)

    # Load Adapter Registry for Pro Mode
    adapter_registry = {}
    if os.path.exists(ADAPTER_REGISTRY_PATH):
        with open(ADAPTER_REGISTRY_PATH, "rb") as f:
            adapter_registry = pickle.load(f)
    else:
        st.warning(f"Adapter registry not found at {ADAPTER_REGISTRY_PATH}. Pro mode might not work for all stocks.")

    return stock_id_map, scalers, lit_universal, base_model_hybrid, adapter_registry


def predict_next_days_universal(symbol: str, scalers: dict,
                   stock_id_map: dict, lit_model: StockPredictorLit,
                   seq_len: int = SEQ_LEN, pred_len_user: int = PRED_LEN) -> pd.DataFrame | None:
  """Predicts next `pred_len_user` days' close prices for a given stock using the universal model."""

  if symbol not in scalers or symbol not in stock_id_map:
      st.error(f"Stock '{symbol}' not found in model resources. Please select another.")
      return None

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  lit_model.to(device)

  today = datetime.today()
  fetch_start = (today - timedelta(days=450)).strftime("%Y-%m-%d")

  raw = yf.download(symbol, start=fetch_start, progress=False, auto_adjust=True)

  if isinstance(raw.columns, pd.MultiIndex):
      raw.columns = raw.columns.get_level_values(0 if 'Close' in raw.columns.get_level_values(0) else 1)

  if raw.empty or 'Close' not in raw.columns:
      st.error(f"No historical data available for {symbol}.")
      return None

  df = add_features(raw[["Open","High","Low","Close","Volume"]])
  all_feat_cols = [c for c in FEATURE_COLS if c in df.columns]

  scaler = scalers[symbol]
  df_norm = scaler.transform(df, all_feat_cols)
  arr = df_norm[all_feat_cols].values.astype(np.float32)

  if len(arr) < seq_len:
      st.warning(f"Not enough historical data for {symbol}. Found {len(arr)} rows, need {seq_len}. Skipping prediction.")
      return None

  window = torch.tensor(arr[-seq_len:]).unsqueeze(0).to(device)
  sid    = torch.tensor([stock_id_map[symbol]], dtype=torch.long).to(device)

  with torch.no_grad():
      # Call the model's forward pass, it will output PRED_LEN predictions
      pred_norm_fixed_len = lit_model.model(window, sid).squeeze(0).cpu().numpy()

  # We need to ensure that pred_norm_fixed_len always has length PRED_LEN (2 in this case)
  # If user requests more than PRED_LEN, we'll repeat the last prediction.
  # If user requests less, we'll truncate.
  actual_pred_len_output = lit_model.model.pred_len # This should be PRED_LEN (e.g., 2)

  if pred_len_user > actual_pred_len_output:
      # Repeat the last prediction for the additional days
      last_prediction_value = pred_norm_fixed_len[-1]
      extended_pred_norm = np.concatenate(
          [pred_norm_fixed_len, np.full(pred_len_user - actual_pred_len_output, last_prediction_value)]
      )
  else:
      extended_pred_norm = pred_norm_fixed_len[:pred_len_user]

  pred_close_ret = extended_pred_norm * scaler.stds["log_ret_1"] + scaler.means["log_ret_1"]

  last_close = df["Close"].iloc[-1]
  predicted_closes = []
  current_base_close = last_close

  for day_idx in range(pred_len_user):
      next_close = current_base_close * np.exp(pred_close_ret[day_idx])
      predicted_closes.append(next_close)
      current_base_close = next_close

  last_date = df.index[-1]
  dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=pred_len_user)

  result = pd.DataFrame({
      "Predicted_Close": predicted_closes,
  }, index=dates)
  result.index.name = "Date"
  return result


def hybrid_predict(
symbol: str,
                   base_model: UniversalStockTransformer,
                   adapter_dir: str,
                   scalers: dict,
                   stock_id_map: dict,
                   pred_len: int = PRED_LEN,
                   confidence: bool = True,
                   n_mc: int = 30) -> pd.DataFrame | None:
    """
    Pro-tier inference: base model + stock adapter + optional MC-Dropout CI.

    Returns DataFrame with:
        Predicted_Close                       (always)
        Close_Low_90, Close_High_90           (if confidence=True)
    Prices are in original INR scale.
    """
    adapter_path = f"{adapter_dir}/{symbol.replace('.', '_')}.pt"
    if not os.path.exists(adapter_path):
        st.warning(f"[HYBRID] No adapter for {symbol}. Please train it first.")
        return None

    if symbol not in scalers or symbol not in stock_id_map:
        st.error(f"[HYBRID] {symbol} missing from scalers/stock_id_map.")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -- Fetch recent data --
    today = datetime.today()
    raw = yf.download(
        symbol,
        start=(today - timedelta(days=450)).strftime("%Y-%m-%d"),
        progress=False, auto_adjust=True)
    raw = _flatten_yf_columns(raw)

    if raw.empty or "Close" not in raw.columns:
        st.error(f"[HYBRID] No historical data available for {symbol}.")
        return None

    df = add_features(raw[["Open", "High", "Low", "Close", "Volume"]])
    feat_cols = [c for c in FEATURE_COLS if c in df.columns]
    df_norm   = scalers[symbol].transform(df, feat_cols)
    arr       = df_norm[feat_cols].values.astype(np.float32)

    if len(arr) < SEQ_LEN:
        st.warning(f"[HYBRID] Insufficient rows for {symbol}: {len(arr)} < {SEQ_LEN}")
        return None

    window = torch.tensor(arr[-SEQ_LEN:]).unsqueeze(0).to(device)  # (1, SEQ_LEN, F)
    sid    = torch.tensor([stock_id_map[symbol]], dtype=torch.long).to(device)

    # -- Load base + adapter --
    adapter = StockAdapter(base_model, rank=ADAPTER_RANK)
    ckpt    = torch.load(adapter_path, map_location="cpu")
    adapter.adapter_mlp.load_state_dict(ckpt["adapter_mlp"])
    adapter.head_override.load_state_dict(ckpt["head_override"])
    adapter.to(device)

    last_close = float(df["Close"].iloc[-1])
    scaler_obj = scalers[symbol]   # named to avoid shadowing outer `scalers` dict

    # -- Inverse-transform log returns -> INR prices --
    def logret_to_prices(pred_np: np.ndarray) -> np.ndarray:
        """
        pred_np : (PRED_LEN,) -- raw model output (normalised log return space)
        Returns : (PRED_LEN,) -- INR close prices, chained from last known close.
        Matches universal_model.predict_next_days exactly:
          step 1: un-normalise  ->  log_ret = pred * std + mean
          step 2: chain prices  ->  price_t = price_{t-1} * exp(log_ret_t)
        """
        # Ensure pred_np has at least pred_len elements for processing
        if len(pred_np) < pred_len:
             # This case should ideally not happen if model output is consistent
             # For robustness, we can pad or handle gracefully, e.g., by replicating last value.
             # For now, let's assume it matches.
            pass # Or raise an error, or pad

        log_rets = (pred_np[:pred_len] * scaler_obj.stds["log_ret_1"]
                    + scaler_obj.means["log_ret_1"])
        prices = np.zeros(pred_len, dtype=np.float32)
        base   = last_close
        for j in range(pred_len):
            base      = base * np.exp(log_rets[j])
            prices[j] = base
        return prices

    # -- Inference --
    if confidence:
        # MC-Dropout: keep dropout active, freeze BN
        adapter.train()
        for m in adapter.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d,
                               nn.LayerNorm)):
                m.eval()

        preds_list = []
        with torch.no_grad():
            for _ in range(n_mc):
                p = adapter(window, sid).squeeze(0).cpu().numpy()  # (PRED_LEN,)
                preds_list.append(logret_to_prices(p))

        preds_arr = np.stack(preds_list, axis=0)     # (n_mc, PRED_LEN)
        mean_price = preds_arr.mean(axis=0)
        lo_price   = np.percentile(preds_arr, 5,  axis=0)
        hi_price   = np.percentile(preds_arr, 95, axis=0)

        last_date = df.index[-1]
        dates     = pd.bdate_range(start=last_date + timedelta(days=1),
                                    periods=pred_len)
        return pd.DataFrame({
            "Predicted_Close": mean_price,
            "Close_Low_90":    lo_price,
            "Close_High_90":   hi_price,
        }, index=dates)

    else:
        adapter.eval()
        with torch.no_grad():
            pred_norm = adapter(window, sid).squeeze(0).cpu().numpy()

        prices    = logret_to_prices(pred_norm)
        last_date = df.index[-1]
        dates     = pd.bdate_range(start=last_date + timedelta(days=1),
                                    periods=pred_len)
        return pd.DataFrame({"Predicted_Close": prices}, index=dates)


# --- Streamlit UI ---
st.set_page_config(page_title="Stock Price Predictor", layout="centered")

st.title("📈 NSE Stock Price Predictor")
st.markdown("Predict future closing prices for Indian stocks using Universal or Hybrid Models.")

# Load resources once and cache them
stock_id_map, scalers, lit_universal_model, base_model_hybrid, adapter_registry = load_model_resources()

available_stocks = sorted(stock_id_map.keys())

# Sidebar for user inputs
st.sidebar.header("Prediction Settings")

prediction_mode = st.sidebar.radio(
    "Select Prediction Mode",
    ("Free Mode (Universal Model)", "Pro Mode (Hybrid Model)")
)

selected_symbol = st.sidebar.selectbox(
    "Select Stock Symbol",
    options=available_stocks,
    index=available_stocks.index("RELIANCE.NS") if "RELIANCE.NS" in available_stocks else 0
)

num_days_to_predict = st.sidebar.slider(
    "Number of Days to Predict",
    min_value=1,
    max_value=10, # Limiting to 10 for demo purposes, can be adjusted
    value=2
)

if st.sidebar.button("Get Predictions"):
    if selected_symbol:
        predictions_df = None
        with st.spinner(f"Fetching data and predicting for {selected_symbol} in {prediction_mode}..."):
            if prediction_mode == "Free Mode (Universal Model)":
                predictions_df = predict_next_days_universal(
                    symbol=selected_symbol,
                    scalers=scalers,
                    stock_id_map=stock_id_map,
                    lit_model=lit_universal_model,
                    pred_len_user=num_days_to_predict
                )
            elif prediction_mode == "Pro Mode (Hybrid Model)":
                if selected_symbol not in adapter_registry or adapter_registry[selected_symbol] is None:
                    st.warning(f"Hybrid adapter not found for {selected_symbol}. Please train it first for Pro Mode. Falling back to Free Mode.")
                    # Fallback to universal if adapter not found
                    predictions_df = predict_next_days_universal(
                        symbol=selected_symbol,
                        scalers=scalers,
                        stock_id_map=stock_id_map,
                        lit_model=lit_universal_model,
                        pred_len_user=num_days_to_predict
                    )
                    st.info("Displayed predictions are from Free Mode (Universal Model).")
                else:
                    predictions_df = hybrid_predict(
                        symbol=selected_symbol,
                        base_model=base_model_hybrid,
                        adapter_dir=ADAPTER_DIR,
                        scalers=scalers,
                        stock_id_map=stock_id_map,
                        pred_len=num_days_to_predict,
                        confidence=True # Enable confidence intervals for pro mode
                    )

            if predictions_df is not None:
                st.subheader(f"Predictions for {selected_symbol} for next {num_days_to_predict} day(s) ({prediction_mode})")

                if "Close_Low_90" in predictions_df.columns:
                    # Display with confidence intervals for Pro Mode
                    st.dataframe(predictions_df.style.format({
                        'Predicted_Close': "₹ {:,.2f}",
                        'Close_Low_90': "₹ {:,.2f}",
                        'Close_High_90': "₹ {:,.2f}"
                    }))

                    # Plot with shaded confidence interval
                    st.line_chart(predictions_df[["Predicted_Close", "Close_Low_90", "Close_High_90"]])
                else:
                    # Display for Free Mode
                    st.dataframe(predictions_df.style.format(formatter={'Predicted_Close': "₹ {:,.2f}"}))
                    st.line_chart(predictions_df['Predicted_Close'])

            else:
                st.error("Could not generate predictions. Please check logs for details.")
    else:
        st.warning("Please select a stock symbol.")

st.sidebar.markdown("""
---
**Note:** This app supports both Universal (Free) and Hybrid (Pro) models.

For deployment, ensure the `nse_predictor2` directory (containing checkpoints, data cache, `stock_id_map.pkl`, `scalers.pkl`, and `adapters` directory) is correctly set up relative to `app.py`.
""")
