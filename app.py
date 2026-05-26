%%writefile app.py

import os
import sys
import pickle
import warnings
from pathlib import Path
from datetime import datetime, timedelta

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
        mask = self.causal_mask[:T, :T] if T < self.seq_len else self.causal_mask
        h = self.encoder(h, mask=mask)
        last = h[:, -1, :]
        out = self.head(last)
        return out.view(B, self.pred_len)


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


# --- Prediction Function (from universal_model.py) ---
@st.cache_resource
def load_model_resources():
    """Load stock_id_map, scalers, and the trained model."""
    STOCK_ID_MAP_PATH = f"{DRIVE_ROOT}/nse_predictor2/stock_id_map.pkl"
    SCALERS_PATH = f"{DRIVE_ROOT}/nse_predictor2/scalers.pkl"
    CHECKPOINT_PATH = f"{CHECKPOINT_DIR}/last.ckpt"

    if not os.path.exists(STOCK_ID_MAP_PATH):
        st.error(f"Error: stock_id_map.pkl not found at {STOCK_ID_MAP_PATH}.")
        st.stop()
    if not os.path.exists(SCALERS_PATH):
        st.error(f"Error: scalers.pkl not found at {SCALERS_PATH}.")
        st.stop()
    if not os.path.exists(CHECKPOINT_PATH):
        st.error(f"Error: Model checkpoint not found at {CHECKPOINT_PATH}.")
        st.stop()

    with open(STOCK_ID_MAP_PATH, "rb") as f:
        stock_id_map = pickle.load(f)
    with open(SCALERS_PATH, "rb") as f:
        scalers = pickle.load(f)

    # Rebuild model from hparams stored in checkpoint
    # Probe feature count from a dummy run or stored config if available.
    # For simplicity, we'll assume FEATURE_COLS length for n_features.
    n_features_val = len(FEATURE_COLS)
    n_stocks_val   = max(stock_id_map.values()) + 1 if stock_id_map else 0

    model = UniversalStockTransformer(
        n_features=n_features_val, n_stocks=n_stocks_val,
        d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
        dropout=DROPOUT, seq_len=SEQ_LEN, pred_len=PRED_LEN,
    )
    lit = StockPredictorLit.load_from_checkpoint(CHECKPOINT_PATH, model=model)
    lit.eval()

    return stock_id_map, scalers, lit


def predict_next_days_universal(symbol: str, scalers: dict,
                   stock_id_map: dict, lit_model: StockPredictorLit,
                   seq_len: int = SEQ_LEN, pred_len_user: int = PRED_LEN) -> pd.DataFrame | None:
  """Predicts next `pred_len_user` days' close prices for a given stock."""

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


# --- Streamlit UI ---
st.set_page_config(page_title="Stock Price Predictor", layout="centered")

st.title("📈 NSE Stock Price Predictor")
st.markdown("Predict future closing prices for Indian stocks using a Universal Transformer Model.")

# Load resources once and cache them
stock_id_map, scalers, lit_model = load_model_resources()

available_stocks = sorted(stock_id_map.keys())

# Sidebar for user inputs
st.sidebar.header("Prediction Settings")
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
        with st.spinner(f"Fetching data and predicting for {selected_symbol}..."):
            predictions_df = predict_next_days_universal(
                symbol=selected_symbol,
                scalers=scalers,
                stock_id_map=stock_id_map,
                lit_model=lit_model,
                pred_len_user=num_days_to_predict
            )
            if predictions_df is not None:
                st.subheader(f"Predictions for {selected_symbol} for next {num_days_to_predict} day(s)")
                st.dataframe(predictions_df.style.format(formatter={'Predicted_Close': "₹ {:,.2f}"}))

                # Simple plot of predictions
                st.line_chart(predictions_df['Predicted_Close'])
            else:
                st.error("Could not generate predictions. Please check logs for details.")
    else:
        st.warning("Please select a stock symbol.")

st.sidebar.markdown("""
--- 
**Note:** This demo uses a universal model. 
Ensure `nse_predictor2` directory with models and data is present for deployment.
""")
