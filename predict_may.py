"""Predict closes for each trading day in May using the universal model.

Run: python predict_may.py
Outputs: predictions_may_<SYMBOL>.csv in the repo root.
"""
import os
import pickle
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
import pytorch_lightning as pl

# --- Configuration (match app.py paths) ---
DRIVE_ROOT = "."
CHECKPOINT_DIR = f"{DRIVE_ROOT}/nse_predictor2/checkpoints/universal"

SEQ_LEN = 60
PRED_LEN = 2

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


class StockScaler:
    def __init__(self):
        self.means = {}
        self.stds = {}

    def transform(self, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        df = df.copy()
        for c in cols:
            df[c] = (df[c] - self.means[c]) / self.stds[c]
        return df

    def inverse_transform_targets(self, arr: np.ndarray,
                                  target_cols: list[str]) -> np.ndarray:
        out = arr.copy()
        if arr.ndim == 1:
            out = out * self.stds[target_cols[0]] + self.means[target_cols[0]]
        else:
            for i, c in enumerate(target_cols):
                out[:, i] = out[:, i] * self.stds[c] + self.means[c]
        return out


def add_features(df: pd.DataFrame) -> pd.DataFrame:
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


class StockPredictorLit(pl.LightningModule):
    def __init__(self, model: nn.Module, lr: float = 1e-3):
        super().__init__()
        self.model = model
        self.lr    = lr


def load_model_resources_local():
    STOCK_ID_MAP_PATH = f"{DRIVE_ROOT}/nse_predictor2/stock_id_map.pkl"
    SCALERS_PATH = f"{DRIVE_ROOT}/nse_predictor2/scalers.pkl"
    CHECKPOINT_PATH = f"{CHECKPOINT_DIR}/last.ckpt"

    if not os.path.exists(STOCK_ID_MAP_PATH) or not os.path.exists(SCALERS_PATH) or not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError("Required model resources (stock_id_map/scalers/last.ckpt) not found under nse_predictor2.")

    with open(STOCK_ID_MAP_PATH, "rb") as f:
        stock_id_map = pickle.load(f)
    with open(SCALERS_PATH, "rb") as f:
        scalers = pickle.load(f)

    n_features_val = len(FEATURE_COLS)
    n_stocks_val   = max(stock_id_map.values()) + 1 if stock_id_map else 0

    model = UniversalStockTransformer(n_features=n_features_val, n_stocks=n_stocks_val,
                                      seq_len=SEQ_LEN, pred_len=PRED_LEN)
    lit = StockPredictorLit.load_from_checkpoint(CHECKPOINT_PATH, model=model)
    lit.eval()
    return stock_id_map, scalers, lit


def predict_for_month(symbol: str, year: int = 2026, month: int = 5):
    stock_id_map, scalers, lit_model = load_model_resources_local()

    if symbol not in stock_id_map:
        raise KeyError(f"Symbol {symbol} not found in stock_id_map")

    bdays = pd.bdate_range(start=datetime(year, month, 1), end=datetime(year, month, 31))
    results = []
    device = torch.device("cpu")

    for target in bdays:
        # use previous calendar day as last known close (yfinance end is inclusive)
        prev_day = target - timedelta(days=1)
        fetch_start = (prev_day - timedelta(days=450)).strftime("%Y-%m-%d")
        end_str = prev_day.strftime("%Y-%m-%d")
        try:
            raw = yf.download(symbol, start=fetch_start, end=end_str, progress=False, auto_adjust=True)
        except Exception as e:
            results.append((target.date(), None, f"download error: {e}"))
            continue

        if raw.empty or 'Close' not in raw.columns:
            results.append((target.date(), None, "no data"))
            continue

        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0 if 'Close' in raw.columns.get_level_values(0) else 1)

        df = add_features(raw[["Open","High","Low","Close","Volume"]])
        all_feat_cols = [c for c in FEATURE_COLS if c in df.columns]

        scaler = scalers[symbol]
        df_norm = scaler.transform(df, all_feat_cols)
        arr = df_norm[all_feat_cols].values.astype(np.float32)

        if len(arr) < SEQ_LEN:
            results.append((target.date(), None, f"insufficient history: {len(arr)} rows"))
            continue

        window = torch.tensor(arr[-SEQ_LEN:]).unsqueeze(0).to(device)
        sid    = torch.tensor([stock_id_map[symbol]], dtype=torch.long).to(device)

        with torch.no_grad():
            pred_norm = lit_model.model(window, sid).squeeze(0).cpu().numpy()

        # first element corresponds to next day
        first_pred_norm = float(pred_norm[0])
        pred_log_ret = first_pred_norm * scaler.stds["log_ret_1"] + scaler.means["log_ret_1"]
        last_close = df["Close"].iloc[-1]
        predicted_close = float(last_close * np.exp(pred_log_ret))
        results.append((target.date(), predicted_close, "ok"))

    out = pd.DataFrame(results, columns=["Date","Predicted_Close","Status"]).set_index("Date")
    out.sort_index(inplace=True)
    return out


def main():
    # pick a default symbol
    stock_id_map_path = f"{DRIVE_ROOT}/nse_predictor2/stock_id_map.pkl"
    if not os.path.exists(stock_id_map_path):
        print("stock_id_map.pkl not found. Ensure nse_predictor2 is present.")
        return
    with open(stock_id_map_path, "rb") as f:
        stock_id_map = pickle.load(f)

    default_symbol = "RELIANCE.NS" if "RELIANCE.NS" in stock_id_map else list(stock_id_map.keys())[0]
    print(f"Using symbol: {default_symbol}")
    print("Predicting for May 2026 business days...")
    df = predict_for_month(default_symbol, year=2026, month=5)
    out_path = f"predictions_may_{default_symbol.replace('/','_')}.csv"
    df.to_csv(out_path)
    print(f"Saved predictions to {out_path}")


if __name__ == '__main__':
    main()
