"""data_manager.py — Data loading and feature engineering for ANP engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

import config

ALL_TICKERS = sorted(set(
    config.EQUITY_SECTORS_TICKERS + config.FI_COMMODITIES_TICKERS
))


def load_data(token: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download master_data.parquet → (log_returns, macro_df)."""
    file_path = hf_hub_download(
        repo_id=config.HF_DATA_REPO,
        filename=config.HF_DATA_FILE,
        repo_type="dataset",
        token=token,
        cache_dir="./hf_cache",
    )
    df = pd.read_parquet(file_path)
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index().rename(columns={"index": "Date"})
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True).set_index("Date")

    available   = [t for t in ALL_TICKERS if t in df.columns]
    prices      = df[available].ffill()
    log_returns = np.log(prices / prices.shift(1)).dropna()

    macro_cols = [c for c in config.MACRO_COLS if c in df.columns]
    macro_df   = df[macro_cols].reindex(log_returns.index).ffill().fillna(0.0)

    print(
        f"Loaded {len(log_returns)} rows × {len(log_returns.columns)} ETFs"
        f" | Macro: {macro_cols}"
    )
    return log_returns, macro_df


def build_features(
    log_returns: pd.DataFrame,
    macro_df: pd.DataFrame,
    tickers: list[str],
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Build feature matrix X and target matrix Y aligned by date.

    Features (X) for day t — concatenation of:
      - Lagged returns: r_{t-1}, r_{t-5}, r_{t-21}  for each ETF
      - Rolling vol:    std(r_{t-21:t})              for each ETF
      - Rolling mom:    mean(r_{t-63:t}) * 252       for each ETF
      - Macro values:   VIX, DXY, T10Y2Y, TBILL_3M  (z-scored)

    Target (Y) for day t:
      - r_t (same-day log return) for each ETF in `tickers`
      - This is what the context pairs teach the model about this regime day

    Returns
    -------
    X     : (T, n_features) float32
    Y     : (T, n_etf)      float32  — targets aligned to X rows
    dates : DatetimeIndex length T
    """
    avail   = [t for t in tickers if t in log_returns.columns]
    n_etf   = len(avail)
    ret     = log_returns[avail]
    mac     = macro_df.copy()

    # Global z-score macro to bring onto same scale as returns
    mac = (mac - mac.mean()) / (mac.std() + 1e-8)

    max_lag = max(config.LOOKBACK_LAGS + [config.ROLLING_MOM_WINDOW])
    start   = max_lag

    feature_rows, target_rows, dates = [], [], []

    for t in range(start, len(ret)):
        row_feats = []

        for tkr in avail:
            r_series = ret[tkr].values

            # Lagged returns
            for lag in config.LOOKBACK_LAGS:
                idx = t - lag
                row_feats.append(r_series[idx] if idx >= 0 else 0.0)

            # Rolling volatility
            vol_window = r_series[max(0, t - config.ROLLING_VOL_WINDOW): t]
            row_feats.append(float(vol_window.std()) if len(vol_window) > 1 else 0.0)

            # Rolling momentum (annualised)
            mom_window = r_series[max(0, t - config.ROLLING_MOM_WINDOW): t]
            row_feats.append(float(mom_window.mean()) * 252 if len(mom_window) > 1 else 0.0)

        # Macro features
        mac_row = mac.iloc[t].values.tolist()
        row_feats.extend(mac_row)

        feature_rows.append(row_feats)
        target_rows.append(ret.iloc[t].values.tolist())
        dates.append(ret.index[t])

    X = np.array(feature_rows, dtype=np.float32)
    Y = np.array(target_rows,  dtype=np.float32)

    # Clip extreme values
    X = np.clip(X, -10.0, 10.0)
    Y = np.clip(Y, -0.20,  0.20)

    return X, Y, pd.DatetimeIndex(dates)


def make_episodes(
    X: np.ndarray,
    Y: np.ndarray,
    dates: pd.DatetimeIndex,
    n_episodes: int,
    context_size: int,
    query_size: int,
    date_end: str,
    date_start: str | None = None,
    rng: np.random.Generator | None = None,
) -> list[dict]:
    """Sample random episodes from (X, Y) for meta-training.

    Each episode:
      context_x : (context_size, n_features)
      context_y : (context_size, n_etf)
      query_x   : (query_size,   n_features)
      query_y   : (query_size,   n_etf)       ← target to predict

    Episode construction:
      - Sample a random start index i in [0, T - context_size - query_size]
      - Context = rows i : i+context_size
      - Query   = rows i+context_size : i+context_size+query_size
      - Ensures no future leakage: query is strictly after context

    Parameters
    ----------
    date_end   : only sample episodes that end before this date
    date_start : only sample episodes that start after this date (optional)
    """
    if rng is None:
        rng = np.random.default_rng(42)

    end_ts   = pd.Timestamp(date_end)
    start_ts = pd.Timestamp(date_start) if date_start else dates[0]

    valid_starts = [
        i for i in range(len(dates) - context_size - query_size)
        if start_ts <= dates[i]
        and dates[i + context_size + query_size - 1] <= end_ts
    ]

    if len(valid_starts) == 0:
        raise ValueError(
            f"No valid episode start indices found between {date_start} and {date_end}."
        )

    episodes = []
    chosen   = rng.choice(
        valid_starts,
        size=min(n_episodes, len(valid_starts)),
        replace=(n_episodes > len(valid_starts)),
    )

    for i in chosen:
        c_end = i + context_size
        q_end = c_end + query_size
        episodes.append({
            "context_x": X[i:c_end].copy(),
            "context_y": Y[i:c_end].copy(),
            "query_x":   X[c_end:q_end].copy(),
            "query_y":   Y[c_end:q_end].copy(),
        })

    return episodes
