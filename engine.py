"""engine.py — ANP daily inference engine.

Loads the latest checkpoint from HF model repo, builds the 21-day context
window from the most recent data, runs MC inference, and produces scores.

Daily pipeline per universe
---------------------------
1. Load model checkpoint (x_dim, y_dim, weights) from HF
2. Build feature matrix X, target matrix Y for full history
3. For each day t in OOS period:
   a. Context = rows [t-CONTEXT_SIZE : t]   (last 21 days of features + returns)
   b. Query   = row  [t]                    (today — what to predict)
   c. Run ANP.predict(): N_LATENT_SAMPLES MC samples → mu_mean, mu_std per ETF
   d. score(i) = mu_mean(i) / (1 + UNCERTAINTY_WT * mu_std(i))
   e. Cross-sectional z-score → rank
4. Output daily DataFrames + latest snapshot
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download

import config
import data_manager
from model import AttentiveNeuralProcess


def load_checkpoint(
    universe_name: str,
    token: str | None,
    device: torch.device,
) -> tuple[AttentiveNeuralProcess, dict]:
    """Download checkpoint from HF model repo and instantiate model."""
    slug      = universe_name.lower().replace("_", "-")
    ckpt_name = f"anp_checkpoint_{slug}.pt"

    local = hf_hub_download(
        repo_id=config.HF_MODEL_REPO,
        filename=ckpt_name,
        repo_type="model",
        token=token,
        cache_dir="./hf_cache",
    )

    buf  = io.BytesIO(open(local, "rb").read())
    ckpt = torch.load(buf, map_location=device, weights_only=False)

    model = AttentiveNeuralProcess(
        x_dim=ckpt["x_dim"],
        y_dim=ckpt["y_dim"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(
        f"  Loaded checkpoint: {ckpt_name}\n"
        f"  Trained: {ckpt.get('train_date','?')}  "
        f"  Val loss: {ckpt.get('best_val_loss', '?'):.6f}\n"
        f"  x_dim={ckpt['x_dim']}  y_dim={ckpt['y_dim']}  "
        f"  tickers={ckpt['tickers']}"
    )
    return model, ckpt


def zscore_cross(arr: np.ndarray) -> np.ndarray:
    mu  = arr.mean()
    std = arr.std() + 1e-8
    return (arr - mu) / std


def run_engine(
    log_returns: pd.DataFrame,
    macro_df: pd.DataFrame,
    universe_tickers: list[str],
    universe_name: str,
    token: str | None = None,
    device: torch.device | None = None,
) -> dict:
    """Run ANP daily inference for one universe."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    avail = [t for t in universe_tickers if t in log_returns.columns]

    print(
        f"\n{'='*60}\n"
        f"Universe: {universe_name}  ({len(avail)} ETFs)\n"
        f"Period: {log_returns.index[0].date()} → {log_returns.index[-1].date()}"
        f"  ({len(log_returns)} days)\n"
        f"{'='*60}"
    )

    # ── Load model ────────────────────────────────────────────────────────────
    model, ckpt = load_checkpoint(universe_name, token, device)
    model_tickers = ckpt["tickers"]   # tickers model was trained on

    # ── Build features ────────────────────────────────────────────────────────
    X, Y, dates = data_manager.build_features(log_returns, macro_df, model_tickers)

    oos_start = pd.Timestamp(config.OOS_START)

    # ── Storage ───────────────────────────────────────────────────────────────
    score_records   : list[dict] = []
    mu_records      : list[dict] = []
    sigma_records   : list[dict] = []
    ranking_records : list[dict] = []
    daily_records   : list[dict] = []

    n_scored = 0

    for t in range(config.CONTEXT_SIZE, len(X)):
        date = dates[t]
        if date < oos_start:
            continue

        # ── Build context window ──────────────────────────────────────────────
        ctx_x = X[t - config.CONTEXT_SIZE : t]   # (CONTEXT_SIZE, x_dim)
        ctx_y = Y[t - config.CONTEXT_SIZE : t]   # (CONTEXT_SIZE, y_dim)
        qry_x = X[t : t + 1]                     # (1, x_dim)  ← today

        ctx_x_t = torch.tensor(ctx_x, dtype=torch.float32).unsqueeze(0).to(device)
        ctx_y_t = torch.tensor(ctx_y, dtype=torch.float32).unsqueeze(0).to(device)
        qry_x_t = torch.tensor(qry_x, dtype=torch.float32).unsqueeze(0).to(device)

        # ── MC inference ──────────────────────────────────────────────────────
        mu_mean, mu_std = model.predict(
            ctx_x_t, ctx_y_t, qry_x_t,
            n_samples=config.N_LATENT_SAMPLES,
        )

        # Extract predictions for query day → shape (y_dim,)
        mu_arr  = mu_mean[0, 0].cpu().numpy()   # (y_dim,)
        std_arr = mu_std[0, 0].cpu().numpy()    # (y_dim,)

        # ── Score ─────────────────────────────────────────────────────────────
        # Uncertainty-discounted score
        raw_score   = mu_arr / (1.0 + config.UNCERTAINTY_WT * std_arr)
        composite_z = zscore_cross(raw_score)

        # ── Rank ──────────────────────────────────────────────────────────────
        ranked_idx = np.argsort(composite_z)[::-1]
        top_ticker = model_tickers[ranked_idx[0]]
        top_score  = float(composite_z[ranked_idx[0]])
        cash_flag  = top_score < config.CASH_THRESHOLD

        ds = date.strftime("%Y-%m-%d")
        n_scored += 1

        score_records.append({"date": ds,
            **{model_tickers[i]: round(float(composite_z[i]), 6)
               for i in range(len(model_tickers))}})

        mu_records.append({"date": ds,
            **{model_tickers[i]: round(float(mu_arr[i]), 8)
               for i in range(len(model_tickers))}})

        sigma_records.append({"date": ds,
            **{model_tickers[i]: round(float(std_arr[i]), 8)
               for i in range(len(model_tickers))}})

        ranking_records.append({"date": ds,
            **{model_tickers[ranked_idx[r]]: r + 1
               for r in range(len(model_tickers))}})

        daily_records.append({
            "date":       ds,
            "top_ticker": "CASH" if cash_flag else top_ticker,
            "top_score":  round(top_score, 6),
            "cash_flag":  cash_flag,
            "mean_mu":    round(float(mu_arr.mean()), 8),
            "mean_sigma": round(float(std_arr.mean()), 8),
            "uncertainty_regime":
                "HIGH" if std_arr.mean() > std_arr.mean() * 1.5 else "NORMAL",
        })

        if n_scored % 252 == 0 or t == len(X) - 1:
            top5 = [
                (model_tickers[ranked_idx[r]],
                 round(float(composite_z[ranked_idx[r]]), 3),
                 round(float(std_arr[ranked_idx[r]]), 4))
                for r in range(min(5, len(model_tickers)))
            ]
            print(
                f"  {ds} | top5: "
                + "  ".join(
                    f"{tk}(z={sc:+.2f} σ={uc:.4f})" for tk, sc, uc in top5
                )
                + (" [CASH]" if cash_flag else "")
            )

    # ── Latest snapshot ───────────────────────────────────────────────────────
    latest_score   = score_records[-1]
    latest_mu      = mu_records[-1]
    latest_sigma   = sigma_records[-1]
    latest_ranking = ranking_records[-1]
    latest_date    = daily_records[-1]["date"]

    latest_out: dict[str, dict] = {}
    for tkr in model_tickers:
        latest_out[tkr] = {
            "composite_score": latest_score[tkr],
            "mu_pred":         latest_mu[tkr],
            "sigma_pred":      latest_sigma[tkr],
            "rank":            int(latest_ranking[tkr]),
        }

    latest_ranked = sorted(
        latest_out.items(),
        key=lambda x: x[1]["composite_score"],
        reverse=True,
    )

    score_df   = pd.DataFrame(score_records).set_index("date")
    mu_df      = pd.DataFrame(mu_records).set_index("date")
    sigma_df   = pd.DataFrame(sigma_records).set_index("date")
    ranking_df = pd.DataFrame(ranking_records).set_index("date")
    daily_df   = pd.DataFrame(daily_records).set_index("date")

    print(
        f"\n  Latest ({latest_date}) top-{config.TOP_N}: "
        + "  ".join(
            f"{t}(z={v['composite_score']:+.3f} σ={v['sigma_pred']:.4f})"
            for t, v in latest_ranked[: config.TOP_N]
        )
    )
    print(f"  Days scored (OOS): {n_scored}")

    return {
        "latest_date":   latest_date,
        "latest_scores": latest_out,
        "latest_ranked": latest_ranked,
        "daily_df":      daily_df,
        "score_df":      score_df,
        "mu_df":         mu_df,
        "sigma_df":      sigma_df,
        "ranking_df":    ranking_df,
        "universe":      universe_name,
        "model_tickers": model_tickers,
        "ckpt_meta":     {
            "train_date":    ckpt.get("train_date", "?"),
            "best_val_loss": ckpt.get("best_val_loss", None),
        },
    }
