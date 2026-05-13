"""config.py — Attentive Neural Process (ANP) Meta-Learning Engine.

Architecture overview
---------------------
  Encoder   : MLP per (x_i, y_i) context pair → deterministic path r_i
  Attention : Cross-attention over context      → aggregated r
  Latent    : MLP → mu_z, log_sigma_z → z ~ N(mu_z, sigma_z)  [stochastic path]
  Decoder   : MLP(x_query, r, z) → mu_pred, log_sigma_pred per ETF

Two workflows
-------------
  1. Meta-training  (meta_train.yml — manual, run weekly or on demand)
     - Samples ~N_EPISODES historical episodes from 2008-2022 data
     - Each episode: CONTEXT_SIZE context days + QUERY_SIZE query days
     - Trains encoder + attention + latent + decoder end-to-end
     - Saves checkpoint to HuggingFace model repo

  2. Daily inference (daily_run.yml — automated, Mon-Fri after market close)
     - Loads latest checkpoint from HuggingFace
     - Builds context from last CONTEXT_SIZE trading days
     - Runs single forward pass → mu_pred, sigma_pred per ETF
     - Scores and ranks ETFs → pushes results to HF dataset repo
"""

import os
from datetime import datetime

# ── HuggingFace ───────────────────────────────────────────────────────────────
HF_DATA_REPO   = "P2SAMAPA/fi-etf-macro-signal-master-data"
HF_DATA_FILE   = "master_data.parquet"
HF_MODEL_REPO  = "P2SAMAPA/p2-etf-anp-model"        # model checkpoint lives here
HF_OUTPUT_REPO = "P2SAMAPA/p2-etf-anp-results"      # daily score results
HF_TOKEN       = os.environ.get("HF_TOKEN", None)

# ── Universes ─────────────────────────────────────────────────────────────────
EQUITY_SECTORS_TICKERS = [
    "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV",
    "XLI", "XLY", "XLP", "XLU", "GDX", "XME",
    "IWF", "XSD", "XBI", "IWM",
]
FI_COMMODITIES_TICKERS = ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"]
COMBINED_TICKERS       = sorted(set(EQUITY_SECTORS_TICKERS + FI_COMMODITIES_TICKERS))

UNIVERSES = {
    "EQUITY_SECTORS":  EQUITY_SECTORS_TICKERS,
    "FI_COMMODITIES":  FI_COMMODITIES_TICKERS,
    "COMBINED":        COMBINED_TICKERS,
}

MACRO_COLS = ["VIX", "DXY", "T10Y2Y", "TBILL_3M"]

# ── Feature engineering ───────────────────────────────────────────────────────
LOOKBACK_LAGS      = [1, 5, 21]      # lagged return features per ETF
ROLLING_VOL_WINDOW = 21              # rolling volatility window
ROLLING_MOM_WINDOW = 63              # rolling momentum window

# ── Episode construction (meta-training) ──────────────────────────────────────
CONTEXT_SIZE       = 21    # number of context days per episode (1 month)
QUERY_SIZE         = 5     # number of query days per episode (1 week ahead)
N_EPISODES         = 3000  # total episodes sampled per universe per training run
META_TRAIN_END     = "2021-12-31"   # training cutoff (2022+ reserved for OOS)
META_VAL_START     = "2019-01-01"   # validation episodes drawn from 2019-2021
EPISODE_MIN_GAP    = 5    # min days between episode start dates (avoid overlap)

# ── Model architecture ────────────────────────────────────────────────────────
# Input feature dimension computed at runtime: n_etf * (n_lags + 2) + n_macro
# (returns at lags + rolling vol + rolling mom for each ETF, plus macro)
ENCODER_HIDDEN     = [128, 128]      # encoder MLP hidden dims
LATENT_DIM         = 64             # dimension of latent z
DECODER_HIDDEN     = [128, 128]      # decoder MLP hidden dims
N_HEADS            = 4              # attention heads in cross-attention
DROPOUT            = 0.10           # dropout rate

# ── Training hyper-parameters ─────────────────────────────────────────────────
LEARNING_RATE      = 3e-4
BATCH_SIZE         = 32             # episodes per batch
N_EPOCHS           = 80             # training epochs
KL_WEIGHT_START    = 0.0            # KL annealing: start weight
KL_WEIGHT_END      = 1.0            # KL annealing: end weight
KL_WARMUP_EPOCHS   = 20            # ramp KL weight over first N epochs
GRAD_CLIP          = 1.0            # gradient clipping norm
PATIENCE           = 15             # early stopping patience (val loss)
LR_SCHEDULER_STEP  = 20            # StepLR: decay LR every N epochs
LR_SCHEDULER_GAMMA = 0.5           # StepLR: multiply LR by this factor

# ── Inference / scoring ───────────────────────────────────────────────────────
N_LATENT_SAMPLES   = 50    # MC samples from z for uncertainty estimation
UNCERTAINTY_WT     = 1.0   # score = mu / (1 + UNCERTAINTY_WT * sigma)
CASH_THRESHOLD     = -0.30 # composite z-score below → recommend CASH
TOP_N              = 6

# ── OOS period ────────────────────────────────────────────────────────────────
OOS_START          = "2022-01-01"   # first date daily inference scores are published

# ── Checkpoint filenames on HF ────────────────────────────────────────────────
CKPT_FILENAME      = "anp_checkpoint.pt"    # latest model weights
CKPT_META_FILENAME = "anp_meta.json"        # training metadata

# ── Output ────────────────────────────────────────────────────────────────────
TODAY = datetime.now().strftime("%Y-%m-%d")
