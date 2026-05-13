# 🧠 P2-ETF-ANP-META-LEARNING

**P2Quant Engine** · Attentive Neural Process · Meta-Learning · ETF Ranking

[![ANP Daily Inference](https://github.com/P2SAMAPA/P2-ETF-ANP-META-LEARNING/actions/workflows/daily_run.yml/badge.svg)](https://github.com/P2SAMAPA/P2-ETF-ANP-META-LEARNING/actions/workflows/daily_run.yml)
[![ANP Meta-Training](https://github.com/P2SAMAPA/P2-ETF-ANP-META-LEARNING/actions/workflows/meta_train.yml/badge.svg)](https://github.com/P2SAMAPA/P2-ETF-ANP-META-LEARNING/actions/workflows/meta_train.yml)

---

## What Is This?

This engine applies **Attentive Neural Processes (ANP)** to ETF ranking.
It is the only engine in the P2Quant suite that **adapts at inference time**
without retraining — given the last 21 days of market context, it predicts
tomorrow's ETF returns and quantifies its own uncertainty.

Every other engine extrapolates from full historical patterns.
The ANP asks: *"What do the last 21 days look like, and what happened next
in historically similar episodes?"* — weighted by cross-attention.

---

## Two Workflows

### 1. Meta-Training (`meta_train.yml`) — Manual, run weekly

```
GitHub → Actions → "ANP Meta-Training (Manual)" → Run workflow
```

- Samples 3,000 historical episodes from 2008–2021 training data
- Each episode: 21 context days + 5 query days
- Trains encoder + cross-attention + latent + decoder end-to-end with ELBO
- KL annealing + early stopping + StepLR decay
- Saves checkpoint to `P2SAMAPA/p2-etf-anp-model` on HuggingFace

**First-time setup:** Run meta-training before the daily workflow will work.

### 2. Daily Inference (`daily_run.yml`) — Automated, Mon-Fri 22:15 UTC

- Loads latest checkpoint from HF model repo (no retraining)
- Builds 21-day context window from today's market data
- Runs 50 MC samples from latent z → μ_pred, σ_pred per ETF
- Pushes scores/rankings to `P2SAMAPA/p2-etf-anp-results`

---

## Architecture (Attentive Neural Process)

```
Context pairs (x_i, y_i) for i = t-21 ... t-1
        ↓
   Deterministic Encoder: MLP(x_i ⊕ y_i) → r_i          per pair
        ↓
   Cross-Attention: Q=x_today, K=r_context, V=r_context → r_agg
        ↓                   ↑ which past days matter most?
   Latent Encoder: MLP(mean(r_i)) → μ_z, log σ_z
        ↓
   z ~ N(μ_z, σ_z)    [50 Monte Carlo samples at inference]
        ↓
   Decoder: MLP(x_today ⊕ r_agg ⊕ z) → μ_pred, log σ_pred  per ETF
```

---

## Scoring Formula

```
raw_score(i)  = μ_pred(i) / (1 + σ_pred(i))
composite(i)  = cross_sectional_zscore(raw_score(i))
```

- **μ_pred** = mean of 50 MC samples → best estimate of next-day return
- **σ_pred** = std of 50 MC samples → **epistemic uncertainty**
- High σ discounts the score: uncertain ETFs are penalised even if μ > 0
- Universe-mean σ spike → potential CASH signal across all ETFs

If `composite_score < CASH_THRESHOLD (−0.30)` → recommend CASH.

---

## Training Loss (ELBO)

```
L = -E_q[log p(y_query | x_query, context)]   ← reconstruction
  + β * KL[q(z | ctx+target) || p(z | ctx)]   ← regularisation

β annealed: 0 → 1 over first 20 epochs (KL warmup)
```

During training: posterior q(z) uses context + target (knows the answer).
During inference: prior p(z) uses context only (predicts from context alone).

---

## Model Hyper-parameters

| Parameter | Value | Meaning |
|---|---|---|
| `CONTEXT_SIZE` | 21 | Days of context per episode (1 month) |
| `QUERY_SIZE` | 5 | Days to predict per episode |
| `LATENT_DIM` | 64 | Dimension of latent z |
| `ENCODER_HIDDEN` | [128, 128] | Encoder MLP hidden layers |
| `DECODER_HIDDEN` | [128, 128] | Decoder MLP hidden layers |
| `N_HEADS` | 4 | Cross-attention heads |
| `N_EPISODES` | 3,000 | Episodes sampled per training run |
| `N_EPOCHS` | 80 | Maximum training epochs |
| `KL_WARMUP_EPOCHS` | 20 | Epochs to ramp KL weight 0→1 |
| `N_LATENT_SAMPLES` | 50 | MC samples at inference |

---

## Data Split

| Period | Dates | Role |
|---|---|---|
| **Meta-train** | 2008-01-01 → 2018-12-31 | Episode sampling for training |
| **Meta-val** | 2019-01-01 → 2021-12-31 | Validation episodes (incl. COVID) |
| **OOS** | 2022-01-01 → present | Daily inference scores published |

---

## Universes

| Universe | Tickers |
|---|---|
| EQUITY_SECTORS | SPY QQQ XLK XLF XLE XLV XLI XLY XLP XLU GDX XME IWF XSD XBI IWM |
| FI_COMMODITIES | TLT VCIT LQD HYG VNQ GLD SLV |
| COMBINED | All above |

---

## HuggingFace Repos

| Repo | Type | Content |
|---|---|---|
| `P2SAMAPA/p2-etf-anp-model` | Model | Trained checkpoints (`.pt`) + metadata |
| `P2SAMAPA/p2-etf-anp-results` | Dataset | Daily scores, μ, σ, rankings |

---

## Output Files (per universe, daily dataset repo)

| File | Content |
|---|---|
| `anp_YYYY-MM-DD_{slug}.json` | Latest μ, σ, scores, rankings, config |
| `daily_{slug}.csv` | Top pick, CASH flag, mean μ, mean σ per day |
| `scores_{slug}.csv` | Full composite score history |
| `mu_{slug}.csv` | Full μ_pred history |
| `sigma_{slug}.csv` | Full σ_pred history |
| `rankings_{slug}.csv` | Full rank history |

---

## Streamlit Dashboard — 5 Tabs

1. **Rankings & Scores** — composite score bar, μ-vs-σ scatter plot, top-N cards
2. **Predictions (μ)** — μ time-series, μ heatmap (green=bullish, red=bearish)
3. **Uncertainty (σ)** — σ time-series, universe σ gauge, σ heatmap
4. **Score History** — composite score time-series + heatmap, top-pick frequency
5. **Full Table** — all μ, σ, μ/σ ratios, checkpoint info, daily summary

---

## References

- Kim, H. et al. (2019). *Attentive Neural Processes.* ICLR 2019.
- Garnelo, M. et al. (2018). *Neural Processes.* ICML Workshop.
- Garnelo, M. et al. (2018). *Conditional Neural Processes.* ICML.
- Kingma, D. & Welling, M. (2013). *Auto-Encoding Variational Bayes.* ICLR.

---

*P2Quant Engine Suite · Built by P2SAMAPA*
