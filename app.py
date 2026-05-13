"""app.py — Attentive Neural Process · Meta-Learning Engine · Streamlit Dashboard."""

from __future__ import annotations

import os
from io import StringIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

import config
from us_calendar import next_trading_day

st.set_page_config(
    page_title="ANP Meta-Learning · P2Quant",
    layout="wide",
    page_icon="🧠",
)

HF_TOKEN = os.environ.get("HF_TOKEN")
BASE_RAW = f"https://huggingface.co/datasets/{config.HF_OUTPUT_REPO}/resolve/main"
BASE_API = f"https://huggingface.co/api/datasets/{config.HF_OUTPUT_REPO}/tree/main"
HEADERS  = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

PALETTE = [
    "#1B4F8A", "#27AE60", "#E74C3C", "#F39C12",
    "#8E44AD", "#148F77", "#CA6F1E", "#2471A3",
    "#CB4335", "#1A5276", "#117A65", "#B7950B",
    "#884EA0", "#1F618D", "#B9770E", "#148F77",
    "#922B21", "#1A5276",
]

def score_colour(v: float) -> str:
    if v >= 0.5:  return "#1D9E75"
    if v >= 0.0:  return "#82C3A9"
    if v >= -0.5: return "#F0A07A"
    return "#E74C3C"

def sigma_colour(v: float, vmax: float) -> str:
    frac = v / (vmax + 1e-8)
    if frac < 0.33: return "#1D9E75"
    if frac < 0.66: return "#F39C12"
    return "#E74C3C"

def fmt(v: float, d: int = 4) -> str:
    return f"{v:+.{d}f}"


# ── Loaders ───────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner="Loading ANP results…")
def load_json(universe: str) -> dict | None:
    slug = universe.lower().replace("_", "-")
    try:
        r = requests.get(BASE_API, headers=HEADERS, timeout=30)
        if r.status_code != 200:
            return None
        files   = sorted(f["path"] for f in r.json() if f["path"].endswith(".json"))
        matches = [f for f in files if f"_{slug}.json" in f]
        if not matches:
            return None
        resp = requests.get(f"{BASE_RAW}/{matches[-1]}", headers=HEADERS, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner="Loading history…")
def load_csv(filename: str) -> pd.DataFrame | None:
    try:
        r = requests.get(f"{BASE_RAW}/{filename}", headers=HEADERS, timeout=60)
        if r.status_code != 200:
            return None
        df = pd.read_csv(StringIO(r.text), index_col=0, parse_dates=True)
        return df if not df.empty else None
    except Exception:
        return None


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    universe = st.selectbox("Universe", list(config.UNIVERSES.keys()))
    st.divider()
    st.markdown(f"**Context window:** {config.CONTEXT_SIZE} days")
    st.markdown(f"**Latent dim:** {config.LATENT_DIM}")
    st.markdown(f"**Attention heads:** {config.N_HEADS}")
    st.markdown(f"**MC samples (σ):** {config.N_LATENT_SAMPLES}")
    st.markdown(f"**Uncertainty weight:** {config.UNCERTAINTY_WT}")
    st.markdown(f"**CASH threshold:** {config.CASH_THRESHOLD}")
    st.markdown(f"**OOS from:** {config.OOS_START}")
    st.markdown(f"**Next trading day:** {next_trading_day()}")
    st.divider()
    st.markdown("**Score formula:**")
    st.code(
        "raw  = mu_pred / (1 + σ_pred)\n"
        "score = cross_sectional_zscore(raw)",
        language="python",
    )
    st.divider()
    st.markdown("**Workflows:**")
    st.markdown("🧠 `meta_train.yml` — weekly manual retraining")
    st.markdown("📅 `daily_run.yml` — daily auto inference")
    if st.button("🔄 Refresh"):
        st.cache_data.clear()
        st.rerun()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🧠 Attentive Neural Process · Meta-Learning Engine")
st.caption(
    f"21-day context window → cross-attention → latent z ~ N(μ,σ) → "
    f"{config.N_LATENT_SAMPLES}-sample MC inference · "
    "Score = μ_pred / (1 + σ_pred) · Uncertainty = built-in CASH signal"
)

slug       = universe.lower().replace("_", "-")
data       = load_json(universe)
daily_df   = load_csv(f"daily_{slug}.csv")
score_df   = load_csv(f"scores_{slug}.csv")
mu_df      = load_csv(f"mu_{slug}.csv")
sigma_df   = load_csv(f"sigma_{slug}.csv")
ranking_df = load_csv(f"rankings_{slug}.csv")

if data is None:
    st.warning(
        "⚠️ No results found. Run `meta_train.yml` first to train the model, "
        "then `daily_run.yml` to generate scores."
    )
    st.stop()

latest_scores = data.get("latest_scores", {})
latest_ranked = data.get("latest_ranked", [])
latest_date   = data.get("latest_date", "?")
run_date      = data.get("run_date", "?")
ckpt_meta     = data.get("ckpt_meta", {})
cfg           = data.get("config", {})

# ── KPI row ───────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("Run Date",      run_date)
k2.metric("Latest Date",   latest_date)
k3.metric("Model Trained", ckpt_meta.get("train_date", "?"))
k4.metric("Val Loss",      f"{ckpt_meta.get('best_val_loss', 0):.6f}"
          if ckpt_meta.get("best_val_loss") else "?")

if latest_ranked:
    top       = latest_ranked[0]
    cash_flag = top.get("composite_score", 0) < config.CASH_THRESHOLD
    sigmas    = [v.get("sigma_pred", 0) for v in latest_scores.values()]
    mean_sig  = float(np.mean(sigmas)) if sigmas else 0.0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🏆 Top Pick",
              "CASH" if cash_flag else top["ticker"])
    m2.metric("Top Score",       fmt(top.get("composite_score", 0)))
    m3.metric("Mean σ (uncertainty)", f"{mean_sig:.5f}",
              help="Higher σ = model more uncertain about predictions")
    m4.metric("CASH Signal",     "Yes ⚠️" if cash_flag else "No ✅")

st.divider()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Rankings & Scores",
    "🔮 Predictions (μ)",
    "📊 Uncertainty (σ)",
    "📈 Score History",
    "📋 Full Table",
])

# ─────────────────────────────────────────────────────────────────────────────
# Tab 1 — Rankings & Scores
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader(f"ANP Rankings as of {latest_date}")

    tickers_r = [r["ticker"] for r in latest_ranked]
    scores_r  = [r.get("composite_score", 0) for r in latest_ranked]
    mu_r      = [r.get("mu_pred", 0)         for r in latest_ranked]
    sigma_r   = [r.get("sigma_pred", 0)      for r in latest_ranked]
    colours_r = [score_colour(s) for s in scores_r]
    max_sigma = max(sigma_r) if sigma_r else 1.0

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("**Composite Score = μ / (1 + σ), z-scored**")
        fig = go.Figure(go.Bar(
            y=tickers_r, x=scores_r, orientation="h",
            marker_color=colours_r,
            text=[fmt(s) for s in scores_r],
            textposition="outside",
        ))
        fig.add_vline(x=0, line_dash="dot", line_color="gray")
        fig.update_layout(
            title="ANP composite score (uncertainty-discounted)",
            xaxis_title="z-score",
            yaxis=dict(autorange="reversed"),
            height=max(300, len(tickers_r) * 30),
            margin=dict(t=50, b=20, l=60, r=80),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True, key="rank_bar")

    with col_r:
        st.markdown("**Predictive Mean μ vs Uncertainty σ**")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=mu_r, y=sigma_r,
            mode="markers+text",
            text=tickers_r,
            textposition="top center",
            marker=dict(
                size=12,
                color=scores_r,
                colorscale="RdYlGn",
                colorbar=dict(title="Score"),
                showscale=True,
            ),
        ))
        fig2.add_vline(x=0, line_dash="dot", line_color="gray")
        fig2.update_layout(
            title="μ vs σ — top-right quadrant = high return, low uncertainty",
            xaxis_title="Predicted return μ",
            yaxis_title="Uncertainty σ",
            height=max(300, len(tickers_r) * 30),
            margin=dict(t=50, b=40, l=60, r=80),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig2, use_container_width=True, key="mu_sigma_scatter")

    # Top-N cards
    st.markdown(f"### 🎯 Top {config.TOP_N} for {next_trading_day()}")
    cols = st.columns(config.TOP_N)
    for i, row in enumerate(latest_ranked[: config.TOP_N]):
        with cols[i]:
            sc  = row.get("composite_score", 0)
            mu  = row.get("mu_pred", 0)
            sig = row.get("sigma_pred", 0)
            bg  = score_colour(sc)
            sc_col = sigma_colour(sig, max_sigma)
            st.markdown(
                f"**#{i+1} {row['ticker']}**\n\n"
                f"Score: `{fmt(sc)}`\n\n"
                f"μ pred: `{fmt(mu, 6)}`\n\n"
                f'σ uncert: <span style="color:{sc_col}">**{sig:.5f}**</span>\n\n'
                f'<span style="background:{bg};color:white;padding:2px 8px;'
                f'border-radius:8px;font-size:11px">Rank #{row.get("rank", i+1)}</span>',
                unsafe_allow_html=True,
            )

# ─────────────────────────────────────────────────────────────────────────────
# Tab 2 — Predictions (μ)
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("ANP Predicted Return μ Over Time")
    st.caption(
        "μ_pred = mean of 50 MC samples from the latent z distribution. "
        "Represents the ANP's best estimate of next-day return given the "
        "21-day context window."
    )

    if mu_df is not None:
        etf_cols = [c for c in mu_df.columns if c in config.UNIVERSES[universe]]
        selected = st.multiselect(
            "Select ETFs", etf_cols, default=etf_cols[:6], key="mu_sel"
        )
        period = st.radio(
            "Period", ["Last 2 years", "Last 3 years", "Full OOS"],
            horizontal=True, key="mu_period"
        )
        df_mu = mu_df.copy()
        if period == "Last 2 years":
            df_mu = df_mu[df_mu.index >= "2024-01-01"]
        elif period == "Last 3 years":
            df_mu = df_mu[df_mu.index >= "2023-01-01"]

        if selected:
            fig_mu = go.Figure()
            for i, tkr in enumerate(selected):
                if tkr in df_mu.columns:
                    fig_mu.add_trace(go.Scatter(
                        x=df_mu.index, y=df_mu[tkr],
                        mode="lines", name=tkr,
                        line=dict(width=1.4, color=PALETTE[i % len(PALETTE)]),
                    ))
            fig_mu.add_hline(y=0, line_dash="dot", line_color="gray")
            fig_mu.update_layout(
                title="ANP predicted return μ per ETF (daily)",
                yaxis_title="μ_pred (log return)",
                height=400,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig_mu, use_container_width=True, key="mu_ts")

        # μ heatmap
        recent_mu = mu_df[[c for c in mu_df.columns
                           if c in config.UNIVERSES[universe]]].tail(126)
        fig_mh = go.Figure(go.Heatmap(
            z=recent_mu.values.T,
            x=recent_mu.index.strftime("%Y-%m-%d"),
            y=list(recent_mu.columns),
            colorscale="RdYlGn", zmid=0,
            colorbar=dict(title="μ_pred"),
        ))
        fig_mh.update_layout(
            title="μ_pred Heatmap — last 126 days (green=bullish, red=bearish)",
            height=max(300, len(recent_mu.columns) * 22 + 80),
            margin=dict(t=40, b=60, l=60, r=20),
            xaxis=dict(tickangle=-45, nticks=10),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_mh, use_container_width=True, key="mu_heat")
    else:
        st.info("No μ history found.")

# ─────────────────────────────────────────────────────────────────────────────
# Tab 3 — Uncertainty (σ)
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Predictive Uncertainty σ Over Time")
    st.caption(
        "σ = standard deviation of 50 MC samples of μ_pred. "
        "High σ = the model is uncertain about this ETF given the current "
        "21-day context — the latent z draws produce very different predictions. "
        "**High universe-mean σ = consider increasing CASH allocation.**"
    )

    if sigma_df is not None:
        etf_cols_s = [c for c in sigma_df.columns if c in config.UNIVERSES[universe]]
        selected_s = st.multiselect(
            "Select ETFs", etf_cols_s, default=etf_cols_s[:6], key="sig_sel"
        )
        period_s = st.radio(
            "Period", ["Last 2 years", "Last 3 years", "Full OOS"],
            horizontal=True, key="sig_period"
        )
        df_sig = sigma_df.copy()
        if period_s == "Last 2 years":
            df_sig = df_sig[df_sig.index >= "2024-01-01"]
        elif period_s == "Last 3 years":
            df_sig = df_sig[df_sig.index >= "2023-01-01"]

        if selected_s:
            fig_sig = go.Figure()
            for i, tkr in enumerate(selected_s):
                if tkr in df_sig.columns:
                    fig_sig.add_trace(go.Scatter(
                        x=df_sig.index, y=df_sig[tkr],
                        mode="lines", name=tkr,
                        line=dict(width=1.4, color=PALETTE[i % len(PALETTE)]),
                    ))
            fig_sig.update_layout(
                title="Predictive uncertainty σ per ETF (higher = more uncertain)",
                yaxis_title="σ",
                height=400,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig_sig, use_container_width=True, key="sig_ts")

        # Universe mean σ — uncertainty regime gauge
        all_sig_cols = [c for c in sigma_df.columns if c in config.UNIVERSES[universe]]
        mean_sig_ts  = sigma_df[all_sig_cols].mean(axis=1)
        roll_mean    = mean_sig_ts.rolling(21).mean()

        fig_usig = go.Figure()
        fig_usig.add_trace(go.Scatter(
            x=mean_sig_ts.index, y=mean_sig_ts.values,
            mode="lines", name="Mean σ",
            line=dict(color="#8E44AD", width=1.5),
            fill="tozeroy", fillcolor="rgba(142,68,173,0.07)",
        ))
        fig_usig.add_trace(go.Scatter(
            x=roll_mean.index, y=roll_mean.values,
            mode="lines", name="21d rolling mean",
            line=dict(color="#E74C3C", width=1.5, dash="dot"),
        ))
        fig_usig.update_layout(
            title="Universe mean σ — ANP uncertainty regime gauge",
            yaxis_title="Mean σ",
            height=320,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_usig, use_container_width=True, key="sig_univ")

        # σ heatmap
        recent_sig = sigma_df[all_sig_cols].tail(126)
        fig_sh = go.Figure(go.Heatmap(
            z=recent_sig.values.T,
            x=recent_sig.index.strftime("%Y-%m-%d"),
            y=list(recent_sig.columns),
            colorscale="YlOrRd",
            colorbar=dict(title="σ"),
        ))
        fig_sh.update_layout(
            title="σ Heatmap — last 126 days (dark = high uncertainty → discount score)",
            height=max(300, len(recent_sig.columns) * 22 + 80),
            margin=dict(t=40, b=60, l=60, r=20),
            xaxis=dict(tickangle=-45, nticks=10),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_sh, use_container_width=True, key="sig_heat")
    else:
        st.info("No σ history found.")

# ─────────────────────────────────────────────────────────────────────────────
# Tab 4 — Score History
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("Composite Score History")

    if score_df is not None:
        etf_cols_c = [c for c in score_df.columns if c in config.UNIVERSES[universe]]
        selected_c = st.multiselect(
            "Select ETFs", etf_cols_c, default=etf_cols_c[:6], key="score_sel"
        )
        period_c = st.radio(
            "Period", ["Last 2 years", "Last 3 years", "Full OOS"],
            horizontal=True, key="score_period"
        )
        df_sc = score_df.copy()
        if period_c == "Last 2 years":
            df_sc = df_sc[df_sc.index >= "2024-01-01"]
        elif period_c == "Last 3 years":
            df_sc = df_sc[df_sc.index >= "2023-01-01"]

        if selected_c:
            fig_sc = go.Figure()
            for i, tkr in enumerate(selected_c):
                if tkr in df_sc.columns:
                    fig_sc.add_trace(go.Scatter(
                        x=df_sc.index, y=df_sc[tkr],
                        mode="lines", name=tkr,
                        line=dict(width=1.4, color=PALETTE[i % len(PALETTE)]),
                    ))
            fig_sc.add_hline(y=0, line_dash="dot", line_color="gray")
            fig_sc.update_layout(
                title="ANP composite score (cross-sectional z-score)",
                yaxis_title="Score",
                height=400,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig_sc, use_container_width=True, key="score_ts")

        # Score heatmap
        recent_sc = score_df[etf_cols_c].tail(252)
        fig_sch = go.Figure(go.Heatmap(
            z=recent_sc.values.T,
            x=recent_sc.index.strftime("%Y-%m-%d"),
            y=list(recent_sc.columns),
            colorscale="RdYlGn", zmid=0,
            colorbar=dict(title="Score"),
        ))
        fig_sch.update_layout(
            title="Score Heatmap — last 252 days",
            height=max(300, len(recent_sc.columns) * 22 + 80),
            margin=dict(t=40, b=60, l=60, r=20),
            xaxis=dict(tickangle=-45, nticks=12),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_sch, use_container_width=True, key="score_heat")

        # Top-pick frequency
        if daily_df is not None and "top_ticker" in daily_df.columns:
            picks = daily_df["top_ticker"].value_counts()
            fig_freq = go.Figure(go.Bar(
                x=picks.index, y=picks.values,
                marker_color="#1B4F8A",
                text=picks.values, textposition="outside",
            ))
            fig_freq.update_layout(
                title="Top-Pick Frequency (full OOS)",
                yaxis_title="Days as #1 ANP pick",
                height=280,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_freq, use_container_width=True, key="pick_freq")
    else:
        st.info("No score history found.")

# ─────────────────────────────────────────────────────────────────────────────
# Tab 5 — Full Table
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    st.subheader(f"Full ANP Predictions — {latest_date}")

    if latest_ranked:
        rows = []
        for i, row in enumerate(latest_ranked):
            rows.append({
                "Rank":            i + 1,
                "Ticker":          row["ticker"],
                "Composite Score": fmt(row.get("composite_score", 0)),
                "μ_pred":          fmt(row.get("mu_pred", 0), 6),
                "σ_pred":          f"{row.get('sigma_pred', 0):.6f}",
                "μ/σ ratio":       fmt(row.get("mu_pred", 0) /
                                       max(row.get("sigma_pred", 1e-6), 1e-6), 4),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True,
                     hide_index=True, height=600)

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Model Checkpoint Info**")
        st.json(ckpt_meta)
    with c2:
        st.markdown("**Engine Configuration**")
        st.json(cfg)

    if daily_df is not None:
        st.divider()
        st.markdown("**Daily summary (last 20 days)**")
        st.dataframe(daily_df.tail(20), use_container_width=True)

    st.divider()
    st.caption(
        f"P2Quant ANP Engine · Run: {run_date} · "
        f"Attentive Neural Process · Kim et al. (2019) · "
        f"Context={config.CONTEXT_SIZE}d · z_dim={config.LATENT_DIM} · "
        f"{config.N_LATENT_SAMPLES} MC samples · Data: {config.HF_DATA_REPO}"
    )
