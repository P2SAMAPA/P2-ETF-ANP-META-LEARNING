"""model.py — Attentive Neural Process (ANP) in PyTorch.

Architecture
------------
                    Context pairs (x_i, y_i)
                           ↓
              ┌────────────────────────────┐
              │  Deterministic encoder     │  h_det(x_i, y_i) → r_i   (per pair)
              │  Cross-attention           │  Q=x_query, K=V=r_context → r_agg
              └────────────────────────────┘
                           ↓ r_agg
              ┌────────────────────────────┐
              │  Latent encoder            │  h_lat(mean(r_i)) → mu_z, log_sigma_z
              │  Reparameterisation trick  │  z = mu_z + eps * sigma_z
              └────────────────────────────┘
                           ↓ z
              ┌────────────────────────────┐
              │  Decoder                   │  f(x_query, r_agg, z) → mu_y, log_sigma_y
              └────────────────────────────┘

Loss (ELBO + Fix1 + Fix2)
-----------
  L = Sharpe-weighted_recon + kl_weight * KL + RANK_LOSS_WT * ListNet_rank

  Fix 1 — Sharpe-weighted reconstruction:
    Per-ETF weight w_i = clip(1 + |Sharpe_i|, MIN, MAX)
    Upweights high-return episodes so model seeks high-return ETFs.

  Fix 2 — ListNet cross-sectional ranking loss:
    p_pred = softmax(mu_pred / RANK_TEMP_PRED)       across ETFs
    p_tgt  = softmax(target_mean / RANK_TEMP_TARGET) across ETFs (sharp)
    rank_loss = KL(p_tgt || p_pred)
    Directly teaches cross-sectional ranking, not just level accuracy.

  During training:
    p(z|context)         = N(mu_z_ctx, sigma_z_ctx)   ← only context
    q(z|context,target)  = N(mu_z_all, sigma_z_all)   ← context + target

  During inference: only p(z|context) is used → sample N_LATENT_SAMPLES times
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


# ── Utility: MLP builder ──────────────────────────────────────────────────────

def mlp(in_dim: int, hidden_dims: list[int], out_dim: int,
        dropout: float = 0.0) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden_dims:
        layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(dropout)]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


# ── Deterministic encoder ─────────────────────────────────────────────────────

class DeterministicEncoder(nn.Module):
    """Encodes each (x_i, y_i) context pair → representation r_i.

    Input:  (B, C, x_dim + y_dim)
    Output: (B, C, r_dim)
    """

    def __init__(self, x_dim: int, y_dim: int, r_dim: int,
                 hidden: list[int], dropout: float) -> None:
        super().__init__()
        self.net = mlp(x_dim + y_dim, hidden, r_dim, dropout=dropout)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x: (B, C, x_dim)  y: (B, C, y_dim)
        xy = torch.cat([x, y], dim=-1)   # (B, C, x_dim+y_dim)
        return self.net(xy)              # (B, C, r_dim)


# ── Cross-attention aggregator ────────────────────────────────────────────────

class CrossAttentionAggregator(nn.Module):
    """Cross-attention: Query=x_query, Key=Value=r_context.

    Allows the decoder to selectively attend to context representations
    that are most relevant for each query point.

    Input:
      query_x    : (B, Q, x_dim)   — query feature vectors
      context_r  : (B, C, r_dim)   — context representations
      context_x  : (B, C, x_dim)   — context feature vectors (used as keys)
    Output:
      r_agg      : (B, Q, r_dim)   — attended representation per query
    """

    def __init__(self, x_dim: int, r_dim: int, n_heads: int,
                 dropout: float) -> None:
        super().__init__()
        self.q_proj = nn.Linear(x_dim, r_dim)
        self.k_proj = nn.Linear(x_dim, r_dim)
        self.attn   = nn.MultiheadAttention(
            embed_dim=r_dim, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )

    def forward(
        self,
        query_x:   torch.Tensor,    # (B, Q, x_dim)
        context_r: torch.Tensor,    # (B, C, r_dim)
        context_x: torch.Tensor,    # (B, C, x_dim)
    ) -> torch.Tensor:
        Q = self.q_proj(query_x)    # (B, Q, r_dim)
        K = self.k_proj(context_x)  # (B, C, r_dim)
        V = context_r               # (B, C, r_dim)
        r_agg, _ = self.attn(Q, K, V)
        return r_agg                # (B, Q, r_dim)


# ── Latent encoder ────────────────────────────────────────────────────────────

class LatentEncoder(nn.Module):
    """Maps mean-pooled context representations → (mu_z, log_sigma_z).

    Input:  (B, C, r_dim)  → mean-pool → (B, r_dim)
    Output: mu_z (B, z_dim), log_sigma_z (B, z_dim)
    """

    def __init__(self, r_dim: int, z_dim: int,
                 hidden: list[int], dropout: float) -> None:
        super().__init__()
        self.net     = mlp(r_dim, hidden, z_dim * 2, dropout=dropout)
        self.z_dim   = z_dim

    def forward(self, r: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # r: (B, C, r_dim)
        r_mean  = r.mean(dim=1)          # (B, r_dim)
        out     = self.net(r_mean)       # (B, 2*z_dim)
        mu      = out[:, : self.z_dim]
        log_sig = out[:, self.z_dim :]
        log_sig = torch.clamp(log_sig, -4.0, 4.0)
        return mu, log_sig


# ── Decoder ───────────────────────────────────────────────────────────────────

class Decoder(nn.Module):
    """Decodes (x_query, r_agg, z) → predicted return distribution.

    Input:  x_query (B, Q, x_dim)
            r_agg   (B, Q, r_dim)   — from cross-attention
            z       (B, z_dim)      — latent sample
    Output: mu_y (B, Q, y_dim), log_sigma_y (B, Q, y_dim)
    """

    def __init__(self, x_dim: int, r_dim: int, z_dim: int,
                 y_dim: int, hidden: list[int], dropout: float) -> None:
        super().__init__()
        in_dim   = x_dim + r_dim + z_dim
        self.net = mlp(in_dim, hidden, y_dim * 2, dropout=dropout)
        self.y_dim = y_dim

    def forward(
        self,
        x:     torch.Tensor,   # (B, Q, x_dim)
        r_agg: torch.Tensor,   # (B, Q, r_dim)
        z:     torch.Tensor,   # (B, z_dim)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        Q    = x.size(1)
        z_ex = z.unsqueeze(1).expand(-1, Q, -1)        # (B, Q, z_dim)
        inp  = torch.cat([x, r_agg, z_ex], dim=-1)     # (B, Q, x+r+z)
        out  = self.net(inp)                            # (B, Q, 2*y_dim)
        mu      = out[..., : self.y_dim]
        log_sig = out[..., self.y_dim :]
        log_sig = torch.clamp(log_sig, -6.0, 2.0)
        return mu, log_sig


# ── Full ANP model ────────────────────────────────────────────────────────────

class AttentiveNeuralProcess(nn.Module):
    """Attentive Neural Process (Kim et al., 2019).

    Parameters
    ----------
    x_dim : int   — feature dimensionality (computed from data)
    y_dim : int   — output dimensionality (= n_etf in universe)
    """

    def __init__(self, x_dim: int, y_dim: int) -> None:
        super().__init__()
        r_dim = config.LATENT_DIM * 2   # representation dim = 2 × latent dim

        self.det_encoder = DeterministicEncoder(
            x_dim=x_dim, y_dim=y_dim, r_dim=r_dim,
            hidden=config.ENCODER_HIDDEN, dropout=config.DROPOUT,
        )
        self.lat_encoder = LatentEncoder(
            r_dim=r_dim, z_dim=config.LATENT_DIM,
            hidden=config.ENCODER_HIDDEN, dropout=config.DROPOUT,
        )
        self.cross_attn  = CrossAttentionAggregator(
            x_dim=x_dim, r_dim=r_dim,
            n_heads=config.N_HEADS, dropout=config.DROPOUT,
        )
        self.decoder     = Decoder(
            x_dim=x_dim, r_dim=r_dim, z_dim=config.LATENT_DIM,
            y_dim=y_dim, hidden=config.DECODER_HIDDEN, dropout=config.DROPOUT,
        )
        self.x_dim = x_dim
        self.y_dim = y_dim

    def forward(
        self,
        context_x:  torch.Tensor,   # (B, C, x_dim)
        context_y:  torch.Tensor,   # (B, C, y_dim)
        query_x:    torch.Tensor,   # (B, Q, x_dim)
        target_y:   torch.Tensor | None = None,  # (B, Q, y_dim) — None at inference
    ) -> dict:
        """Forward pass.

        Training:  target_y provided → compute ELBO loss
        Inference: target_y=None    → sample z from prior, return predictions
        """
        # ── Deterministic path ────────────────────────────────────────────────
        r_ctx  = self.det_encoder(context_x, context_y)    # (B, C, r_dim)
        r_agg  = self.cross_attn(query_x, r_ctx, context_x)# (B, Q, r_dim)

        # ── Latent path — prior from context only ─────────────────────────────
        mu_prior, log_sig_prior = self.lat_encoder(r_ctx)  # (B, z_dim)

        if target_y is not None:
            # Training: posterior from context + target
            # Concatenate context and target for posterior estimation
            all_x = torch.cat([context_x, query_x],   dim=1)  # (B, C+Q, x_dim)
            all_y = torch.cat([context_y, target_y],  dim=1)  # (B, C+Q, y_dim)
            r_all = self.det_encoder(all_x, all_y)             # (B, C+Q, r_dim)
            mu_post, log_sig_post = self.lat_encoder(r_all)

            # Sample z from posterior (reparameterisation)
            eps = torch.randn_like(mu_post)
            z   = mu_post + eps * torch.exp(0.5 * log_sig_post)

            # Decode
            mu_y, log_sig_y = self.decoder(query_x, r_agg, z)

            return {
                "mu_y":          mu_y,
                "log_sig_y":     log_sig_y,
                "mu_prior":      mu_prior,
                "log_sig_prior": log_sig_prior,
                "mu_post":       mu_post,
                "log_sig_post":  log_sig_post,
            }

        else:
            # Inference: sample z from prior only
            eps = torch.randn_like(mu_prior)
            z   = mu_prior + eps * torch.exp(0.5 * log_sig_prior)
            mu_y, log_sig_y = self.decoder(query_x, r_agg, z)
            return {
                "mu_y":      mu_y,
                "log_sig_y": log_sig_y,
            }

    @torch.no_grad()
    def predict(
        self,
        context_x: torch.Tensor,   # (1, C, x_dim)
        context_y: torch.Tensor,   # (1, C, y_dim)
        query_x:   torch.Tensor,   # (1, Q, x_dim)
        n_samples: int = config.N_LATENT_SAMPLES,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """MC inference: draw n_samples latent z, return mean & std of predictions.

        Returns
        -------
        mu_mean  : (1, Q, y_dim) — predictive mean
        mu_std   : (1, Q, y_dim) — predictive std (epistemic uncertainty)
        """
        self.eval()

        r_ctx = self.det_encoder(context_x, context_y)
        r_agg = self.cross_attn(query_x, r_ctx, context_x)
        mu_prior, log_sig_prior = self.lat_encoder(r_ctx)

        preds = []
        for _ in range(n_samples):
            eps = torch.randn_like(mu_prior)
            z   = mu_prior + eps * torch.exp(0.5 * log_sig_prior)
            mu_y, _ = self.decoder(query_x, r_agg, z)
            preds.append(mu_y)

        preds    = torch.stack(preds, dim=0)   # (n_samples, 1, Q, y_dim)
        mu_mean  = preds.mean(dim=0)           # (1, Q, y_dim)
        mu_std   = preds.std(dim=0)            # (1, Q, y_dim)
        return mu_mean, mu_std


# ── Fix 2: ListNet cross-sectional ranking loss ──────────────────────────────

def ranking_loss(
    mu_y: torch.Tensor,       # (B, Q, y_dim) — predicted returns
    target_y: torch.Tensor,   # (B, Q, y_dim) — realised returns
) -> torch.Tensor:
    """ListNet cross-sectional ranking loss (Cao et al., 2007).

    Treats each ETF (y_dim axis) as an item to rank. For each query day,
    computes a soft probability distribution over ETF ranks from both
    predicted and target returns, then minimises their KL divergence.

    This directly teaches the model to get the cross-sectional ETF rank
    right — not just minimise prediction error at the level of each ETF.

    L_rank = mean over batch and query days of:
        KL(p_target || p_pred)
      where:
        p_pred   = softmax(mu_mean / RANK_TEMP_PRED)    across ETFs
        p_target = softmax(tgt_mean / RANK_TEMP_TARGET) across ETFs

    RANK_TEMP_TARGET is very small (0.005) → near-one-hot distribution
    heavily concentrating weight on the best-performing ETF → pushes
    the model to identify the top ETF, not just get average returns right.
    """
    # Collapse Q dimension → mean return per ETF per batch element
    mu_mean  = mu_y.mean(dim=1)      # (B, y_dim)
    tgt_mean = target_y.mean(dim=1)  # (B, y_dim)

    # Soft rank distributions over ETFs
    p_pred   = torch.softmax(mu_mean  / config.RANK_TEMP_PRED,   dim=-1)  # (B, y_dim)
    p_target = torch.softmax(tgt_mean / config.RANK_TEMP_TARGET, dim=-1)  # (B, y_dim) — sharp

    # KL(p_target || p_pred) — target is the "truth" distribution
    # KL = sum p_target * log(p_target / p_pred)
    kl_rank = (p_target * (torch.log(p_target + 1e-8) - torch.log(p_pred + 1e-8)))
    return kl_rank.sum(dim=-1).mean()  # scalar


# ── ELBO loss with Fix 1 (Sharpe weighting) and Fix 2 (ranking) ───────────────

def elbo_loss(
    out: dict,
    target_y: torch.Tensor,
    kl_weight: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    """ELBO + Sharpe-weighted reconstruction (Fix 1) + ListNet ranking (Fix 2).

    Total loss:
      L = sharpe_weighted_recon
        + kl_weight * KL[q(z|ctx+tgt) || p(z|ctx)]
        + RANK_LOSS_WT * ListNet_ranking_loss

    Fix 1 — Sharpe-weighted reconstruction
    ---------------------------------------
    Per-ETF Sharpe ratio computed from the query target batch:
      sharpe_i = mean(y_i) / (std(y_i) + eps)         over Q query days
      weight_i = clip(1 + |sharpe_i|, MIN, MAX)        in [1.0, 3.0]

    High-Sharpe ETFs (high return relative to their own volatility) get
    up to 3× more gradient weight. This biases the model to predict
    high-return stable ETFs accurately — directly addressing the FI bias.

    Fix 2 — ListNet cross-sectional ranking loss
    --------------------------------------------
    See ranking_loss() above. Added with weight RANK_LOSS_WT = 0.30.
    """
    mu_y      = out["mu_y"]       # (B, Q, y_dim)
    log_sig_y = out["log_sig_y"]
    sig_y     = torch.exp(0.5 * log_sig_y) + 1e-6

    # ── Fix 1: Sharpe-weighted reconstruction ─────────────────────────────────
    if config.SHARPE_WEIGHT_RECON:
        # Per-ETF Sharpe over the Q query days in this batch
        t_mean   = target_y.mean(dim=1, keepdim=True)         # (B, 1, y_dim)
        t_std    = target_y.std(dim=1, keepdim=True) + 1e-6   # (B, 1, y_dim)
        sharpe   = (t_mean / t_std).abs()                     # (B, 1, y_dim)
        # Weight: 1 + |Sharpe|, clipped to [MIN, MAX]
        w = torch.clamp(
            1.0 + sharpe,
            config.SHARPE_WEIGHT_MIN,
            config.SHARPE_WEIGHT_MAX,
        )                                                      # (B, 1, y_dim)
    else:
        w = 1.0

    # Reconstruction: Sharpe-weighted negative Gaussian log-likelihood
    dist  = torch.distributions.Normal(mu_y, sig_y)
    log_p = dist.log_prob(target_y)      # (B, Q, y_dim)
    recon = -(log_p * w).mean()          # scalar — weighted mean

    # ── KL divergence (analytic, two Gaussians) ───────────────────────────────
    mu_q  = out["mu_post"]
    sig_q = torch.exp(0.5 * out["log_sig_post"]) + 1e-6
    mu_p  = out["mu_prior"]
    sig_p = torch.exp(0.5 * out["log_sig_prior"]) + 1e-6

    kl = torch.distributions.kl_divergence(
        torch.distributions.Normal(mu_q, sig_q),
        torch.distributions.Normal(mu_p, sig_p),
    ).mean()

    # ── Fix 2: ListNet cross-sectional ranking loss ───────────────────────────
    rank_l = ranking_loss(mu_y, target_y)

    # ── Total loss ────────────────────────────────────────────────────────────
    loss = recon + kl_weight * kl + config.RANK_LOSS_WT * rank_l

    return loss, {
        "loss":      loss.item(),
        "recon":     recon.item(),
        "kl":        kl.item(),
        "rank_loss": rank_l.item(),
    }
