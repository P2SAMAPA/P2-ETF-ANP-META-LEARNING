"""meta_trainer.py — Meta-training for the Attentive Neural Process.

Run this manually (or via meta_train.yml) to train / retrain the ANP model.
Saves checkpoint to HuggingFace model repo.

Usage
-----
  python meta_trainer.py [--universe EQUITY_SECTORS|FI_COMMODITIES|COMBINED]
                         [--epochs 80]
                         [--episodes 3000]

Workflow
--------
  1. Load full dataset (2008 → META_TRAIN_END)
  2. Build feature matrix X, target matrix Y
  3. Sample N_EPISODES training episodes + validation episodes
  4. Train ANP with ELBO loss + KL annealing + early stopping
  5. Save best checkpoint to HF model repo
  6. Save training metadata (loss curves, config) alongside checkpoint
"""

from __future__ import annotations

import argparse
import io
import json
import os
import time

import numpy as np
import torch
import torch.optim as optim
from huggingface_hub import HfApi

import config
import data_manager
from model import AttentiveNeuralProcess, elbo_loss


def episodes_to_tensors(
    episodes: list[dict],
    device: torch.device,
) -> list[dict]:
    """Convert episode dicts of numpy arrays to tensors on device."""
    out = []
    for ep in episodes:
        out.append({k: torch.tensor(v, dtype=torch.float32).unsqueeze(0).to(device)
                    for k, v in ep.items()})
    return out


def make_batch(
    tensor_episodes: list[dict],
    indices: np.ndarray,
) -> dict:
    """Stack a batch of episode tensors. Each episode has shape (1, T, D)."""
    batch = {}
    for key in tensor_episodes[0].keys():
        batch[key] = torch.cat([tensor_episodes[i][key] for i in indices], dim=0)
    return batch


def train(
    universe_name: str,
    tickers: list[str],
    log_returns,
    macro_df,
    n_epochs: int,
    n_episodes: int,
    device: torch.device,
    token: str,
) -> None:
    print(f"\n{'='*60}")
    print(f"Meta-training ANP — Universe: {universe_name}")
    print(f"Device: {device} | Epochs: {n_epochs} | Episodes: {n_episodes}")
    print(f"Train data: 2008 → {config.META_TRAIN_END}")
    print(f"{'='*60}")

    # ── Feature engineering ───────────────────────────────────────────────────
    avail = [t for t in tickers if t in log_returns.columns]
    X, Y, dates = data_manager.build_features(log_returns, macro_df, avail)

    x_dim  = X.shape[1]
    y_dim  = Y.shape[1]
    print(f"Feature dim: {x_dim}  |  Target dim (n_etf): {y_dim}")
    print(f"Total data rows: {len(X)}  |  Date range: {dates[0].date()} → {dates[-1].date()}")

    # ── Sample episodes ───────────────────────────────────────────────────────
    rng = np.random.default_rng(42)

    print(f"\nSampling {n_episodes} train episodes (up to {config.META_TRAIN_END})...")
    train_eps = data_manager.make_episodes(
        X, Y, dates,
        n_episodes=n_episodes,
        context_size=config.CONTEXT_SIZE,
        query_size=config.QUERY_SIZE,
        date_end=config.META_TRAIN_END,
        date_start=None,
        rng=rng,
    )

    val_n = max(200, n_episodes // 10)
    print(f"Sampling {val_n} validation episodes ({config.META_VAL_START} → {config.META_TRAIN_END})...")
    val_eps = data_manager.make_episodes(
        X, Y, dates,
        n_episodes=val_n,
        context_size=config.CONTEXT_SIZE,
        query_size=config.QUERY_SIZE,
        date_end=config.META_TRAIN_END,
        date_start=config.META_VAL_START,
        rng=rng,
    )
    print(f"Train: {len(train_eps)} eps  |  Val: {len(val_eps)} eps")

    # Convert to tensors
    train_tensors = episodes_to_tensors(train_eps, device)
    val_tensors   = episodes_to_tensors(val_eps,   device)

    # ── Model ─────────────────────────────────────────────────────────────────
    model     = AttentiveNeuralProcess(x_dim=x_dim, y_dim=y_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.LR_SCHEDULER_STEP,
        gamma=config.LR_SCHEDULER_GAMMA,
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {n_params:,}")

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    patience_cnt  = 0
    history       = {"train_loss": [], "val_loss": [], "recon": [], "kl": []}

    idx_train = np.arange(len(train_tensors))

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()

        # KL weight annealing: ramp from 0 → 1 over KL_WARMUP_EPOCHS
        kl_weight = min(
            1.0,
            config.KL_WEIGHT_START
            + (config.KL_WEIGHT_END - config.KL_WEIGHT_START)
            * epoch / max(config.KL_WARMUP_EPOCHS, 1),
        )

        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        rng.shuffle(idx_train)
        train_losses, train_recons, train_kls = [], [], []

        for b_start in range(0, len(train_tensors), config.BATCH_SIZE):
            b_idx  = idx_train[b_start: b_start + config.BATCH_SIZE]
            batch  = make_batch(train_tensors, b_idx)

            out = model(
                context_x=batch["context_x"],
                context_y=batch["context_y"],
                query_x=batch["query_x"],
                target_y=batch["query_y"],
            )
            loss, info = elbo_loss(out, batch["query_y"], kl_weight=kl_weight)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            optimizer.step()

            train_losses.append(info["loss"])
            train_recons.append(info["recon"])
            train_kls.append(info["kl"])

        scheduler.step()

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        val_losses = []
        with torch.no_grad():
            val_idx = np.arange(len(val_tensors))
            for b_start in range(0, len(val_tensors), config.BATCH_SIZE):
                b_idx  = val_idx[b_start: b_start + config.BATCH_SIZE]
                batch  = make_batch(val_tensors, b_idx)
                out    = model(
                    context_x=batch["context_x"],
                    context_y=batch["context_y"],
                    query_x=batch["query_x"],
                    target_y=batch["query_y"],
                )
                loss, _ = elbo_loss(out, batch["query_y"], kl_weight=1.0)
                val_losses.append(loss.item())

        train_l = float(np.mean(train_losses))
        val_l   = float(np.mean(val_losses))
        recon_l = float(np.mean(train_recons))
        kl_l    = float(np.mean(train_kls))

        history["train_loss"].append(train_l)
        history["val_loss"].append(val_l)
        history["recon"].append(recon_l)
        history["kl"].append(kl_l)

        elapsed = time.time() - t0
        print(
            f"  Epoch {epoch:3d}/{n_epochs} | "
            f"train={train_l:.4f}  val={val_l:.4f}  "
            f"recon={recon_l:.4f}  kl={kl_l:.4f}  "
            f"kl_w={kl_weight:.2f}  lr={scheduler.get_last_lr()[0]:.2e}  "
            f"[{elapsed:.1f}s]"
        )

        # Early stopping
        if val_l < best_val_loss - 1e-4:
            best_val_loss = val_l
            patience_cnt  = 0
            # Save best checkpoint buffer
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"    ✅ New best val loss: {best_val_loss:.4f}")
        else:
            patience_cnt += 1
            if patience_cnt >= config.PATIENCE:
                print(f"\n  Early stopping at epoch {epoch} (patience={config.PATIENCE})")
                break

    # ── Save checkpoint ───────────────────────────────────────────────────────
    slug = universe_name.lower().replace("_", "-")

    ckpt = {
        "model_state_dict": best_state,
        "x_dim":            x_dim,
        "y_dim":            y_dim,
        "tickers":          avail,
        "universe":         universe_name,
        "train_date":       config.TODAY,
        "best_val_loss":    best_val_loss,
        "config": {
            "encoder_hidden":    config.ENCODER_HIDDEN,
            "latent_dim":        config.LATENT_DIM,
            "decoder_hidden":    config.DECODER_HIDDEN,
            "n_heads":           config.N_HEADS,
            "dropout":           config.DROPOUT,
            "context_size":      config.CONTEXT_SIZE,
            "query_size":        config.QUERY_SIZE,
            "n_episodes":        n_episodes,
            "n_epochs":          n_epochs,
            "learning_rate":     config.LEARNING_RATE,
            "meta_train_end":    config.META_TRAIN_END,
        },
    }

    buf = io.BytesIO()
    torch.save(ckpt, buf)
    buf.seek(0)

    meta = {
        "universe":      universe_name,
        "train_date":    config.TODAY,
        "best_val_loss": best_val_loss,
        "n_epochs_run":  len(history["train_loss"]),
        "x_dim":         x_dim,
        "y_dim":         y_dim,
        "tickers":       avail,
        "history":       history,
    }

    api = HfApi(token=token)
    api.create_repo(
        repo_id=config.HF_MODEL_REPO,
        repo_type="model",
        exist_ok=True,
        private=False,
    )

    ckpt_name = f"anp_checkpoint_{slug}.pt"
    meta_name = f"anp_meta_{slug}.json"

    api.upload_file(
        path_or_fileobj=buf,
        path_in_repo=ckpt_name,
        repo_id=config.HF_MODEL_REPO,
        repo_type="model",
        commit_message=f"ANP checkpoint {slug} — {config.TODAY} val={best_val_loss:.4f}",
    )
    api.upload_file(
        path_or_fileobj=io.BytesIO(json.dumps(meta, indent=2).encode()),
        path_in_repo=meta_name,
        repo_id=config.HF_MODEL_REPO,
        repo_type="model",
        commit_message=f"ANP metadata {slug} — {config.TODAY}",
    )

    print(f"\n  ✅ Checkpoint saved → {config.HF_MODEL_REPO}/{ckpt_name}")
    print(f"  ✅ Metadata  saved → {config.HF_MODEL_REPO}/{meta_name}")
    print(f"  Best val loss: {best_val_loss:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="ANP Meta-Trainer")
    parser.add_argument("--universe", default="ALL",
                        help="Universe to train (ALL / EQUITY_SECTORS / FI_COMMODITIES / COMBINED)")
    parser.add_argument("--epochs",   type=int, default=config.N_EPOCHS)
    parser.add_argument("--episodes", type=int, default=config.N_EPISODES)
    args = parser.parse_args()

    token = config.HF_TOKEN
    if not token:
        print("HF_TOKEN not set — aborting.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    log_returns, macro_df = data_manager.load_data(token=token)

    target = args.universe.upper()
    for universe_name, tickers in config.UNIVERSES.items():
        if target != "ALL" and universe_name != target:
            continue
        train(
            universe_name=universe_name,
            tickers=tickers,
            log_returns=log_returns,
            macro_df=macro_df,
            n_epochs=args.epochs,
            n_episodes=args.episodes,
            device=device,
            token=token,
        )

    print("\n✅ Meta-training complete.")


if __name__ == "__main__":
    main()
