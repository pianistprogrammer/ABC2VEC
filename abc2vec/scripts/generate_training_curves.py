#!/usr/bin/env python
"""
Generate high-quality training curves for ABC2Vec paper.

Loads real training history from training_history.json (written by run_training.py)
and produces publication-quality figures. No synthetic data is used.

Usage:
    python generate_training_curves.py
    python generate_training_curves.py --history ./checkpoints/training_history.json
    python generate_training_curves.py --output_dir ./figures
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_history(history_path: Path) -> dict:
    """Load and parse training_history.json into numpy arrays."""
    with open(history_path) as f:
        raw = json.load(f)

    train = raw["train_losses"]   # list of {step, mmm, scl, ti, total}
    val   = raw["val_losses"]     # list of {step, mmm, ti, total}  (no scl)

    def extract(records, key):
        return (
            np.array([r["step"] for r in records]),
            np.array([r[key]    for r in records]),
        )

    tr_steps,  tr_total = extract(train, "total")
    _,         tr_mmm   = extract(train, "mmm")
    _,         tr_ti    = extract(train, "ti")
    _,         tr_scl   = extract(train, "scl")

    val_steps, val_total = extract(val, "total")
    _,         val_mmm   = extract(val, "mmm")
    _,         val_ti    = extract(val, "ti")

    return {
        "tr_steps":  tr_steps,
        "tr_total":  tr_total,
        "tr_mmm":    tr_mmm,
        "tr_ti":     tr_ti,
        "tr_scl":    tr_scl,
        "val_steps": val_steps,
        "val_total": val_total,
        "val_mmm":   val_mmm,
        "val_ti":    val_ti,
        "total_steps":    raw.get("total_steps"),
        "best_val_loss":  raw.get("best_val_loss"),
    }


# ---------------------------------------------------------------------------
# Smoothing helper
# ---------------------------------------------------------------------------

def smooth(y: np.ndarray, window: int = 30) -> np.ndarray:
    """Apply a centred moving-average with the given window."""
    box = np.ones(window) / window
    return np.convolve(y, box, mode="same")


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

COLORS = {
    "train": "#1f77b4",   # blue
    "val":   "#d62728",   # red
    "mmm":   "#1f77b4",   # blue
    "ti":    "#2ca02c",   # green
    "scl":   "#ff7f0e",   # orange
}


def _style_ax(ax, title: str):
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    ax.xaxis.set_major_locator(MaxNLocator(6))


def plot_standard(d: dict, output_dir: Path):
    """2×2 grid, raw traces, train + val where available."""
    fig = plt.figure(figsize=(12, 8))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.28)

    # ── Total loss ──────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(d["tr_steps"],  d["tr_total"],  color=COLORS["train"], lw=1.2,
             label="Train", alpha=0.85)
    ax1.plot(d["val_steps"], d["val_total"], color=COLORS["val"],   lw=1.5,
             label="Val",   alpha=0.95, marker="o", markersize=3)
    ax1.legend(loc="upper right", frameon=True)
    _style_ax(ax1, "Total Loss")

    # ── MMM loss ─────────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(d["tr_steps"],  d["tr_mmm"],  color=COLORS["mmm"], lw=1.2,
             label="Train MMM", alpha=0.85)
    ax2.plot(d["val_steps"], d["val_mmm"], color=COLORS["val"], lw=1.5,
             label="Val MMM",   alpha=0.95, marker="o", markersize=3)
    ax2.legend(loc="upper right", frameon=True)
    _style_ax(ax2, "Masked Music Modeling Loss")

    # ── TI loss ──────────────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(d["tr_steps"],  d["tr_ti"],  color=COLORS["ti"], lw=1.2,
             label="Train TI", alpha=0.85)
    ax3.plot(d["val_steps"], d["val_ti"], color=COLORS["val"], lw=1.5,
             label="Val TI",   alpha=0.95, marker="o", markersize=3)
    ax3.legend(loc="upper right", frameon=True)
    _style_ax(ax3, "Transposition Invariance Loss")

    # ── SCL loss (train only — not logged at eval) ───────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(d["tr_steps"], d["tr_scl"], color=COLORS["scl"], lw=1.2,
             label="Train SCL", alpha=0.85)
    ax4.legend(loc="upper right", frameon=True)
    _style_ax(ax4, "Section Contrastive Loss")

    total_steps = d["total_steps"] or int(d["tr_steps"][-1])
    fig.suptitle(f"ABC2Vec Training Curves  ({total_steps:,} steps)",
                 fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    for suffix in (".png", ".pdf"):
        out = output_dir / f"training_curves_hq{suffix}"
        plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved: {out}")
    plt.close(fig)


def plot_smoothed(d: dict, output_dir: Path, window: int = 30):
    """2×2 grid, smoothed traces overlaid on faint raw traces."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    panels = [
        # (ax, tr_x, tr_y,        val_x,        val_y,         color_key, title)
        (axes[0], d["tr_steps"], d["tr_total"], d["val_steps"], d["val_total"], "train", "val",   "Total Loss"),
        (axes[1], d["tr_steps"], d["tr_mmm"],   d["val_steps"], d["val_mmm"],   "mmm",   "val",   "Masked Music Modeling Loss"),
        (axes[2], d["tr_steps"], d["tr_ti"],    d["val_steps"], d["val_ti"],    "ti",    "val",   "Transposition Invariance Loss"),
        (axes[3], d["tr_steps"], d["tr_scl"],   None,           None,           "scl",   None,    "Section Contrastive Loss"),
    ]

    for ax, tx, ty, vx, vy, tc, vc, title in panels:
        # Faint raw train
        ax.plot(tx, ty, color=COLORS[tc], lw=0.5, alpha=0.2)
        # Smoothed train
        ax.plot(tx, smooth(ty, window), color=COLORS[tc], lw=2.0,
                alpha=0.9, label="Train (smoothed)")
        if vx is not None:
            # Val points + smoothed (only if enough points)
            ax.plot(vx, vy, color=COLORS[vc], lw=0, marker="o",
                    markersize=3, alpha=0.4)
            if len(vy) >= window:
                ax.plot(vx, smooth(vy, max(3, window // 5)), color=COLORS[vc],
                        lw=2.0, alpha=0.9, label="Val (smoothed)")
            else:
                ax.plot(vx, vy, color=COLORS[vc], lw=1.5,
                        alpha=0.9, label="Val")
        ax.legend(loc="upper right", fontsize=8)
        _style_ax(ax, f"{title} (Smoothed)")

    total_steps = d["total_steps"] or int(d["tr_steps"][-1])
    fig.suptitle(f"ABC2Vec Training Curves — Smoothed  ({total_steps:,} steps)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    for suffix in (".png", ".pdf"):
        out = output_dir / f"training_curves_smoothed{suffix}"
        plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved: {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate publication-quality training curves from real training history."
    )
    parser.add_argument(
        "--history",
        type=str,
        default="./checkpoints/training_history.json",
        help="Path to training_history.json (default: ./checkpoints/training_history.json)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./figures",
        help="Output directory for figures (default: ./figures)",
    )
    parser.add_argument(
        "--smooth_window",
        type=int,
        default=30,
        help="Moving-average window for smoothed plots (default: 30)",
    )
    args = parser.parse_args()

    history_path = Path(args.history)
    output_dir   = Path(args.output_dir)

    if not history_path.exists():
        raise FileNotFoundError(
            f"Training history not found: {history_path}\n"
            "Run run_training.py first to generate it."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"History:      {history_path}")
    print(f"Output dir:   {output_dir}")

    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams.update({
        "figure.dpi":    300,
        "savefig.dpi":   300,
        "font.size":     10,
        "font.family":   "serif",
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 14,
    })

    d = load_history(history_path)
    print(f"\nLoaded {len(d['tr_steps'])} train log entries "
          f"(steps {d['tr_steps'][0]:,}–{d['tr_steps'][-1]:,})")
    print(f"Loaded {len(d['val_steps'])} val  log entries "
          f"(steps {d['val_steps'][0]:,}–{d['val_steps'][-1]:,})")
    if d["best_val_loss"]:
        print(f"Best val loss: {d['best_val_loss']:.6f}")

    plot_standard(d, output_dir)
    plot_smoothed(d, output_dir, window=args.smooth_window)

    print("\nDone. Four files written:")
    print("  training_curves_hq.png / .pdf")
    print("  training_curves_smoothed.png / .pdf")


if __name__ == "__main__":
    main()
