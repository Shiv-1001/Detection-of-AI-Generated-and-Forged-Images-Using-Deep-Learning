"""
Plotting helpers for reporting. Run AFTER model/train.py has produced
model/checkpoints/history.json — this script does not train anything itself
and does not invent numbers; it only visualizes what train.py actually logged.

Usage:
    python -m model.plots
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np

from core.config import CHECKPOINTS_DIR


def plot_training_curves(history_path=None, out_path=None):
    history_path = history_path or (CHECKPOINTS_DIR / "history.json")
    out_path = out_path or (CHECKPOINTS_DIR / "training_curves.png")

    if not history_path.exists():
        raise FileNotFoundError(
            f"{history_path} not found — run model/train.py first so there is "
            f"real epoch-by-epoch history to plot."
        )

    with open(history_path) as f:
        history = json.load(f)

    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    axes[0].plot(epochs, [h["loss"] for h in history], label="train loss", color="#c0392b")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Training loss"); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, [h["auc"] for h in history], label="val AUC", color="#2980b9")
    if "accuracy" in history[0]:
        axes[1].plot(epochs, [h["accuracy"] for h in history], label="val accuracy", color="#27ae60")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Score")
    axes[1].set_title("Validation AUC / accuracy"); axes[1].legend(); axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
    return out_path


def plot_confusion_matrix(cm, class_names=("original", "ai-generated", "forged"), out_path=None):
    out_path = out_path or (CHECKPOINTS_DIR / "confusion_matrix.png")
    cm = np.array(cm)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names))); ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticks(range(len(class_names))); ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix (validation set)")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")

    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
    return out_path


if __name__ == "__main__":
    plot_training_curves()

    hist_file = CHECKPOINTS_DIR / "history.json"
    with open(hist_file) as f:
        last = json.load(f)[-1]
    if "confusion_matrix" in last:
        plot_confusion_matrix(last["confusion_matrix"])
    else:
        print("No confusion_matrix in the latest history entry — re-run train.py "
              "with the updated model/evaluate.py to log one.")
