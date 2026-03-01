"""Visualization utilities for EEG training results."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def save_training_curves(
    history: list,
    best_epoch: int,
    best_val_accuracy: float,
    plots_dir: Path,
    filename: str,
    title: str,
) -> None:
    """Save training loss + validation accuracy curves.

    Args:
        history: List of epoch records with ``.epoch``, ``.train_loss``,
            ``.val_loss``, ``.val_accuracy`` attributes.
        best_epoch: Epoch number of the best checkpoint.
        best_val_accuracy: Best validation accuracy (%).
        plots_dir: Directory to write PNG files into (created if absent).
        filename: Output filename (without extension).
        title: Human-readable title for the plot.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir.mkdir(parents=True, exist_ok=True)

    epochs_range = [r.epoch for r in history]
    train_losses = [r.train_loss for r in history]
    val_losses = [r.val_loss for r in history]
    val_accs = [r.val_accuracy for r in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs_range, train_losses, label="Train Loss")
    axes[0].plot(epochs_range, val_losses, label="Val Loss")
    axes[0].axvline(
        x=best_epoch,
        color="green",
        linestyle="--",
        alpha=0.7,
        label=f"Best epoch {best_epoch}",
    )
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{title} \u2013 Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs_range, val_accs, label="Val Accuracy", color="orange")
    axes[1].axvline(
        x=best_epoch,
        color="green",
        linestyle="--",
        alpha=0.7,
        label=f"Best: {best_val_accuracy:.1f}%",
    )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title(f"{title} \u2013 Validation Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(plots_dir / f"{filename}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    plots_dir: Path,
    filename: str,
    title: str,
    class_labels: list[str] | None = None,
) -> None:
    """Save a confusion matrix heatmap.

    Args:
        y_true: Ground-truth integer labels (aggregated across all folds).
        y_pred: Predicted integer labels.
        plots_dir: Directory to write PNG files into (created if absent).
        filename: Output filename (without extension).
        title: Human-readable title for the plot.
        class_labels: Optional class names for axes.  Defaults to
            ``["Left Hand", "Right Hand"]``.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    if class_labels is None:
        class_labels = ["Left Hand", "Right Hand"]

    plots_dir.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_labels,
        yticklabels=class_labels,
        ax=ax,
        annot_kws={"size": 14},
        linewidths=0.5,
        linecolor="gray",
    )
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    plt.tight_layout()
    fig.savefig(plots_dir / f"{filename}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_per_subject_accuracy(
    per_subject: dict[int, float],
    plots_dir: Path,
    filename: str,
    title: str,
) -> None:
    """Save a per-subject accuracy bar chart.

    Args:
        per_subject: Mapping of subject_id -> mean accuracy (%).
        plots_dir: Directory to write PNG files into (created if absent).
        filename: Output filename (without extension).
        title: Human-readable title for the plot.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    plots_dir.mkdir(parents=True, exist_ok=True)

    sids = sorted(per_subject.keys())
    accs = [per_subject[s] for s in sids]
    mean_a = sum(accs) / len(accs)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = sns.color_palette("viridis", n_colors=len(sids))
    bars = ax.bar([f"S{s}" for s in sids], accs, color=colors)
    ax.axhline(y=mean_a, color="red", linestyle="--", label=f"Mean: {mean_a:.1f}%")
    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.5,
            f"{acc:.1f}",
            ha="center",
            fontsize=9,
        )
    ax.set_xlabel("Subject")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title)
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(plots_dir / f"{filename}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
