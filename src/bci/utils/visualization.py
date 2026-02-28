"""Visualization utilities for EEG training results."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def save_fold_plots(
    train_result,
    y_true,
    y_pred,
    plots_dir: Path,
    tag: str,
) -> None:
    """Save per-fold training curves and confusion matrix.

    Used by stage scripts (05, 06, 07) to persist diagnostic plots for each
    cross-validation fold.

    Args:
        train_result: A ``TrainResult`` returned by ``Trainer.train()``.
            Must expose ``.history`` (list of epoch records with ``.epoch``,
            ``.train_loss``, ``.val_loss``, ``.val_accuracy``),
            ``.best_epoch``, and ``.best_val_accuracy``.
        y_true: Ground-truth integer labels for the test fold.
        y_pred: Predicted integer labels for the test fold.
        plots_dir: Directory to write PNG files into (created if absent).
        tag: Short identifier prefixed to each saved filename.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: F811 (re-import for Agg backend)
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    plots_dir.mkdir(parents=True, exist_ok=True)

    history = train_result.history
    epochs_range = [r.epoch for r in history]
    train_losses = [r.train_loss for r in history]
    val_losses = [r.val_loss for r in history]
    val_accs = [r.val_accuracy for r in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs_range, train_losses, label="Train Loss")
    axes[0].plot(epochs_range, val_losses, label="Val Loss")
    axes[0].axvline(
        x=train_result.best_epoch,
        color="green",
        linestyle="--",
        alpha=0.7,
        label=f"Best epoch {train_result.best_epoch}",
    )
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{tag} \u2013 Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs_range, val_accs, label="Val Accuracy", color="orange")
    axes[1].axvline(
        x=train_result.best_epoch,
        color="green",
        linestyle="--",
        alpha=0.7,
        label=f"Best: {train_result.best_val_accuracy:.1f}%",
    )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title(f"{tag} \u2013 Val Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(plots_dir / f"{tag}_curves.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    cm = confusion_matrix(y_true, y_pred)
    fig2, ax2 = plt.subplots(figsize=(4, 3))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Left", "Right"],
        yticklabels=["Left", "Right"],
        ax=ax2,
    )
    ax2.set_title(f"{tag} \u2013 Confusion Matrix")
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("True")
    plt.tight_layout()
    fig2.savefig(plots_dir / f"{tag}_confusion.png", dpi=120, bbox_inches="tight")
    plt.close(fig2)
