"""Visualization utilities for EEG data, spectrograms, and results."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

logger = logging.getLogger(__name__)


def plot_spectrogram(
    spectrogram: np.ndarray,
    title: str = "CWT Spectrogram",
    freqs: np.ndarray | None = None,
    times: np.ndarray | None = None,
    ax: plt.Axes | None = None,
    save_path: str | Path | None = None,
) -> plt.Axes:
    """Plot a single CWT spectrogram.

    Args:
        spectrogram: 2D array of shape (n_freqs, n_times).
        title: Plot title.
        freqs: Frequency axis labels.
        times: Time axis labels.
        ax: Matplotlib axes to plot on. Creates new if None.
        save_path: Path to save the figure.

    Returns:
        The matplotlib Axes object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))

    extent = None
    if times is not None and freqs is not None:
        extent = [times[0], times[-1], freqs[0], freqs[-1]]

    im = ax.imshow(
        spectrogram,
        aspect="auto",
        origin="lower",
        cmap="jet",
        extent=extent,
    )
    ax.set_title(title)
    ax.set_xlabel("Time (s)" if times is not None else "Time samples")
    ax.set_ylabel("Frequency (Hz)" if freqs is not None else "Frequency bins")
    plt.colorbar(im, ax=ax, label="Magnitude")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved spectrogram plot to %s", save_path)

    return ax


def plot_spectrogram_rgb(
    image: np.ndarray,
    title: str = "RGB Spectrogram (C3/Cz/C4)",
    ax: plt.Axes | None = None,
    save_path: str | Path | None = None,
) -> plt.Axes:
    """Plot an RGB spectrogram image.

    Args:
        image: RGB array of shape (H, W, 3) as uint8.
        title: Plot title.
        ax: Matplotlib axes.
        save_path: Path to save.

    Returns:
        Axes object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    ax.imshow(image)
    ax.set_title(title)
    ax.axis("off")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return ax


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str] | None = None,
    title: str = "Confusion Matrix",
    ax: plt.Axes | None = None,
    save_path: str | Path | None = None,
) -> plt.Axes:
    """Plot a confusion matrix.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        class_names: Names for the classes.
        title: Plot title.
        ax: Matplotlib axes.
        save_path: Path to save.

    Returns:
        Axes object.
    """
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)

    if class_names is None:
        class_names = [f"Class {i}" for i in range(cm.shape[0])]

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return ax


def plot_training_curves(
    train_losses: list[float],
    val_losses: list[float],
    train_accs: list[float] | None = None,
    val_accs: list[float] | None = None,
    title: str = "Training Curves",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot training and validation loss/accuracy curves.

    Args:
        train_losses: Training loss per epoch.
        val_losses: Validation loss per epoch.
        train_accs: Training accuracy per epoch (optional).
        val_accs: Validation accuracy per epoch (optional).
        title: Plot title.
        save_path: Path to save.

    Returns:
        Matplotlib figure.
    """
    has_acc = train_accs is not None and val_accs is not None
    n_plots = 2 if has_acc else 1

    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    epochs = range(1, len(train_losses) + 1)

    # Loss plot
    axes[0].plot(epochs, train_losses, label="Train Loss")
    axes[0].plot(epochs, val_losses, label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{title} - Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy plot
    if has_acc:
        axes[1].plot(epochs, train_accs, label="Train Acc")
        axes[1].plot(epochs, val_accs, label="Val Acc")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy (%)")
        axes[1].set_title(f"{title} - Accuracy")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_subject_accuracies(
    subject_ids: list[int],
    accuracies: list[float],
    title: str = "Per-Subject Accuracy",
    save_path: str | Path | None = None,
) -> plt.Axes:
    """Bar plot of per-subject classification accuracies.

    Args:
        subject_ids: List of subject identifiers.
        accuracies: Corresponding accuracy values (0-100).
        title: Plot title.
        save_path: Path to save.

    Returns:
        Axes object.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    colors = sns.color_palette("viridis", n_colors=len(subject_ids))
    bars = ax.bar(
        [f"S{s}" for s in subject_ids],
        accuracies,
        color=colors,
    )

    # Add mean line
    mean_acc = np.mean(accuracies)
    ax.axhline(y=mean_acc, color="red", linestyle="--", label=f"Mean: {mean_acc:.1f}%")

    ax.set_xlabel("Subject")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title)
    ax.legend()
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 1,
            f"{acc:.1f}",
            ha="center",
            fontsize=9,
        )

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return ax
