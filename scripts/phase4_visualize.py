"""Phase 4 – Step 2: Visualization.

Generates all figures for the thesis:
    1. CWT spectrogram examples (left vs right hand MI)
    2. Accuracy-vs-data-fraction curves (transfer vs scratch)
    3. Fusion method ablation bar chart
    4. Per-subject accuracy heatmap
    5. Training curve (loss + val accuracy over epochs)

Requires matplotlib and the combined summary JSON from phase4_compile_results.py.

Usage:
    # After running phase4_compile_results.py
    uv run python scripts/phase4_visualize.py

    # Custom paths
    uv run python scripts/phase4_visualize.py \\
        --summary results/phase4_summary.json \\
        --data real --data-dir ~/mne_data \\
        --output-dir figures/
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _save(fig, path: Path, dpi: int = 150) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    logger.info("Saved figure: %s", path)


def _load_summary(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Summary file not found: {path}\n"
            f"Run `uv run python scripts/phase4_compile_results.py` first."
        )
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Figure 1: CWT Spectrogram Examples
# ---------------------------------------------------------------------------


def plot_spectrogram_examples(
    data_dir: str,
    output_path: Path,
    n_examples: int = 3,
) -> None:
    """Plot CWT spectrogram examples for left and right hand MI.

    Loads one subject from BCI IV-2a (or synthetic), generates CWT spectrograms
    for C3, Cz, C4 channels, and shows side-by-side comparison.
    """
    import matplotlib.pyplot as plt
    import mne
    from moabb.datasets import BNCI2014_001
    from moabb.paradigms import LeftRightImagery

    from bci.data.transforms import CWTSpectrogramTransform
    from bci.utils.config import SpectrogramConfig

    mne.set_log_level("ERROR")
    logger.info("Loading data for spectrogram examples...")
    dataset = BNCI2014_001()
    paradigm = LeftRightImagery(fmin=4.0, fmax=40.0, resample=128.0)
    X, y_labels, _ = paradigm.get_data(dataset=dataset, subjects=[1])

    classes = sorted(np.unique(y_labels))
    label_map = {c: i for i, c in enumerate(classes)}
    y = np.array([label_map[lb] for lb in y_labels])

    channel_names = (
        paradigm.get_data(dataset=dataset, subjects=[1], return_epochs=True)[0].ch_names
        if hasattr(paradigm, "_get_epochs")
        else [
            "Fz",
            "FC3",
            "FC1",
            "FCz",
            "FC2",
            "FC4",
            "C5",
            "C3",
            "C1",
            "Cz",
            "C2",
            "C4",
            "C6",
            "CP3",
            "CP1",
            "CPz",
            "CP2",
            "CP4",
            "P1",
            "Pz",
            "P2",
            "POz",
        ]
    )

    config = SpectrogramConfig(
        wavelet="morl",
        freq_min=4.0,
        freq_max=40.0,
        n_freqs=64,
        image_size=(224, 224),
        channel_mode="rgb_c3_cz_c4",
    )
    transform = CWTSpectrogramTransform(config)

    # Pick examples
    left_idx = np.where(y == 0)[0][:n_examples]
    right_idx = np.where(y == 1)[0][:n_examples]

    fig, axes = plt.subplots(
        2,
        n_examples,
        figsize=(4 * n_examples, 8),
        constrained_layout=True,
    )
    fig.suptitle("CWT Spectrogram Examples (C3→R, Cz→G, C4→B)", fontsize=14)

    for col, idx in enumerate(left_idx):
        img = transform.transform_trial_rgb(X[idx], channel_names, sfreq=128.0)
        axes[0, col].imshow(img)
        axes[0, col].set_title(f"Left Hand (Trial {idx})")
        axes[0, col].axis("off")

    for col, idx in enumerate(right_idx):
        img = transform.transform_trial_rgb(X[idx], channel_names, sfreq=128.0)
        axes[1, col].imshow(img)
        axes[1, col].set_title(f"Right Hand (Trial {idx})")
        axes[1, col].axis("off")

    axes[0, 0].set_ylabel("Left Hand", fontsize=12)
    axes[1, 0].set_ylabel("Right Hand", fontsize=12)

    _save(fig, output_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: Accuracy vs Training Data Fraction
# ---------------------------------------------------------------------------


def plot_reduced_data_curves(summary: dict, output_path: Path) -> None:
    """Plot accuracy vs fraction of training data for transfer vs scratch."""
    import matplotlib.pyplot as plt

    reduced = summary.get("reduced_data")
    if not reduced:
        logger.warning("No reduced-data results found in summary. Skipping plot.")
        return

    results = reduced.get("results", {})
    fractions = reduced.get("fractions", [0.10, 0.25, 0.50, 0.75, 1.00])
    conditions = list(results.keys())

    colors = {"scratch": "#e74c3c", "imagenet": "#3498db", "transfer": "#2ecc71"}
    labels = {"scratch": "Scratch", "imagenet": "ImageNet", "transfer": "EEG-Pretrained (Ours)"}
    markers = {"scratch": "o", "imagenet": "s", "transfer": "^"}

    fig, ax = plt.subplots(figsize=(8, 5))

    for cond in conditions:
        frac_data = results.get(cond, {})
        xs, ys, errs = [], [], []
        for frac in fractions:
            frac_str = f"{frac:.2f}"
            d = frac_data.get(frac_str, {})
            mean = d.get("mean_accuracy", float("nan"))
            std = d.get("std_accuracy", float("nan"))
            xs.append(frac * 100)
            ys.append(mean)
            errs.append(std)

        xs_arr = np.array(xs)
        ys_arr = np.array(ys)
        errs_arr = np.array(errs)
        valid = ~np.isnan(ys_arr)

        color = colors.get(cond, "gray")
        marker = markers.get(cond, "o")
        label = labels.get(cond, cond.upper())

        ax.plot(
            xs_arr[valid],
            ys_arr[valid],
            color=color,
            marker=marker,
            linewidth=2,
            markersize=8,
            label=label,
        )
        ax.fill_between(
            xs_arr[valid],
            ys_arr[valid] - errs_arr[valid],
            ys_arr[valid] + errs_arr[valid],
            color=color,
            alpha=0.15,
        )

    ax.axhline(50, color="gray", linestyle="--", linewidth=1, label="Chance (50%)")
    ax.set_xlabel("Training Data Used (%)", fontsize=12)
    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.set_title("Transfer Learning Advantage Under Data Scarcity", fontsize=13)
    ax.set_xlim(0, 110)
    ax.set_ylim(40, 105)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    _save(fig, output_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: Fusion Method Ablation Bar Chart
# ---------------------------------------------------------------------------


def plot_fusion_ablation(summary: dict, output_path: Path) -> None:
    """Bar chart comparing attention / gated fusion methods."""
    import matplotlib.pyplot as plt

    dual = summary.get("dual_branch", {})
    methods = ["attention", "gated"]
    labels_map = {
        "attention": "Attention\nFusion",
        "gated": "Gated\nFusion",
    }
    colors = ["#2ecc71", "#9b59b6"]

    accs, stds, names = [], [], []
    for m in methods:
        r = dual.get(m)
        if r:
            accs.append(r.get("within_acc", float("nan")))
            stds.append(r.get("within_std", float("nan")))
            names.append(labels_map[m])

    if not accs:
        logger.warning("No dual-branch results found for ablation plot. Skipping.")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(names))
    bars = ax.bar(
        x,
        accs,
        yerr=stds,
        capsize=6,
        color=colors[: len(names)],
        alpha=0.85,
        edgecolor="black",
        linewidth=0.8,
    )

    # Value labels
    for bar, acc in zip(bars, accs):
        if not np.isnan(acc):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{acc:.1f}%",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel("Within-Subject Accuracy (%)", fontsize=11)
    ax.set_title("Dual-Branch Fusion Method Ablation", fontsize=12)
    ax.set_ylim(40, min(110, max(accs) + 15) if accs else 110)
    ax.grid(True, axis="y", alpha=0.3)

    _save(fig, output_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4: Per-Subject Accuracy Heatmap
# ---------------------------------------------------------------------------


def plot_per_subject_heatmap(summary: dict, output_path: Path) -> None:
    """Heatmap: rows=model, cols=subject, cells=accuracy."""
    import matplotlib.pyplot as plt

    # Collect per-subject data
    model_rows: list[tuple[str, dict]] = []

    ba = summary.get("baselines", {}).get("csp_lda")
    if ba and ba.get("loso_per_subject"):
        model_rows.append(("CSP+LDA\n(LOSO)", ba["loso_per_subject"]))

    bb = summary.get("baselines", {}).get("riemannian")
    if bb and bb.get("loso_per_subject"):
        model_rows.append(("Riemannian\n(LOSO)", bb["loso_per_subject"]))

    tl = summary.get("transfer_learning", {})
    for cond, label in [
        ("scratch", "Scratch"),
        ("imagenet", "ImageNet"),
        ("transfer", "EEG-Pretrained"),
    ]:
        r = tl.get(cond)
        if r and r.get("per_subject"):
            model_rows.append((label, r["per_subject"]))

    dual_attn = summary.get("dual_branch", {}).get("attention")
    if dual_attn and dual_attn.get("per_subject"):
        model_rows.append(("DualBranch\n(Attention)", dual_attn["per_subject"]))

    if not model_rows:
        logger.warning("No per-subject data found. Skipping heatmap.")
        return

    # Build matrix
    all_subjects = sorted(set(int(sid) for _, ps in model_rows for sid in ps.keys()))
    matrix = np.full((len(model_rows), len(all_subjects)), np.nan)
    for row_i, (_, ps) in enumerate(model_rows):
        for col_j, sid in enumerate(all_subjects):
            val = ps.get(str(sid)) or ps.get(sid)
            if val is not None:
                matrix[row_i, col_j] = float(val)

    fig, ax = plt.subplots(figsize=(max(8, len(all_subjects) * 0.9), max(4, len(model_rows) * 0.8)))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=40, vmax=100)
    plt.colorbar(im, ax=ax, label="Accuracy (%)")

    ax.set_xticks(range(len(all_subjects)))
    ax.set_xticklabels([f"S{s:02d}" for s in all_subjects], fontsize=10)
    ax.set_yticks(range(len(model_rows)))
    ax.set_yticklabels([r[0] for r in model_rows], fontsize=10)
    ax.set_title("Per-Subject Accuracy by Model", fontsize=12)

    # Annotate cells
    for i in range(len(model_rows)):
        for j in range(len(all_subjects)):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(
                    j,
                    i,
                    f"{val:.0f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black",
                )

    _save(fig, output_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5: Baseline Comparison Bar Chart
# ---------------------------------------------------------------------------


def plot_baseline_comparison(summary: dict, output_path: Path) -> None:
    """Bar chart comparing all models on within-subject accuracy."""
    import matplotlib.pyplot as plt

    entries = [
        ("CSP+LDA", summary.get("baselines", {}).get("csp_lda"), "#e74c3c"),
        ("Riemannian+LDA", summary.get("baselines", {}).get("riemannian"), "#e67e22"),
        ("ViT-Only", summary.get("baselines", {}).get("vit_only"), "#f1c40f"),
        ("DualBranch\n(Attention)", summary.get("dual_branch", {}).get("attention"), "#2ecc71"),
        ("DualBranch\n(Gated)", summary.get("dual_branch", {}).get("gated"), "#9b59b6"),
    ]

    names, accs, stds, colors = [], [], [], []
    for name, r, color in entries:
        if r and r.get("within_acc") is not None:
            names.append(name)
            accs.append(r["within_acc"])
            stds.append(r.get("within_std", 0.0))
            colors.append(color)

    if not names:
        logger.warning("Not enough data for baseline comparison chart. Skipping.")
        return

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.2), 5))
    x = np.arange(len(names))
    bars = ax.bar(
        x,
        accs,
        yerr=stds,
        capsize=6,
        color=colors,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.8,
    )
    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"{acc:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    ax.axhline(50, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="Chance")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("Within-Subject Accuracy (%)", fontsize=11)
    ax.set_title("Model Comparison: Within-Subject CV Accuracy", fontsize=12)
    ax.set_ylim(40, min(115, max(accs) + 15))
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)

    _save(fig, output_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4: Generate thesis figures")
    parser.add_argument(
        "--summary",
        default="results/phase4_summary.json",
        help="Path to phase4_summary.json from compile_results.py",
    )
    parser.add_argument("--output-dir", default="figures", help="Output directory for figures")
    parser.add_argument("--data", choices=["synthetic", "real"], default="real")
    parser.add_argument("--data-dir", default="~/mne_data")
    parser.add_argument(
        "--skip-spectrograms",
        action="store_true",
        help="Skip CWT spectrogram examples (requires BCI IV-2a data)",
    )
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load summary
    try:
        summary = _load_summary(Path(args.summary))
    except FileNotFoundError as e:
        logger.error(str(e))
        return

    # Figure 1: CWT spectrograms
    if not args.skip_spectrograms and args.data == "real":
        try:
            plot_spectrogram_examples(
                data_dir=args.data_dir,
                output_path=out / "fig1_cwt_spectrograms.png",
            )
        except Exception as e:
            logger.warning("Spectrogram examples failed: %s", e)
    else:
        logger.info("Skipping spectrogram examples (use --data real to generate).")

    # Figure 2: Reduced-data curves
    plot_reduced_data_curves(summary, out / "fig2_reduced_data_curves.png")

    # Figure 3: Fusion ablation
    plot_fusion_ablation(summary, out / "fig3_fusion_ablation.png")

    # Figure 4: Per-subject heatmap
    plot_per_subject_heatmap(summary, out / "fig4_per_subject_heatmap.png")

    # Figure 5: Baseline comparison
    plot_baseline_comparison(summary, out / "fig5_baseline_comparison.png")

    logger.info("All figures saved to %s/", out)


if __name__ == "__main__":
    main()
