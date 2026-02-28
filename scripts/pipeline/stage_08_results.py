"""Stage 08 – Result Analysis, Plotting, and Statistical Tests.

Reads all result JSON files produced by Stages 02–07, prints a comprehensive
comparison table, runs statistical significance tests, generates five thesis
figures, and saves a combined summary JSON.

Consolidates logic previously spread across:
  - scripts/phase4_compile_results.py
  - scripts/phase4_visualize.py
  - scripts/phase4_stats.py

Result files consumed (all under <run-dir>/results/):
  - real_baseline_a_csp_lda.json         (Stage 02)
  - real_baseline_a_csp_lda_loso.json    (Stage 02)
  - real_baseline_b_riemannian.json       (Stage 03)
  - real_baseline_b_riemannian_loso.json  (Stage 03)
  - real_baseline_c_vit.json              (Stage 05)
  - real_baseline_c_vit_loso.json         (Stage 05)
  - real_dual_branch_attention_vit.json   (Stage 06)
  - real_dual_branch_attention_vit_loso.json (Stage 06)
  - real_dual_branch_gated_vit.json       (Stage 06)
  - real_dual_branch_gated_vit_loso.json  (Stage 06)
  - real_reduced_data_results_vit.json    (Stage 07)

Outputs (under <run-dir>/):
  - results/phase4_summary.json           combined summary for downstream use
  - results/phase4_stats.json             statistical test results
  - figures/fig1_cwt_spectrograms.png     CWT spectrogram examples
  - figures/fig2_reduced_data_curves.png  accuracy vs data fraction
  - figures/fig3_fusion_ablation.png      fusion method bar chart
  - figures/fig4_per_subject_heatmap.png  per-subject accuracy heatmap
  - figures/fig5_baseline_comparison.png  overall model comparison

Usage::

    uv run python scripts/pipeline/stage_08_results.py --run-dir runs/my_run
    uv run python scripts/pipeline/stage_08_results.py \\
        --run-dir runs/my_run --skip-spectrograms --dpi 300
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def setup_logging(run_dir: Path) -> logging.Logger:
    fmt = "%(asctime)s  %(levelname)-8s  %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt, datefmt="%H:%M:%S", stream=sys.stdout)
    for lib in ("mne", "matplotlib", "moabb"):
        logging.getLogger(lib).setLevel(logging.ERROR)
    log = logging.getLogger("stage_08")
    run_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(run_dir / "stage_08_results.log")
    fh.setFormatter(logging.Formatter(fmt, "%H:%M:%S"))
    log.addHandler(fh)
    return log


# ---------------------------------------------------------------------------
# Result loaders
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_baseline_ab(path: Path, loso_path: Path | None = None) -> dict | None:
    """Load Baseline A or B – has within_subject + optional loso keys.

    Accepts either:
    - A single JSON with top-level ``within_subject`` and ``loso`` keys, or
    - Separate within-subject and LOSO JSON files (new stage 02/03 format).
    """
    data = _load_json(path)
    if data is None:
        return None

    # New format: separate files for within vs LOSO
    if "within_subject" in data:
        ws = data["within_subject"]
        lo = data.get("loso", {})
        if not lo and loso_path is not None:
            lo_data = _load_json(loso_path)
            lo = lo_data.get("loso", lo_data) if lo_data else {}
        return {
            "model": data.get("model", path.stem),
            "within_acc": ws.get("mean_accuracy"),
            "within_std": ws.get("std_accuracy"),
            "within_kappa": ws.get("mean_kappa"),
            "within_f1": ws.get("mean_f1"),
            "loso_acc": lo.get("mean_accuracy"),
            "loso_std": lo.get("std_accuracy"),
            "loso_kappa": lo.get("mean_kappa"),
            "loso_f1": lo.get("mean_f1"),
            "loso_per_subject": lo.get("per_subject", {}),
        }

    # Flat format (mean_accuracy at top level – within only)
    return {
        "model": data.get("model", path.stem),
        "within_acc": data.get("mean_accuracy"),
        "within_std": data.get("std_accuracy"),
        "within_kappa": data.get("mean_kappa"),
        "within_f1": data.get("mean_f1"),
        "loso_acc": None,
        "loso_std": None,
        "loso_kappa": None,
        "loso_f1": None,
        "loso_per_subject": {},
    }


def load_baseline_c(path: Path, loso_path: Path | None = None) -> dict | None:
    """Load Baseline C (ViT-only) – within + optional LOSO JSON."""
    data = _load_json(path)
    if data is None:
        return None

    ws = data.get("within_subject", data)  # support both flat and nested
    lo: dict = {}
    if loso_path is not None:
        lo_data = _load_json(loso_path)
        if lo_data is not None:
            lo = lo_data.get("loso", lo_data)

    return {
        "model": data.get("model", path.stem),
        "within_acc": ws.get("mean_accuracy"),
        "within_std": ws.get("std_accuracy"),
        "within_kappa": ws.get("mean_kappa"),
        "within_f1": ws.get("mean_f1"),
        "loso_acc": lo.get("mean_accuracy"),
        "loso_std": lo.get("std_accuracy"),
        "loso_kappa": lo.get("mean_kappa"),
        "loso_f1": lo.get("mean_f1"),
        "loso_per_subject": lo.get("per_subject", {}),
    }


def load_dual_branch(path: Path, loso_path: Path | None = None) -> dict | None:
    """Load dual-branch result."""
    data = _load_json(path)
    if data is None:
        return None

    lo: dict = {}
    if loso_path is not None:
        lo_data = _load_json(loso_path)
        if lo_data is not None:
            lo = lo_data.get("loso", lo_data)

    return {
        "model": f"{data.get('model', 'DualBranch')} [{data.get('fusion', '?')}]",
        "strategy": data.get("strategy", "within_subject"),
        "within_acc": data.get("mean_accuracy"),
        "within_std": data.get("std_accuracy"),
        "within_kappa": data.get("mean_kappa"),
        "within_f1": data.get("mean_f1"),
        "loso_acc": lo.get("mean_accuracy"),
        "loso_std": lo.get("std_accuracy"),
        "loso_kappa": lo.get("mean_kappa"),
        "loso_f1": lo.get("mean_f1"),
        "per_subject": data.get("per_subject", {}),
        "loso_per_subject": lo.get("per_subject", {}),
    }


def load_reduced_data(path: Path) -> dict | None:
    """Load reduced-data experiment results (full nested structure)."""
    return _load_json(path)


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------


def _fmt(val: float | None, pct: bool = True) -> str:
    if val is None:
        return "   N/A  "
    if pct:
        return f"{val:7.2f}%"
    return f"{val:7.4f}"


def print_main_table(rows: list[dict]) -> None:
    W = 80
    print("\n" + "=" * W)
    print("  THESIS RESULTS: COMPREHENSIVE COMPARISON TABLE")
    print("=" * W)
    print(f"\n{'Model':<35} {'Within-Acc':>10} {'±Std':>7} {'Kappa':>7} {'F1':>7}")
    print("-" * W)
    for r in rows:
        name = r["model"][:34]
        acc = _fmt(r.get("within_acc"))
        std = _fmt(r.get("within_std"))
        kappa = _fmt(r.get("within_kappa"), pct=False)
        f1 = _fmt(r.get("within_f1"), pct=False)
        print(f"{name:<35} {acc:>10} {std:>7} {kappa:>7} {f1:>7}")
    print("=" * W)


def print_loso_table(rows: list[dict]) -> None:
    rows_loso = [r for r in rows if r.get("loso_acc") is not None]
    if not rows_loso:
        return
    W = 80
    print("\n" + "=" * W)
    print("  LOSO CROSS-SUBJECT RESULTS")
    print("=" * W)
    print(f"\n{'Model':<35} {'LOSO-Acc':>10} {'±Std':>7} {'Kappa':>7} {'F1':>7}")
    print("-" * W)
    for r in rows_loso:
        name = r["model"][:34]
        acc = _fmt(r.get("loso_acc"))
        std = _fmt(r.get("loso_std"))
        kappa = _fmt(r.get("loso_kappa"), pct=False)
        f1 = _fmt(r.get("loso_f1"), pct=False)
        print(f"{name:<35} {acc:>10} {std:>7} {kappa:>7} {f1:>7}")
    print("=" * W)


def print_per_subject_table(label: str, per_subject: dict) -> None:
    if not per_subject:
        return
    print(f"\n  Per-subject accuracy ({label}):")
    for sid, acc in sorted(per_subject.items(), key=lambda x: int(x[0])):
        bar = "#" * int(float(acc) / 5)
        print(f"    S{int(sid):02d}: {float(acc):6.2f}%  {bar}")


def print_reduced_data_table(data: dict | None) -> None:
    if not data:
        return
    results = data.get("results", {})
    fractions = data.get("fractions", [])
    conditions = list(results.keys())
    if not conditions or not fractions:
        return
    W = 14 + 22 * len(conditions)
    print("\n" + "=" * W)
    print("  STAGE 07: Reduced-Data Accuracy vs Training Set Size")
    print("=" * W)
    header = f"  {'Fraction':>8}"
    for c in conditions:
        header += f"  {c.upper():>18}"
    print(header)
    print("-" * W)
    for frac in fractions:
        frac_str = f"{frac:.2f}"
        row = f"  {frac * 100:>7.0f}%"
        for c in conditions:
            d = results.get(c, {}).get(frac_str, {})
            mean = d.get("mean_accuracy", float("nan"))
            std = d.get("std_accuracy", float("nan"))
            row += f"  {mean:>7.2f}% ±{std:>5.2f}%"
        print(row)
    print("=" * W + "\n")


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------


def cohens_d(a: list[float], b: list[float]) -> float:
    a_arr, b_arr = np.array(a), np.array(b)
    n_a, n_b = len(a_arr), len(b_arr)
    if n_a < 2 or n_b < 2:
        return float("nan")
    pooled_std = math.sqrt(
        ((n_a - 1) * np.var(a_arr, ddof=1) + (n_b - 1) * np.var(b_arr, ddof=1)) / (n_a + n_b - 2)
    )
    if pooled_std < 1e-10:
        return float("nan")
    return float((np.mean(a_arr) - np.mean(b_arr)) / pooled_std)


def interpret_d(d: float) -> str:
    ad = abs(d)
    if np.isnan(ad):
        return "N/A"
    if ad < 0.2:
        return "negligible"
    if ad < 0.5:
        return "small"
    if ad < 0.8:
        return "medium"
    return "large"


def wilcoxon_test(a: list[float], b: list[float]) -> tuple[float, float]:
    try:
        from scipy.stats import wilcoxon  # type: ignore

        diffs = np.array(a) - np.array(b)
        if np.all(diffs == 0):
            return float("nan"), 1.0
        stat, p = wilcoxon(diffs)
        return float(stat), float(p)
    except ImportError:
        logging.getLogger("stage_08").warning("scipy not available; skipping Wilcoxon test.")
        return float("nan"), float("nan")
    except Exception as e:
        logging.getLogger("stage_08").warning("Wilcoxon test failed: %s", e)
        return float("nan"), float("nan")


def t_test_paired(a: list[float], b: list[float]) -> tuple[float, float]:
    try:
        from scipy.stats import ttest_rel  # type: ignore

        stat, p = ttest_rel(a, b)
        return float(stat), float(p)
    except ImportError:
        a_arr, b_arr = np.array(a), np.array(b)
        diffs = a_arr - b_arr
        n = len(diffs)
        if n < 2:
            return float("nan"), float("nan")
        mean_d = float(np.mean(diffs))
        std_d = float(np.std(diffs, ddof=1))
        if std_d < 1e-10:
            return float("nan"), 0.0 if mean_d != 0 else 1.0
        t = mean_d / (std_d / math.sqrt(n))
        return float(t), float("nan")
    except Exception as e:
        logging.getLogger("stage_08").warning("t-test failed: %s", e)
        return float("nan"), float("nan")


def friedman_test(*groups: list[float]) -> tuple[float, float]:
    try:
        from scipy.stats import friedmanchisquare  # type: ignore

        stat, p = friedmanchisquare(*groups)
        return float(stat), float(p)
    except ImportError:
        logging.getLogger("stage_08").warning("scipy not available; skipping Friedman test.")
        return float("nan"), float("nan")
    except Exception as e:
        logging.getLogger("stage_08").warning("Friedman test failed: %s", e)
        return float("nan"), float("nan")


def _get_per_subject_accs(result: dict | None) -> dict[int, float]:
    if result is None:
        return {}
    ps = result.get("loso_per_subject") or result.get("per_subject") or {}
    return {int(k): float(v) for k, v in ps.items()}


def _aligned(ps_a: dict, ps_b: dict) -> tuple[list[float], list[float]]:
    common = sorted(set(ps_a) & set(ps_b))
    return [ps_a[s] for s in common], [ps_b[s] for s in common]


def _pval_stars(p: float) -> str:
    if np.isnan(p):
        return "N/A"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def analyze_baseline_vs_dual(summary: dict) -> list[dict]:
    log = logging.getLogger("stage_08")
    results = []
    dual_attn = summary.get("dual_branch", {}).get("attention")
    if dual_attn is None:
        log.warning("No dual-branch attention results. Skipping baseline comparison.")
        return results

    dual_ps = _get_per_subject_accs(dual_attn)
    if not dual_ps:
        log.warning("Dual-branch has no per-subject data. Skipping.")
        return results

    baselines = [
        ("CSP+LDA", summary.get("baselines", {}).get("csp_lda")),
        ("Riemannian+LDA", summary.get("baselines", {}).get("riemannian")),
        ("ViT-Only", summary.get("baselines", {}).get("vit_only")),
    ]
    for name, bsl in baselines:
        if bsl is None:
            continue
        bsl_ps = _get_per_subject_accs(bsl)
        dual_vals, bsl_vals = _aligned(dual_ps, bsl_ps)
        if len(dual_vals) < 2:
            continue
        w_stat, w_p = wilcoxon_test(dual_vals, bsl_vals)
        t_stat, t_p = t_test_paired(dual_vals, bsl_vals)
        d = cohens_d(dual_vals, bsl_vals)
        results.append(
            {
                "comparison": f"DualBranch(Attn) vs {name}",
                "n_subjects": len(dual_vals),
                "dual_mean": float(np.mean(dual_vals)),
                "baseline_mean": float(np.mean(bsl_vals)),
                "mean_diff": float(np.mean(dual_vals)) - float(np.mean(bsl_vals)),
                "wilcoxon_stat": w_stat,
                "wilcoxon_p": w_p,
                "t_stat": t_stat,
                "t_p": t_p,
                "cohens_d": d,
                "effect_size": interpret_d(d),
            }
        )
    return results


def analyze_transfer_conditions(summary: dict) -> list[dict]:
    results = []
    tl = summary.get("transfer_learning", {})
    conditions = {
        "scratch": tl.get("scratch"),
        "imagenet": tl.get("imagenet"),
        "transfer": tl.get("transfer"),
    }
    pairs = [
        ("transfer", "scratch", "EEG-Pretrained vs Scratch"),
        ("transfer", "imagenet", "EEG-Pretrained vs ImageNet"),
        ("imagenet", "scratch", "ImageNet vs Scratch"),
    ]
    for cond_a, cond_b, label in pairs:
        ra = conditions.get(cond_a)
        rb = conditions.get(cond_b)
        if ra is None or rb is None:
            continue
        ps_a = _get_per_subject_accs(ra)
        ps_b = _get_per_subject_accs(rb)
        vals_a, vals_b = _aligned(ps_a, ps_b)
        if not vals_a:
            if ra.get("within_acc") is not None:
                results.append(
                    {
                        "comparison": label,
                        "n_subjects": 0,
                        f"{cond_a}_mean": ra.get("within_acc"),
                        f"{cond_b}_mean": rb.get("within_acc"),
                        "mean_diff": (ra.get("within_acc", 0) - rb.get("within_acc", 0)),
                        "note": "Insufficient per-subject data for significance test",
                    }
                )
            continue
        if len(vals_a) < 2:
            continue
        w_stat, w_p = wilcoxon_test(vals_a, vals_b)
        t_stat, t_p = t_test_paired(vals_a, vals_b)
        d = cohens_d(vals_a, vals_b)
        results.append(
            {
                "comparison": label,
                "n_subjects": len(vals_a),
                f"{cond_a}_mean": float(np.mean(vals_a)),
                f"{cond_b}_mean": float(np.mean(vals_b)),
                "mean_diff": float(np.mean(vals_a)) - float(np.mean(vals_b)),
                "wilcoxon_stat": w_stat,
                "wilcoxon_p": w_p,
                "t_stat": t_stat,
                "t_p": t_p,
                "cohens_d": d,
                "effect_size": interpret_d(d),
            }
        )
    return results


def analyze_fusion_methods(summary: dict) -> dict:
    dual = summary.get("dual_branch", {})
    methods = ["attention", "gated"]
    accs_by_method: dict[str, list[float]] = {}
    for m in methods:
        r = dual.get(m)
        if r:
            ps = _get_per_subject_accs(r)
            if ps:
                accs_by_method[m] = [ps[s] for s in sorted(ps)]
    if len(accs_by_method) < 2:
        return {"note": "Not enough fusion results for Friedman test."}
    groups = [list(accs_by_method[m]) for m in methods if m in accs_by_method]
    lengths = [len(g) for g in groups]
    summary_stats = {
        m: {"mean": float(np.mean(g)), "std": float(np.std(g)), "n": len(g)}
        for m, g in zip(methods, groups)
        if m in accs_by_method
    }
    if len(set(lengths)) == 1 and lengths[0] >= 3:
        f_stat, f_p = friedman_test(*groups)
        return {
            "test": "Friedman",
            "statistic": f_stat,
            "p_value": f_p,
            "significant_at_0.05": (f_p < 0.05 if not np.isnan(f_p) else None),
            "per_method": summary_stats,
        }
    return {
        "test": "Descriptive only (insufficient data for Friedman)",
        "per_method": summary_stats,
    }


def analyze_per_subject_variance(summary: dict) -> dict:
    dual_attn = summary.get("dual_branch", {}).get("attention")
    if dual_attn is None:
        return {}
    ps = _get_per_subject_accs(dual_attn)
    if not ps:
        return {}
    accs = list(ps.values())
    return {
        "model": "DualBranch (Attention)",
        "n_subjects": len(accs),
        "mean": float(np.mean(accs)),
        "std": float(np.std(accs)),
        "min": float(np.min(accs)),
        "max": float(np.max(accs)),
        "range": float(np.max(accs) - np.min(accs)),
        "per_subject": {str(k): v for k, v in sorted(ps.items())},
    }


def print_stats_table(baseline_results: list[dict], transfer_results: list[dict]) -> None:
    W = 90
    print("\n" + "=" * W)
    print("  STATISTICAL ANALYSIS RESULTS")
    print("=" * W)

    if baseline_results:
        print("\n--- Dual-Branch vs Baselines (Wilcoxon Signed-Rank, per-subject) ---")
        print(
            f"{'Comparison':<40} {'N':>3} {'Diff':>7} {'W-stat':>8} {'p':>8} {'Sig':>5} {'d':>6} {'Effect':>10}"
        )
        print("-" * W)
        for r in baseline_results:
            n = r.get("n_subjects", 0)
            diff = r.get("mean_diff", float("nan"))
            w = r.get("wilcoxon_stat", float("nan"))
            p = r.get("wilcoxon_p", float("nan"))
            d = r.get("cohens_d", float("nan"))
            eff = r.get("effect_size", "N/A")
            sig = _pval_stars(p)
            name = r["comparison"][:39]
            print(
                f"{name:<40} {n:>3} {diff:>+6.2f}% {w:>8.2f} {p:>8.4f} {sig:>5} {d:>6.3f} {eff:>10}"
            )

    if transfer_results:
        print("\n--- Transfer Learning Conditions (Paired t-test, per-subject) ---")
        print(
            f"{'Comparison':<40} {'N':>3} {'Diff':>7} {'t-stat':>8} {'p':>8} {'Sig':>5} {'d':>6} {'Effect':>10}"
        )
        print("-" * W)
        for r in transfer_results:
            n = r.get("n_subjects", 0)
            diff = r.get("mean_diff", float("nan"))
            t = r.get("t_stat", float("nan"))
            p = r.get("t_p", float("nan"))
            d = r.get("cohens_d", float("nan"))
            eff = r.get("effect_size", "N/A")
            note = r.get("note", "")
            sig = _pval_stars(p) if not note else "–"
            name = r["comparison"][:39]
            if note:
                print(f"{name:<40} {n:>3} {diff:>+6.2f}%  [{note}]")
            else:
                print(
                    f"{name:<40} {n:>3} {diff:>+6.2f}% {t:>8.3f} {p:>8.4f} {sig:>5} {d:>6.3f} {eff:>10}"
                )

    print("\n  Significance: *** p<0.001, ** p<0.01, * p<0.05, n.s. = not significant")
    print("=" * W + "\n")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _save_fig(fig, path: Path, dpi: int = 150) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    logging.getLogger("stage_08").info("Saved figure: %s", path)


def plot_spectrogram_examples(
    processed_dir: Path,
    output_path: Path,
    n_examples: int = 3,
    dpi: int = 150,
) -> None:
    """Plot 9-channel CWT spectrogram examples from cached BCI IV-2a data.

    Loads subject 1 spectrograms from the Stage 01 cache and shows a
    montage of left-hand vs right-hand trials for the first three channels.
    Falls back gracefully if the cache is not present.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    spec_path = processed_dir / "bci_iv2a" / "subject_01_spectrograms.npz"
    if not spec_path.exists():
        logging.getLogger("stage_08").warning(
            "Spectrogram cache not found at %s; skipping Figure 1.", spec_path
        )
        return

    npz = np.load(spec_path)
    images: np.ndarray = npz["images"]  # (N, 9, 224, 224) float32
    y: np.ndarray = npz["y"]  # (N,) int

    CHANNEL_NAMES = ["C3", "C1", "Cz", "C2", "C4", "FC3", "FC4", "CP3", "CP4"]
    display_ch = [0, 2, 4]  # C3, Cz, C4 – three representative channels
    n_display = len(display_ch)

    classes = np.unique(y)
    if len(classes) < 2:
        logging.getLogger("stage_08").warning("Only one class in cache; skipping Figure 1.")
        return

    left_idx = np.where(y == classes[0])[0][:n_examples]
    right_idx = np.where(y == classes[1])[0][:n_examples]

    fig, axes = plt.subplots(
        2 * n_display,
        n_examples,
        figsize=(4 * n_examples, 4 * n_display * 2),
        constrained_layout=True,
    )
    if axes.ndim == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle("CWT Spectrogram Examples – 9-ch multichannel (BCI IV-2a, Subject 1)", fontsize=12)

    for row_group, (class_idx, class_label) in enumerate(
        zip([left_idx, right_idx], ["Left Hand", "Right Hand"])
    ):
        for ch_offset, ch_idx in enumerate(display_ch):
            row = row_group * n_display + ch_offset
            ch_name = CHANNEL_NAMES[ch_idx]
            for col, trial_idx in enumerate(class_idx):
                ax = axes[row, col]
                img = images[trial_idx, ch_idx]  # (224, 224) float32
                ax.imshow(img, cmap="viridis", aspect="auto", vmin=0, vmax=1)
                if col == 0:
                    ax.set_ylabel(f"{class_label}\n{ch_name}", fontsize=9)
                if row == 0:
                    ax.set_title(f"Trial {trial_idx}", fontsize=9)
                ax.axis("off")

    _save_fig(fig, output_path, dpi=dpi)
    plt.close(fig)


def plot_reduced_data_curves(summary: dict, output_path: Path, dpi: int = 150) -> None:
    """Accuracy vs fraction of training data (transfer vs scratch)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reduced = summary.get("reduced_data")
    if not reduced:
        logging.getLogger("stage_08").warning("No reduced-data results. Skipping Figure 2.")
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
            xs.append(frac * 100)
            ys.append(d.get("mean_accuracy", float("nan")))
            errs.append(d.get("std_accuracy", float("nan")))

        xs_a = np.array(xs)
        ys_a = np.array(ys)
        errs_a = np.array(errs)
        valid = ~np.isnan(ys_a)
        color = colors.get(cond, "gray")
        ax.plot(
            xs_a[valid],
            ys_a[valid],
            color=color,
            marker=markers.get(cond, "o"),
            linewidth=2,
            markersize=8,
            label=labels.get(cond, cond.upper()),
        )
        ax.fill_between(
            xs_a[valid],
            ys_a[valid] - errs_a[valid],
            ys_a[valid] + errs_a[valid],
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

    _save_fig(fig, output_path, dpi=dpi)
    plt.close(fig)


def plot_fusion_ablation(summary: dict, output_path: Path, dpi: int = 150) -> None:
    """Bar chart comparing attention / gated fusion methods."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dual = summary.get("dual_branch", {})
    methods = ["attention", "gated"]
    labels_map = {"attention": "Attention\nFusion", "gated": "Gated\nFusion"}
    colors = ["#2ecc71", "#9b59b6"]

    accs, stds, names, bar_colors = [], [], [], []
    for m, c in zip(methods, colors):
        r = dual.get(m)
        if r and r.get("within_acc") is not None:
            accs.append(r["within_acc"])
            stds.append(r.get("within_std", 0.0))
            names.append(labels_map[m])
            bar_colors.append(c)

    if not accs:
        logging.getLogger("stage_08").warning(
            "No dual-branch results for ablation. Skipping Figure 3."
        )
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(names))
    bars = ax.bar(
        x,
        accs,
        yerr=stds,
        capsize=6,
        color=bar_colors,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.8,
    )
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

    _save_fig(fig, output_path, dpi=dpi)
    plt.close(fig)


def plot_per_subject_heatmap(summary: dict, output_path: Path, dpi: int = 150) -> None:
    """Heatmap: rows=model, cols=subject, cells=accuracy."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model_rows: list[tuple[str, dict]] = []

    ba = summary.get("baselines", {}).get("csp_lda")
    if ba and ba.get("loso_per_subject"):
        model_rows.append(("CSP+LDA\n(LOSO)", ba["loso_per_subject"]))

    bb = summary.get("baselines", {}).get("riemannian")
    if bb and bb.get("loso_per_subject"):
        model_rows.append(("Riemannian\n(LOSO)", bb["loso_per_subject"]))

    bc = summary.get("baselines", {}).get("vit_only")
    if bc and bc.get("loso_per_subject"):
        model_rows.append(("ViT-Only\n(LOSO)", bc["loso_per_subject"]))

    dual_attn = summary.get("dual_branch", {}).get("attention")
    if dual_attn and (dual_attn.get("loso_per_subject") or dual_attn.get("per_subject")):
        ps = dual_attn.get("loso_per_subject") or dual_attn.get("per_subject")
        model_rows.append(("DualBranch\n(Attention)", ps))

    dual_gated = summary.get("dual_branch", {}).get("gated")
    if dual_gated and (dual_gated.get("loso_per_subject") or dual_gated.get("per_subject")):
        ps = dual_gated.get("loso_per_subject") or dual_gated.get("per_subject")
        model_rows.append(("DualBranch\n(Gated)", ps))

    if not model_rows:
        logging.getLogger("stage_08").warning("No per-subject data found. Skipping Figure 4.")
        return

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
    for i in range(len(model_rows)):
        for j in range(len(all_subjects)):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=8, color="black")

    _save_fig(fig, output_path, dpi=dpi)
    plt.close(fig)


def plot_baseline_comparison(summary: dict, output_path: Path, dpi: int = 150) -> None:
    """Bar chart comparing all models on within-subject accuracy."""
    import matplotlib

    matplotlib.use("Agg")
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
        logging.getLogger("stage_08").warning(
            "Not enough data for comparison chart. Skipping Figure 5."
        )
        return

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.2), 5))
    x = np.arange(len(names))
    bars = ax.bar(
        x, accs, yerr=stds, capsize=6, color=colors, alpha=0.85, edgecolor="black", linewidth=0.8
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

    _save_fig(fig, output_path, dpi=dpi)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 08: Result analysis, plotting, and statistical tests.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--run-dir", required=True, help="Experiment run directory.")
    p.add_argument(
        "--processed-dir",
        default=None,
        help="Root of processed .npz cache (default: data/processed/). "
        "Used only for Figure 1 spectrogram examples.",
    )
    p.add_argument(
        "--skip-spectrograms",
        action="store_true",
        help="Skip Figure 1 (CWT spectrogram examples). Useful when cache is absent.",
    )
    p.add_argument("--dpi", type=int, default=150, help="DPI for saved figures.")
    p.add_argument(
        "--backbone",
        default="vit",
        help="Backbone short-name used in stage 06/07 output filenames (e.g. 'vit').",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    log = setup_logging(run_dir / "logs")

    rd = run_dir / "results"
    figures_dir = run_dir / "figures"
    bshort = args.backbone  # e.g. "vit"

    processed_dir = Path(args.processed_dir) if args.processed_dir else Path("data/processed")

    # ------------------------------------------------------------------
    # 1. Load all result files
    # ------------------------------------------------------------------
    log.info("Loading result files from %s ...", rd)

    baseline_a = load_baseline_ab(
        rd / "real_baseline_a_csp_lda.json",
        rd / "real_baseline_a_csp_lda_loso.json",
    )
    if baseline_a:
        baseline_a["model"] = "Baseline A: CSP+LDA"
        log.info("  Loaded Baseline A")

    baseline_b = load_baseline_ab(
        rd / "real_baseline_b_riemannian.json",
        rd / "real_baseline_b_riemannian_loso.json",
    )
    if baseline_b:
        baseline_b["model"] = "Baseline B: Riemannian+LDA"
        log.info("  Loaded Baseline B")

    baseline_c = load_baseline_c(
        rd / "real_baseline_c_vit.json",
        rd / "real_baseline_c_vit_loso.json",
    )
    if baseline_c:
        baseline_c["model"] = "Baseline C: CWT+ViT-Tiny"
        log.info("  Loaded Baseline C (ViT-only)")

    dual_attn = load_dual_branch(
        rd / f"real_dual_branch_attention_{bshort}.json",
        rd / f"real_dual_branch_attention_{bshort}_loso.json",
    )
    if dual_attn:
        log.info("  Loaded Dual-Branch (attention)")

    dual_gated = load_dual_branch(
        rd / f"real_dual_branch_gated_{bshort}.json",
        rd / f"real_dual_branch_gated_{bshort}_loso.json",
    )
    if dual_gated:
        log.info("  Loaded Dual-Branch (gated)")

    reduced = load_reduced_data(rd / f"real_reduced_data_results_{bshort}.json")
    if reduced:
        log.info("  Loaded reduced-data results")

    # ------------------------------------------------------------------
    # 2. Build summary structure
    # ------------------------------------------------------------------
    summary = {
        "baselines": {
            "csp_lda": baseline_a,
            "riemannian": baseline_b,
            "vit_only": baseline_c,
        },
        "dual_branch": {
            "attention": dual_attn,
            "gated": dual_gated,
        },
        # transfer_learning kept for stats compatibility; dual-branch serves as primary
        "transfer_learning": {
            "attention": dual_attn,
            "gated": dual_gated,
        },
        "reduced_data": reduced,
    }

    # ------------------------------------------------------------------
    # 3. Print tables
    # ------------------------------------------------------------------
    rows: list[dict] = [
        r for r in [baseline_a, baseline_b, baseline_c, dual_attn, dual_gated] if r is not None
    ]

    if not rows:
        log.warning("No result files found in %s. Run stages 02–07 first.", rd)
    else:
        print_main_table(rows)
        print_loso_table(rows)

        for label, r in [
            ("CSP+LDA LOSO", baseline_a),
            ("Riemannian LOSO", baseline_b),
            ("DualBranch Attention", dual_attn),
            ("DualBranch Gated", dual_gated),
        ]:
            if r is not None:
                ps = r.get("loso_per_subject") or r.get("per_subject") or {}
                print_per_subject_table(label, ps)

        print_reduced_data_table(reduced)

    # ------------------------------------------------------------------
    # 4. Statistical tests
    # ------------------------------------------------------------------
    log.info("Running statistical analyses ...")
    baseline_results = analyze_baseline_vs_dual(summary)
    transfer_results = analyze_transfer_conditions(summary)
    fusion_results = analyze_fusion_methods(summary)
    variance_results = analyze_per_subject_variance(summary)

    print_stats_table(baseline_results, transfer_results)

    if fusion_results.get("per_method"):
        print("\n--- Fusion Method Comparison ---")
        for m, s in fusion_results["per_method"].items():
            print(f"  {m:<12}: {s['mean']:.2f}% \u00b1 {s['std']:.2f}% (n={s['n']})")
        if "p_value" in fusion_results:
            fp = fusion_results["p_value"]
            print(
                f"\n  Friedman test: chi2={fusion_results['statistic']:.3f}, "
                f"p={fp:.4f} {_pval_stars(fp)}"
            )

    # ------------------------------------------------------------------
    # 5. Save summary + stats JSONs
    # ------------------------------------------------------------------
    rd.mkdir(parents=True, exist_ok=True)
    summary_path = rd / "phase4_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Summary saved to %s", summary_path)

    stats_output = {
        "baseline_vs_dual": baseline_results,
        "transfer_conditions": transfer_results,
        "fusion_methods": fusion_results,
        "per_subject_variance": variance_results,
    }
    stats_path = rd / "phase4_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats_output, f, indent=2)
    log.info("Statistics saved to %s", stats_path)

    # ------------------------------------------------------------------
    # 6. Generate figures
    # ------------------------------------------------------------------
    log.info("Generating figures ...")
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: CWT spectrogram examples from cache
    if not args.skip_spectrograms:
        try:
            plot_spectrogram_examples(
                processed_dir=processed_dir,
                output_path=figures_dir / "fig1_cwt_spectrograms.png",
                dpi=args.dpi,
            )
        except Exception as e:
            log.warning("Spectrogram examples failed: %s", e)
    else:
        log.info("Skipping Figure 1 (--skip-spectrograms).")

    # Figure 2: Reduced-data curves
    plot_reduced_data_curves(summary, figures_dir / "fig2_reduced_data_curves.png", dpi=args.dpi)

    # Figure 3: Fusion ablation
    plot_fusion_ablation(summary, figures_dir / "fig3_fusion_ablation.png", dpi=args.dpi)

    # Figure 4: Per-subject heatmap
    plot_per_subject_heatmap(summary, figures_dir / "fig4_per_subject_heatmap.png", dpi=args.dpi)

    # Figure 5: Baseline comparison
    plot_baseline_comparison(summary, figures_dir / "fig5_baseline_comparison.png", dpi=args.dpi)

    log.info("Stage 08 complete. Figures saved to %s/", figures_dir)


if __name__ == "__main__":
    main()
