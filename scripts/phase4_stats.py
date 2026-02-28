"""Phase 4 – Step 3: Statistical Analysis.

Runs statistical significance tests on the experiment results:
    1. Wilcoxon signed-rank test: dual-branch vs each baseline (per subject)
    2. Effect size (Cohen's d) between transfer and scratch conditions
    3. Friedman test across fusion methods
    4. Per-subject variance analysis

Outputs a statistical summary table and saves results to JSON.

Usage:
    uv run python scripts/phase4_stats.py

    uv run python scripts/phase4_stats.py \\
        --summary results/phase4_summary.json \\
        --output results/phase4_stats.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------


def cohens_d(a: list[float], b: list[float]) -> float:
    """Compute Cohen's d effect size between two groups."""
    a, b = np.array(a), np.array(b)
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return float("nan")
    pooled_std = math.sqrt(
        ((n_a - 1) * np.var(a, ddof=1) + (n_b - 1) * np.var(b, ddof=1)) / (n_a + n_b - 2)
    )
    if pooled_std < 1e-10:
        return float("nan")
    return float((np.mean(a) - np.mean(b)) / pooled_std)


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
    """Wilcoxon signed-rank test. Returns (statistic, p_value)."""
    try:
        from scipy.stats import wilcoxon  # type: ignore

        diffs = np.array(a) - np.array(b)
        if np.all(diffs == 0):
            return float("nan"), 1.0
        stat, p = wilcoxon(diffs)
        return float(stat), float(p)
    except ImportError:
        logger.warning("scipy not available; skipping Wilcoxon test.")
        return float("nan"), float("nan")
    except Exception as e:
        logger.warning("Wilcoxon test failed: %s", e)
        return float("nan"), float("nan")


def t_test_paired(a: list[float], b: list[float]) -> tuple[float, float]:
    """Paired t-test. Returns (statistic, p_value)."""
    try:
        from scipy.stats import ttest_rel  # type: ignore

        stat, p = ttest_rel(a, b)
        return float(stat), float(p)
    except ImportError:
        logger.warning("scipy not available; using manual t-test.")
        a, b = np.array(a), np.array(b)
        diffs = a - b
        n = len(diffs)
        if n < 2:
            return float("nan"), float("nan")
        mean_d = np.mean(diffs)
        std_d = np.std(diffs, ddof=1)
        if std_d < 1e-10:
            return float("nan"), 0.0 if mean_d != 0 else 1.0
        t = mean_d / (std_d / math.sqrt(n))
        return float(t), float("nan")  # p-value needs scipy
    except Exception as e:
        logger.warning("t-test failed: %s", e)
        return float("nan"), float("nan")


def friedman_test(*groups: list[float]) -> tuple[float, float]:
    """Friedman test for k related samples. Returns (statistic, p_value)."""
    try:
        from scipy.stats import friedmanchisquare  # type: ignore

        stat, p = friedmanchisquare(*groups)
        return float(stat), float(p)
    except ImportError:
        logger.warning("scipy not available; skipping Friedman test.")
        return float("nan"), float("nan")
    except Exception as e:
        logger.warning("Friedman test failed: %s", e)
        return float("nan"), float("nan")


# ---------------------------------------------------------------------------
# Extract per-subject accuracy arrays from summary
# ---------------------------------------------------------------------------


def _get_per_subject_accs(result: dict | None) -> dict[int, float]:
    """Return {subject_id: accuracy} from a result dict."""
    if result is None:
        return {}
    ps = result.get("loso_per_subject") or result.get("per_subject") or {}
    return {int(k): float(v) for k, v in ps.items()}


def _aligned_arrays(ps_a: dict, ps_b: dict) -> tuple[list[float], list[float]]:
    """Return two lists aligned on shared subject IDs."""
    common = sorted(set(ps_a) & set(ps_b))
    return [ps_a[s] for s in common], [ps_b[s] for s in common]


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------


def analyze_baseline_vs_dual(summary: dict) -> list[dict]:
    """Compare dual-branch (attention) against each baseline."""
    results = []

    dual_attn = summary.get("dual_branch", {}).get("attention")
    if dual_attn is None:
        logger.warning("No dual-branch attention results. Skipping baseline comparison.")
        return results

    dual_ps = _get_per_subject_accs(dual_attn)
    if not dual_ps:
        logger.warning("Dual-branch has no per-subject data. Skipping.")
        return results

    baselines = [
        ("CSP+LDA", summary.get("baselines", {}).get("csp_lda")),
        ("Riemannian+LDA", summary.get("baselines", {}).get("riemannian")),
        ("ViT-Only", summary.get("baselines", {}).get("vit_only")),
    ]

    for name, bsl in baselines:
        if bsl is None:
            logger.info("  Skipping %s (no data)", name)
            continue
        bsl_ps = _get_per_subject_accs(bsl)
        dual_vals, bsl_vals = _aligned_arrays(dual_ps, bsl_ps)

        if len(dual_vals) < 2:
            logger.info("  Not enough subjects for %s comparison", name)
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
    """Compare transfer learning conditions pairwise."""
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
            logger.info("  Skipping %s (missing data)", label)
            continue

        ps_a = _get_per_subject_accs(ra)
        ps_b = _get_per_subject_accs(rb)
        vals_a, vals_b = _aligned_arrays(ps_a, ps_b)

        # Fall back to fold-level comparison from CVResult folds
        # If per_subject is empty but we have mean_accuracy
        if not vals_a and ra.get("within_acc") is not None:
            logger.info("  %s: no per-subject data; using mean only", label)
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
    """Friedman test across fusion methods."""
    dual = summary.get("dual_branch", {})
    methods = ["attention", "gated"]

    accs_by_method = {}
    for m in methods:
        r = dual.get(m)
        if r:
            ps = _get_per_subject_accs(r)
            if ps:
                accs_by_method[m] = [ps[s] for s in sorted(ps)]

    if len(accs_by_method) < 2:
        return {"note": "Not enough fusion results for Friedman test."}

    # Align subjects
    common_subjects = set.intersection(*[set(range(len(v))) for v in accs_by_method.values()])
    groups = [list(accs_by_method[m]) for m in methods if m in accs_by_method]

    # Only Friedman if all same length
    lengths = [len(g) for g in groups]
    summary_stats = {}
    for m, g in zip(methods, groups):
        summary_stats[m] = {
            "mean": float(np.mean(g)),
            "std": float(np.std(g)),
            "n": len(g),
        }

    if len(set(lengths)) == 1 and lengths[0] >= 3:
        f_stat, f_p = friedman_test(*groups)
        return {
            "test": "Friedman",
            "statistic": f_stat,
            "p_value": f_p,
            "significant_at_0.05": f_p < 0.05 if not np.isnan(f_p) else None,
            "per_method": summary_stats,
        }
    else:
        return {
            "test": "Descriptive only (insufficient data for Friedman)",
            "per_method": summary_stats,
        }


def analyze_per_subject_variance(summary: dict) -> dict:
    """Report per-subject variance for the best model."""
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


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------


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


def print_stats_table(baseline_results: list, transfer_results: list) -> None:
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
                    f"{name:<40} {n:>3} "
                    f"{diff:>+6.2f}% "
                    f"{t:>8.3f} "
                    f"{p:>8.4f} "
                    f"{sig:>5} "
                    f"{d:>6.3f} "
                    f"{eff:>10}"
                )

    print("\n  Significance: *** p<0.001, ** p<0.01, * p<0.05, n.s. = not significant")
    print("=" * W + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4: Statistical Analysis")
    parser.add_argument(
        "--summary",
        default="results/phase4_summary.json",
        help="Path to phase4_summary.json from compile_results.py",
    )
    parser.add_argument(
        "--output",
        default="results/phase4_stats.json",
        help="Output path for statistics JSON",
    )
    args = parser.parse_args()

    summary_path = Path(args.summary)
    if not summary_path.exists():
        logger.error(
            "Summary file not found: %s\n"
            "Run `uv run python scripts/phase4_compile_results.py` first.",
            summary_path,
        )
        return

    with open(summary_path) as f:
        summary = json.load(f)

    # Run analyses
    logger.info("Running baseline vs dual-branch analysis...")
    baseline_results = analyze_baseline_vs_dual(summary)

    logger.info("Running transfer learning condition analysis...")
    transfer_results = analyze_transfer_conditions(summary)

    logger.info("Running fusion method analysis...")
    fusion_results = analyze_fusion_methods(summary)

    logger.info("Running per-subject variance analysis...")
    variance_results = analyze_per_subject_variance(summary)

    # Print
    print_stats_table(baseline_results, transfer_results)

    if fusion_results:
        print("\n--- Fusion Method Comparison ---")
        pm = fusion_results.get("per_method", {})
        for m, s in pm.items():
            print(f"  {m:<12}: {s['mean']:.2f}% ± {s['std']:.2f}% (n={s['n']})")
        if "p_value" in fusion_results:
            p = fusion_results["p_value"]
            print(
                f"\n  Friedman test: chi2={fusion_results['statistic']:.3f}, "
                f"p={p:.4f} {_pval_stars(p)}"
            )

    # Save
    stats_output = {
        "baseline_vs_dual": baseline_results,
        "transfer_conditions": transfer_results,
        "fusion_methods": fusion_results,
        "per_subject_variance": variance_results,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(stats_output, f, indent=2)
    logger.info("Statistics saved to %s", args.output)


if __name__ == "__main__":
    main()
