"""Phase 4 – Step 1: Compile All Results into a Comparison Table.

Reads all results JSON files from the results/ directory and prints a single
unified comparison table covering:
    - Baseline A: CSP + LDA
    - Baseline B: Riemannian + LDA
    - Baseline C: CWT + ViT-Tiny (no math branch)
    - Dual-Branch (attention fusion)
    - Dual-Branch (concat fusion)
    - Dual-Branch (gated fusion)
    - Transfer Learning: scratch vs imagenet vs transfer
    - Reduced-data curves (summary)

Also writes a combined JSON summary for downstream plotting.

Usage:
    # After all experiments are done
    uv run python scripts/phase4_compile_results.py

    # Specify results directory
    uv run python scripts/phase4_compile_results.py --results-dir results/
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loaders for each JSON format
# ---------------------------------------------------------------------------


def load_baseline_ab(path: Path) -> dict | None:
    """Load Baseline A or B result (has within_subject + loso keys)."""
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    return {
        "model": data.get("model", path.stem),
        "within_acc": data["within_subject"]["mean_accuracy"],
        "within_std": data["within_subject"]["std_accuracy"],
        "within_kappa": data["within_subject"]["mean_kappa"],
        "within_f1": data["within_subject"]["mean_f1"],
        "loso_acc": data["loso"]["mean_accuracy"],
        "loso_std": data["loso"]["std_accuracy"],
        "loso_kappa": data["loso"]["mean_kappa"],
        "loso_f1": data["loso"]["mean_f1"],
        "loso_per_subject": data["loso"].get("per_subject", {}),
    }


def load_baseline_c(path: Path) -> dict | None:
    """Load Baseline C result (within_subject only, no loso)."""
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    return {
        "model": data.get("model", path.stem),
        "within_acc": data["within_subject"]["mean_accuracy"],
        "within_std": data["within_subject"]["std_accuracy"],
        "within_kappa": data["within_subject"]["mean_kappa"],
        "within_f1": data["within_subject"]["mean_f1"],
        "loso_acc": None,
        "loso_std": None,
        "loso_kappa": None,
        "loso_f1": None,
        "loso_per_subject": {},
    }


def load_dual_branch(path: Path) -> dict | None:
    """Load dual-branch result (strategy + mean_accuracy etc.)."""
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    return {
        "model": f"{data.get('model', 'DualBranch')} [{data.get('fusion', '?')}]",
        "strategy": data.get("strategy", "within_subject"),
        "within_acc": data["mean_accuracy"],
        "within_std": data["std_accuracy"],
        "within_kappa": data["mean_kappa"],
        "within_f1": data["mean_f1"],
        "loso_acc": None,
        "loso_std": None,
        "loso_kappa": None,
        "loso_f1": None,
        "per_subject": data.get("per_subject", {}),
    }


def load_finetune(path: Path) -> dict | None:
    """Load finetune condition result."""
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    return {
        "model": data.get("model", path.stem),
        "condition": data.get("condition", "?"),
        "within_acc": data["mean_accuracy"],
        "within_std": data["std_accuracy"],
        "within_kappa": data["mean_kappa"],
        "within_f1": data["mean_f1"],
        "per_subject": data.get("per_subject", {}),
    }


def load_reduced_data(path: Path) -> dict | None:
    """Load reduced-data experiment results."""
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    return data  # full nested structure


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
    rows_with_loso = [r for r in rows if r.get("loso_acc") is not None]
    if not rows_with_loso:
        return

    W = 80
    print("\n" + "=" * W)
    print("  LOSO CROSS-SUBJECT RESULTS")
    print("=" * W)
    print(f"\n{'Model':<35} {'LOSO-Acc':>10} {'±Std':>7} {'Kappa':>7} {'F1':>7}")
    print("-" * W)
    for r in rows_with_loso:
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


def print_reduced_data_table(data: dict) -> None:
    if not data:
        return
    results = data.get("results", {})
    fractions = data.get("fractions", [])
    conditions = list(results.keys())
    if not conditions or not fractions:
        return

    W = 14 + 22 * len(conditions)
    print("\n" + "=" * W)
    print("  PHASE 3: Reduced-Data Accuracy vs Training Set Size")
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
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4: Compile all experiment results")
    parser.add_argument("--results-dir", default="results", help="Results directory")
    parser.add_argument(
        "--output",
        default="results/phase4_summary.json",
        help="Output path for combined JSON summary",
    )
    parser.add_argument(
        "--prefix",
        default="real_",
        help="File prefix for real-data results (default: 'real_'). "
        "Use '' for synthetic-only results.",
    )
    args = parser.parse_args()

    rd = Path(args.results_dir)
    p = args.prefix  # e.g. "real_" or ""

    # --- Load all results ---
    baseline_a = load_baseline_ab(rd / f"{p}baseline_a_csp_lda.json")
    if baseline_a:
        baseline_a["model"] = "Baseline A: CSP+LDA"

    baseline_b = load_baseline_ab(rd / f"{p}baseline_b_riemannian.json")
    if baseline_b:
        baseline_b["model"] = "Baseline B: Riemannian+LDA"

    baseline_c = load_baseline_c(rd / f"{p}baseline_c_vit.json")
    if baseline_c:
        baseline_c["model"] = "Baseline C: CWT+ViT-Tiny"

    dual_attn = load_dual_branch(rd / f"{p}dual_branch_attention.json")
    dual_gated = load_dual_branch(rd / f"{p}dual_branch_gated.json")
    # Also check _full suffixes from Phase 2 synthetic runs
    if dual_attn is None:
        dual_attn = load_dual_branch(rd / "dual_branch_attention_full.json")
    if dual_gated is None:
        dual_gated = load_dual_branch(rd / "dual_branch_gated_full.json")

    ft_scratch = load_finetune(rd / f"{p}finetune_scratch.json")
    ft_imagenet = load_finetune(rd / f"{p}finetune_imagenet.json")
    ft_transfer = load_finetune(rd / f"{p}finetune_transfer.json")
    # Fallback to synthetic finetune results
    if ft_scratch is None:
        ft_scratch = load_finetune(rd / "finetune_scratch.json")
    if ft_imagenet is None:
        ft_imagenet = load_finetune(rd / "finetune_imagenet.json")
    if ft_transfer is None:
        ft_transfer = load_finetune(rd / "finetune_transfer.json")

    if ft_scratch:
        ft_scratch["model"] = "Transfer: Scratch"
    if ft_imagenet:
        ft_imagenet["model"] = "Transfer: ImageNet"
    if ft_transfer:
        ft_transfer["model"] = "Transfer: EEG-Pretrained"

    reduced = load_reduced_data(rd / f"{p}reduced_data_results.json")
    if reduced is None:
        reduced = load_reduced_data(rd / "reduced_data_results.json")

    # --- Build main table rows ---
    rows = []
    for r in [
        baseline_a,
        baseline_b,
        baseline_c,
        dual_attn,
        dual_gated,
        ft_scratch,
        ft_imagenet,
        ft_transfer,
    ]:
        if r is not None:
            rows.append(r)

    if not rows:
        logger.warning(
            "No result files found in %s (prefix='%s'). Run the experiments first.",
            rd,
            p,
        )
        return

    # --- Print tables ---
    print_main_table(rows)
    print_loso_table(rows)

    # Per-subject breakdown for dual-branch and transfer
    for label, r in [
        ("CSP+LDA LOSO", baseline_a),
        ("Riemannian LOSO", baseline_b),
        ("DualBranch Attention", dual_attn),
        ("Transfer: EEG-Pretrained", ft_transfer),
    ]:
        if r is not None:
            ps = r.get("loso_per_subject") or r.get("per_subject") or {}
            print_per_subject_table(label, ps)

    # Reduced-data table
    print_reduced_data_table(reduced)

    # --- Save combined summary ---
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
        "transfer_learning": {
            "scratch": ft_scratch,
            "imagenet": ft_imagenet,
            "transfer": ft_transfer,
        },
        "reduced_data": reduced,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Combined summary saved to %s", args.output)


if __name__ == "__main__":
    main()
