"""Overnight Full Experiment Runner.

Runs the entire 8-stage thesis pipeline from start to finish.
Stages are skipped automatically if their result file already exists, making
the script resumable after crashes.

Stages:
    01. Download + process → save .npz epochs + generate & cache spectrograms
    02. Baseline A – CSP + LDA (within-subject + LOSO)
    03. Baseline B – Riemannian + LDA (within-subject + LOSO)
    04. Pretrain ViT (in_chans=9) on PhysioNet spectrograms
    05. ViT-only baseline using PhysioNet-pretrained weights
    06. Dual-branch (attention + gated fusion ablation)
    07. Reduced-data transfer learning experiment
    08. Result analysis, plotting, and statistical tests

All outputs go to a timestamped run directory, e.g.::

    runs/2024-01-15_143022/
        results/
            real_baseline_a_csp_lda.json
            real_baseline_b_riemannian.json
            ...
        checkpoints/
            vit_pretrained_physionet_vit.pt
        figures/
            fig1_spectrogram_examples.png
            ...
        plots/
            stage_02_csp_lda/
            ...
        experiment.log

Usage (GPU machine, overnight)::

    uv run python scripts/run_full_experiment.py --device cuda

    # Skip already-done stages (safe to re-run after crash)
    uv run python scripts/run_full_experiment.py --run-dir runs/2024-01-15_143022

    # Dry run: just print the plan
    uv run python scripts/run_full_experiment.py --dry-run

    # Override hyperparameters
    uv run python scripts/run_full_experiment.py --epochs 30 --batch-size 64 --device cuda
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s  %(levelname)-8s  %(message)s"
    datefmt = "%H:%M:%S"

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter(fmt, datefmt))
    root.addHandler(ch)

    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(logging.Formatter(fmt, datefmt))
    root.addHandler(fh)

    return logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Banner helpers
# ---------------------------------------------------------------------------


def banner(msg: str, char: str = "=", width: int = 70) -> None:
    line = char * width
    print(f"\n{line}")
    print(f"  {msg}")
    print(f"{line}\n", flush=True)


# ---------------------------------------------------------------------------
# Stage runner
# ---------------------------------------------------------------------------


def run_stage(n: int, total: int, name: str, cmd: list[str], log: logging.Logger) -> bool:
    """Run a single stage as a subprocess.  Returns True on success."""
    banner(f"STAGE {n}/{total}: {name}")
    log.info("Running: %s", " ".join(cmd))
    t0 = time.time()
    result = subprocess.run(cmd, check=False)
    elapsed = time.time() - t0
    if result.returncode == 0:
        log.info("Stage %d/%d done in %.1fs.", n, total, elapsed)
        return True
    else:
        log.error(
            "Stage %d/%d FAILED (exit code %d) after %.1fs.", n, total, result.returncode, elapsed
        )
        return False


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Full overnight experiment runner for BCI thesis (8 stages).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--run-dir",
        default=None,
        help="Resume from an existing run directory. "
        "If omitted, a new timestamped directory is created.",
    )
    p.add_argument("--device", default="auto", help="Device: auto | cpu | cuda | mps")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-folds", type=int, default=5, help="CV folds (within-subject)")
    p.add_argument("--epochs", type=int, default=50, help="Max training epochs per fold")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--backbone", default="vit_tiny_patch16_224", help="timm backbone name")
    p.add_argument(
        "--n-repeats",
        type=int,
        default=3,
        help="Repetitions per fraction in reduced-data experiment (Stage 07)",
    )
    p.add_argument(
        "--fractions",
        nargs="+",
        type=float,
        default=[0.10, 0.25, 0.50, 0.75, 1.00],
        help="Training-set fractions for Stage 07",
    )
    p.add_argument(
        "--checkpoint",
        default=None,
        help="Path to PhysioNet-pretrained checkpoint for Stages 05/06/07. "
        "Defaults to <run-dir>/checkpoints/vit_pretrained_physionet_vit.pt",
    )
    p.add_argument(
        "--processed-dir",
        default=None,
        help="Root of .npz cache (default: data/processed/)",
    )
    p.add_argument("--skip-download", action="store_true", help="Skip Stage 01")
    p.add_argument("--skip-baselines", action="store_true", help="Skip Stages 02-03")
    p.add_argument("--skip-pretrain", action="store_true", help="Skip Stage 04")
    p.add_argument("--skip-vit", action="store_true", help="Skip Stage 05")
    p.add_argument("--skip-dual", action="store_true", help="Skip Stage 06")
    p.add_argument("--skip-reduced", action="store_true", help="Skip Stage 07")
    p.add_argument("--skip-results", action="store_true", help="Skip Stage 08")
    p.add_argument(
        "--dry-run", action="store_true", help="Print the plan and exit without running."
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    # ── Run directory ──────────────────────────────────────────────────────
    if args.run_dir:
        run_dir = Path(args.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"Resuming run in: {run_dir}")
    else:
        ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        run_dir = Path("runs") / ts
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"New run directory: {run_dir}")

    log = setup_logging(run_dir / "experiment.log")
    log.info("Run directory: %s", run_dir)

    # ── Derive backbone short ──────────────────────────────────────────────
    backbone_short = "vit" if args.backbone == "vit_tiny_patch16_224" else args.backbone
    checkpoint = args.checkpoint or str(
        run_dir / "checkpoints" / f"vit_pretrained_physionet_{backbone_short}.pt"
    )

    TOTAL = 8
    uv = ["uv", "run", "python"]

    # Shared CLI fragments
    base = ["--run-dir", str(run_dir)]
    nn = [
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--device",
        args.device,
        "--backbone",
        args.backbone,
        "--seed",
        str(args.seed),
    ]
    cv = ["--n-folds", str(args.n_folds)]
    proc = ["--processed-dir", str(args.processed_dir)] if args.processed_dir else []

    # ── Plan ──────────────────────────────────────────────────────────────
    plan = [
        (
            1,
            "Download + process → .npz cache + spectrograms",
            not args.skip_download,
            uv + ["scripts/pipeline/stage_01_download.py"] + base + proc,
        ),
        (
            2,
            "Baseline A: CSP + LDA",
            not args.skip_baselines,
            uv
            + ["scripts/pipeline/stage_02_baseline_a.py"]
            + base
            + cv
            + ["--seed", str(args.seed)]
            + proc,
        ),
        (
            3,
            "Baseline B: Riemannian + LDA",
            not args.skip_baselines,
            uv
            + ["scripts/pipeline/stage_03_baseline_b.py"]
            + base
            + cv
            + ["--seed", str(args.seed)]
            + proc,
        ),
        (
            4,
            "Pretrain ViT on PhysioNet",
            not args.skip_pretrain,
            uv + ["scripts/pipeline/stage_04_pretrain_vit.py"] + base + nn + proc,
        ),
        (
            5,
            "ViT-only baseline (PhysioNet-pretrained)",
            not args.skip_vit,
            uv
            + ["scripts/pipeline/stage_05_vit_baseline.py"]
            + base
            + nn
            + cv
            + proc
            + ["--checkpoint", checkpoint],
        ),
        (
            6,
            "Dual-branch (attention + gated fusion ablation)",
            not args.skip_dual,
            uv
            + ["scripts/pipeline/stage_06_dual_branch.py"]
            + base
            + nn
            + cv
            + proc
            + ["--checkpoint", checkpoint],
        ),
        (
            7,
            "Reduced-data transfer learning experiment",
            not args.skip_reduced,
            uv
            + ["scripts/pipeline/stage_07_reduced_data.py"]
            + base
            + nn
            + ["--n-repeats", str(args.n_repeats)]
            + ["--fractions"]
            + [str(f) for f in args.fractions]
            + ["--checkpoint", checkpoint]
            + proc,
        ),
        (
            8,
            "Result analysis + plots + stats",
            not args.skip_results,
            uv + ["scripts/pipeline/stage_08_results.py"] + base,
        ),
    ]

    banner(f"BCI THESIS PIPELINE – 8-stage full run")
    log.info("Backbone      : %s  (short: %s)", args.backbone, backbone_short)
    log.info("Device        : %s", args.device)
    log.info("Epochs        : %d", args.epochs)
    log.info("Batch size    : %d", args.batch_size)
    log.info("CV folds      : %d", args.n_folds)
    log.info("Repeats (S07) : %d", args.n_repeats)
    log.info("Seed          : %d", args.seed)
    log.info("Checkpoint    : %s", checkpoint)
    log.info("")
    log.info("Stages to run:")
    for n, name, enabled, _ in plan:
        status = "RUN " if enabled else "SKIP"
        log.info("  [%s] Stage %02d: %s", status, n, name)

    if args.dry_run:
        banner("DRY RUN: plan printed. Exiting without running.")
        return

    # ── Execute ───────────────────────────────────────────────────────────
    t_start = time.time()
    failures: list[int] = []

    for n, name, enabled, cmd in plan:
        if not enabled:
            log.info("Stage %02d skipped.", n)
            continue
        ok = run_stage(n, TOTAL, name, cmd, log)
        if not ok:
            failures.append(n)

    # ── Summary ───────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    hours, rem = divmod(int(elapsed), 3600)
    mins, secs = divmod(rem, 60)

    if failures:
        banner(f"PIPELINE FINISHED WITH ERRORS  ({hours}h {mins}m {secs}s)", char="!")
        log.error("Failed stages: %s", failures)
        log.error("Check logs above for details.")
        sys.exit(1)
    else:
        banner(f"ALL STAGES COMPLETE  ({hours}h {mins}m {secs}s)", char="*")
        log.info("Run directory : %s", run_dir)
        log.info("Results       : %s/results/", run_dir)
        log.info("Figures       : %s/figures/", run_dir)
        log.info("Log           : %s/experiment.log", run_dir)


if __name__ == "__main__":
    main()
