#!/usr/bin/env bash
# =============================================================================
# run_all.sh  –  Run the full BCI thesis experiment pipeline (8 stages)
# =============================================================================
#
# This script runs all 8 pipeline stages in order.
# Every stage saves its output to a shared run directory so that:
#   • stages can be resumed if something crashes
#   • each stage can also be run independently (see ONE-BY-ONE section below)
#
# ─────────────────────────────────────────────────────────────────────────────
# QUICK START  (full overnight run)
# ─────────────────────────────────────────────────────────────────────────────
#
#   RUN_DIR=runs/gpu_run bash scripts/run_all.sh --device cuda
#
#   # Resume an existing run (completed stages are skipped automatically):
#   RUN_DIR=runs/gpu_run bash scripts/run_all.sh --device cuda
#
# ─────────────────────────────────────────────────────────────────────────────
# HOW TO RUN STAGES ONE BY ONE
# ─────────────────────────────────────────────────────────────────────────────
#
# First, create a run directory (pick any name):
#
#   export RUN=runs/my_experiment
#
# Then run each stage individually in order.
# Every stage skips itself automatically if its output file already exists,
# so it is always safe to re-run a stage.
#
#   Stage 1 – Download + process datasets → save .npz cache + spectrograms:
#     uv run python scripts/pipeline/stage_01_download.py \
#         --run-dir $RUN
#
#   Stage 2 – Baseline A: CSP + LDA (within-subject + LOSO):
#     uv run python scripts/pipeline/stage_02_baseline_a.py \
#         --run-dir $RUN --n-folds 5 --seed 42
#
#   Stage 3 – Baseline B: Riemannian + LDA (within-subject + LOSO):
#     uv run python scripts/pipeline/stage_03_baseline_b.py \
#         --run-dir $RUN --n-folds 5 --seed 42
#
#   Stage 4 – Pretrain ViT on PhysioNet (9-channel spectrograms):
#     uv run python scripts/pipeline/stage_04_pretrain_vit.py \
#         --run-dir $RUN --epochs 50 --batch-size 32 --device cuda
#
#   Stage 5 – ViT-only baseline (PhysioNet-pretrained weights):
#     uv run python scripts/pipeline/stage_05_vit_baseline.py \
#         --run-dir $RUN --epochs 50 --batch-size 32 --device cuda
#
#   Stage 6 – Dual-branch (attention + gated fusion ablation):
#     uv run python scripts/pipeline/stage_06_dual_branch.py \
#         --run-dir $RUN --epochs 50 --batch-size 32 --device cuda
#
#   Stage 7 – Reduced-data transfer learning experiment:
#     uv run python scripts/pipeline/stage_07_reduced_data.py \
#         --run-dir $RUN --epochs 50 --n-repeats 3 --device cuda
#
#   Stage 8 – Result analysis, plotting, statistical tests:
#     uv run python scripts/pipeline/stage_08_results.py \
#         --run-dir $RUN
#
# ─────────────────────────────────────────────────────────────────────────────
# OPTIONS (for this script)
# ─────────────────────────────────────────────────────────────────────────────
#   --device DEVICE    pytorch device     (default: auto)
#   --epochs N         max epochs/fold    (default: 50)
#   --batch-size N     batch size         (default: 32)
#   --n-folds N        CV folds           (default: 5)
#   --n-repeats N      repeats for Stage 7 (default: 3)
#   --seed N           random seed        (default: 42)
#
# Set RUN_DIR env variable to reuse an existing directory, e.g.:
#   RUN_DIR=runs/2024-01-15_143022 bash scripts/run_all.sh
#
# ─────────────────────────────────────────────────────────────────────────────
# NOTES
# ─────────────────────────────────────────────────────────────────────────────
#   • Must be run from the project root (where pyproject.toml lives).
#   • Requires `uv` to be installed and `uv sync` already run.
#   • Stage 01 must complete before any other stage (builds .npz cache).
#   • Each stage writes results under the same run directory.
#     Completed stages are skipped on re-run (idempotent).
#   • All output is also logged to <run-dir>/run_all.log.
#   • Stages 4-7 benefit greatly from a GPU. On CPU they may take many hours.
# =============================================================================

set -euo pipefail

# ── Verify project root ────────────────────────────────────────────────────
if [[ ! -f "pyproject.toml" ]]; then
    echo "ERROR: Run this script from the project root (where pyproject.toml lives)." >&2
    exit 1
fi

# ── Defaults ──────────────────────────────────────────────────────────────
DEVICE="auto"
EPOCHS=50
BATCH_SIZE=32
N_FOLDS=5
N_REPEATS=3
SEED=42

# ── Parse flags ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --device)      DEVICE="$2";      shift 2 ;;
        --epochs)      EPOCHS="$2";      shift 2 ;;
        --batch-size)  BATCH_SIZE="$2";  shift 2 ;;
        --n-folds)     N_FOLDS="$2";     shift 2 ;;
        --n-repeats)   N_REPEATS="$2";   shift 2 ;;
        --seed)        SEED="$2";        shift 2 ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Run: bash scripts/run_all.sh  (with no args) to see usage in the header." >&2
            exit 1
            ;;
    esac
done

# ── Run directory ─────────────────────────────────────────────────────────
if [[ -z "${RUN_DIR:-}" ]]; then
    TS=$(date +"%Y-%m-%d_%H%M%S")
    RUN_DIR="runs/$TS"
fi
mkdir -p "$RUN_DIR"

LOG="$RUN_DIR/run_all.log"

# ── Logging helpers ───────────────────────────────────────────────────────
_log() {
    local msg="[$(date '+%H:%M:%S')] $*"
    echo "$msg" | tee -a "$LOG"
}

_banner() {
    local line="======================================================================"
    _log ""
    _log "$line"
    _log "  $*"
    _log "$line"
}

# ── Print settings ────────────────────────────────────────────────────────
_banner "BCI THESIS PIPELINE  –  Full run (8 stages)"
_log "Run directory : $RUN_DIR"
_log "Device        : $DEVICE"
_log "Epochs        : $EPOCHS"
_log "Batch size    : $BATCH_SIZE"
_log "CV folds      : $N_FOLDS"
_log "Repeats (S7)  : $N_REPEATS"
_log "Seed          : $SEED"
_log ""

T_ALL=$(date +%s)

# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 – Download + process → .npz cache + spectrograms
# ─────────────────────────────────────────────────────────────────────────────
_banner "Stage 1/8 – Download + process datasets"
T0=$(date +%s)
uv run python scripts/pipeline/stage_01_download.py \
    --run-dir "$RUN_DIR" \
    2>&1 | tee -a "$LOG"
_log "Stage 1 done in $(( $(date +%s) - T0 ))s"

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 – Baseline A: CSP + LDA
# ─────────────────────────────────────────────────────────────────────────────
_banner "Stage 2/8 – Baseline A: CSP + LDA"
T0=$(date +%s)
uv run python scripts/pipeline/stage_02_baseline_a.py \
    --run-dir "$RUN_DIR" \
    --n-folds "$N_FOLDS" --seed "$SEED" \
    2>&1 | tee -a "$LOG"
_log "Stage 2 done in $(( $(date +%s) - T0 ))s"

# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 – Baseline B: Riemannian + LDA
# ─────────────────────────────────────────────────────────────────────────────
_banner "Stage 3/8 – Baseline B: Riemannian + LDA"
T0=$(date +%s)
uv run python scripts/pipeline/stage_03_baseline_b.py \
    --run-dir "$RUN_DIR" \
    --n-folds "$N_FOLDS" --seed "$SEED" \
    2>&1 | tee -a "$LOG"
_log "Stage 3 done in $(( $(date +%s) - T0 ))s"

# ─────────────────────────────────────────────────────────────────────────────
# Stage 4 – Pretrain ViT on PhysioNet (9-channel spectrograms)
# ─────────────────────────────────────────────────────────────────────────────
_banner "Stage 4/8 – Pretrain ViT on PhysioNet"
T0=$(date +%s)
uv run python scripts/pipeline/stage_04_pretrain_vit.py \
    --run-dir "$RUN_DIR" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" --device "$DEVICE" --seed "$SEED" \
    2>&1 | tee -a "$LOG"
_log "Stage 4 done in $(( $(date +%s) - T0 ))s"

# ─────────────────────────────────────────────────────────────────────────────
# Stage 5 – ViT-only baseline (PhysioNet-pretrained)
# ─────────────────────────────────────────────────────────────────────────────
_banner "Stage 5/8 – ViT-only baseline"
T0=$(date +%s)
uv run python scripts/pipeline/stage_05_vit_baseline.py \
    --run-dir "$RUN_DIR" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" --device "$DEVICE" --seed "$SEED" \
    --n-folds "$N_FOLDS" \
    2>&1 | tee -a "$LOG"
_log "Stage 5 done in $(( $(date +%s) - T0 ))s"

# ─────────────────────────────────────────────────────────────────────────────
# Stage 6 – Dual-branch (attention + gated fusion ablation)
# ─────────────────────────────────────────────────────────────────────────────
_banner "Stage 6/8 – Dual-branch (attention + gated fusion)"
T0=$(date +%s)
uv run python scripts/pipeline/stage_06_dual_branch.py \
    --run-dir "$RUN_DIR" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" --device "$DEVICE" --seed "$SEED" \
    --n-folds "$N_FOLDS" \
    2>&1 | tee -a "$LOG"
_log "Stage 6 done in $(( $(date +%s) - T0 ))s"

# ─────────────────────────────────────────────────────────────────────────────
# Stage 7 – Reduced-data transfer learning experiment
# ─────────────────────────────────────────────────────────────────────────────
_banner "Stage 7/8 – Reduced-data transfer learning experiment"
T0=$(date +%s)
uv run python scripts/pipeline/stage_07_reduced_data.py \
    --run-dir "$RUN_DIR" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" --device "$DEVICE" --seed "$SEED" \
    --n-repeats "$N_REPEATS" \
    2>&1 | tee -a "$LOG"
_log "Stage 7 done in $(( $(date +%s) - T0 ))s"

# ─────────────────────────────────────────────────────────────────────────────
# Stage 8 – Result analysis, plotting, statistical tests
# ─────────────────────────────────────────────────────────────────────────────
_banner "Stage 8/8 – Result analysis + plots + stats"
T0=$(date +%s)
uv run python scripts/pipeline/stage_08_results.py \
    --run-dir "$RUN_DIR" \
    2>&1 | tee -a "$LOG"
_log "Stage 8 done in $(( $(date +%s) - T0 ))s"

# ─────────────────────────────────────────────────────────────────────────────
# Done
# ─────────────────────────────────────────────────────────────────────────────
TOTAL=$(( $(date +%s) - T_ALL ))
H=$(( TOTAL / 3600 ))
M=$(( (TOTAL % 3600) / 60 ))
S=$(( TOTAL % 60 ))

_banner "ALL STAGES COMPLETE  (${H}h ${M}m ${S}s)"
_log "Run directory : $RUN_DIR"
_log "Results       : $RUN_DIR/results/"
_log "Plots         : $RUN_DIR/plots/"
_log "Figures       : $RUN_DIR/figures/"
_log "Log           : $LOG"
