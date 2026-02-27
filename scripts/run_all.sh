#!/usr/bin/env bash
# =============================================================================
# run_all.sh  –  Run the full BCI thesis experiment pipeline
# =============================================================================
#
# This script runs all 11 pipeline stages in order.
# Every stage saves its output to a shared run directory so that:
#   • stages can be resumed if something crashes
#   • each stage can also be run independently (see ONE-BY-ONE section below)
#
# ─────────────────────────────────────────────────────────────────────────────
# QUICK START  (full overnight run)
# ─────────────────────────────────────────────────────────────────────────────
#
#   bash scripts/run_all.sh
#
#   # Specify GPU and backbone:
#   RUN_DIR=runs/gpu_run bash scripts/run_all.sh --device cuda --backbone efficientnet_b0
#
#   # Use ViT backbone instead:
#   bash scripts/run_all.sh --backbone vit_tiny_patch16_224 --device cuda
#
# ─────────────────────────────────────────────────────────────────────────────
# HOW TO RUN STAGES ONE BY ONE
# ─────────────────────────────────────────────────────────────────────────────
#
# First, create a run directory (pick any name):
#
#   export RUN=runs/my_experiment
#   export BB=efficientnet_b0   # or vit_tiny_patch16_224
#
# Then run each stage individually in order.
# Every stage skips itself automatically if its output file already exists,
# so it is always safe to re-run a stage.
#
#   Stage 1 – Download / verify datasets:
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
#   Stage 4 – Baseline C: CWT + ViT-Tiny (within-subject):
#     uv run python scripts/pipeline/stage_04_baseline_c.py \
#         --run-dir $RUN --backbone $BB --epochs 50 --batch-size 32 --device cuda
#
#   Stage 5 – Dual-branch, attention fusion (within-subject + LOSO):
#     uv run python scripts/pipeline/stage_05_dual_attention.py \
#         --run-dir $RUN --backbone $BB --epochs 50 --batch-size 32 --device cuda
#
#   Stage 6 – Dual-branch, concat fusion (within-subject):
#     uv run python scripts/pipeline/stage_06_dual_concat.py \
#         --run-dir $RUN --backbone $BB --epochs 50 --batch-size 32 --device cuda
#
#   Stage 7 – Dual-branch, gated fusion (within-subject):
#     uv run python scripts/pipeline/stage_07_dual_gated.py \
#         --run-dir $RUN --backbone $BB --epochs 50 --batch-size 32 --device cuda
#
#   Stage 8 – Pretrain backbone on PhysioNet MMIDB:
#     uv run python scripts/pipeline/stage_08_pretrain.py \
#         --run-dir $RUN --backbone $BB --epochs 50 --batch-size 32 --device cuda
#
#   Stage 9 – Finetune comparison (scratch / imagenet / transfer):
#     uv run python scripts/pipeline/stage_09_finetune.py \
#         --run-dir $RUN --backbone $BB --epochs 50 --batch-size 32 --device cuda
#
#   Stage 10 – Reduced-data transfer learning experiment:
#     uv run python scripts/pipeline/stage_10_reduced_data.py \
#         --run-dir $RUN --backbone $BB --epochs 50 --n-repeats 3 --device cuda
#
#   Stage 11 – Compile results, figures, and stats:
#     uv run python scripts/pipeline/stage_11_phase4.py \
#         --run-dir $RUN
#
# ─────────────────────────────────────────────────────────────────────────────
# OPTIONS (for this script)
# ─────────────────────────────────────────────────────────────────────────────
#   --backbone NAME    timm backbone name  (default: vit_tiny_patch16_224)
#                        choices: vit_tiny_patch16_224 | efficientnet_b0
#   --device DEVICE    pytorch device     (default: auto)
#   --epochs N         max epochs/fold    (default: 50)
#   --batch-size N     batch size         (default: 32)
#   --n-folds N        CV folds           (default: 5)
#   --n-repeats N      repeats for Stage 10 (default: 3)
#   --seed N           random seed        (default: 42)
#   --n-subjects N     PhysioNet subjects for Stage 8 (default: all 109)
#   --data-dir DIR     MNE data directory (default: ~/mne_data)
#   --data MODE        'real' (default) or 'synthetic' (fast CPU smoke test)
#
# Set RUN_DIR env variable to reuse an existing directory, e.g.:
#   RUN_DIR=runs/2024-01-15_143022 bash scripts/run_all.sh
#
# ─────────────────────────────────────────────────────────────────────────────
# NOTES
# ─────────────────────────────────────────────────────────────────────────────
#   • Must be run from the project root (where pyproject.toml lives).
#   • Requires `uv` to be installed and `uv sync` already run.
#   • Each stage writes results under the same run directory.
#     Completed stages are skipped on re-run (idempotent).
#   • All output is also logged to <run-dir>/run_all.log.
#   • Stages 4-10 benefit greatly from a GPU. On CPU they may take many hours.
# =============================================================================

set -euo pipefail

# ── Verify project root ────────────────────────────────────────────────────
if [[ ! -f "pyproject.toml" ]]; then
    echo "ERROR: Run this script from the project root (where pyproject.toml lives)." >&2
    exit 1
fi

# ── Defaults ──────────────────────────────────────────────────────────────
BACKBONE="vit_tiny_patch16_224"
DEVICE="auto"
EPOCHS=50
BATCH_SIZE=32
N_FOLDS=5
N_REPEATS=3
SEED=42
N_SUBJECTS=""
DATA_DIR="~/mne_data"
DATA_MODE="real"

# ── Parse flags ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --backbone)    BACKBONE="$2";    shift 2 ;;
        --device)      DEVICE="$2";      shift 2 ;;
        --epochs)      EPOCHS="$2";      shift 2 ;;
        --batch-size)  BATCH_SIZE="$2";  shift 2 ;;
        --n-folds)     N_FOLDS="$2";     shift 2 ;;
        --n-repeats)   N_REPEATS="$2";   shift 2 ;;
        --seed)        SEED="$2";        shift 2 ;;
        --n-subjects)  N_SUBJECTS="$2";  shift 2 ;;
        --data-dir)    DATA_DIR="$2";    shift 2 ;;
        --data)        DATA_MODE="$2";   shift 2 ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Run: bash scripts/run_all.sh  (with no args) to see usage in the header." >&2
            exit 1
            ;;
    esac
done

# ── Derive backbone short tag (for checkpoint path) ───────────────────────
case "$BACKBONE" in
    vit_tiny_patch16_224) BACKBONE_SHORT="vit"         ;;
    efficientnet_b0)      BACKBONE_SHORT="efficientnet" ;;
    *)                    BACKBONE_SHORT="$BACKBONE"   ;;
esac

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

# ── Shared Python flags ───────────────────────────────────────────────────
PY_BASE=(
    --run-dir      "$RUN_DIR"
    --data-dir     "$DATA_DIR"
    --device       "$DEVICE"
    --seed         "$SEED"
    --n-folds      "$N_FOLDS"
    --epochs       "$EPOCHS"
    --batch-size   "$BATCH_SIZE"
    --backbone     "$BACKBONE"
    --data         "$DATA_MODE"
)

PRETRAIN_FLAGS=("${PY_BASE[@]}")
[[ -n "$N_SUBJECTS" ]] && PRETRAIN_FLAGS+=(--n-subjects "$N_SUBJECTS")

CKPT="$RUN_DIR/checkpoints/vit_pretrained_physionet_${BACKBONE_SHORT}.pt"

# ── Print settings ────────────────────────────────────────────────────────
_banner "BCI THESIS PIPELINE  –  Full run"
_log "Run directory : $RUN_DIR"
_log "Backbone      : $BACKBONE  (short: $BACKBONE_SHORT)"
_log "Device        : $DEVICE"
_log "Epochs        : $EPOCHS"
_log "Batch size    : $BATCH_SIZE"
_log "CV folds      : $N_FOLDS"
_log "Repeats (S10) : $N_REPEATS"
_log "Seed          : $SEED"
_log "Data dir      : $DATA_DIR"
_log "Data mode     : $DATA_MODE"
_log ""

T_ALL=$(date +%s)

# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 – Download / verify datasets
# ─────────────────────────────────────────────────────────────────────────────
_banner "Stage 1/11 – Download / verify datasets"
T0=$(date +%s)
uv run python scripts/pipeline/stage_01_download.py \
    --run-dir "$RUN_DIR" --data-dir "$DATA_DIR" \
    2>&1 | tee -a "$LOG"
_log "Stage 1 done in $(( $(date +%s) - T0 ))s"

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 – Baseline A: CSP + LDA
# ─────────────────────────────────────────────────────────────────────────────
_banner "Stage 2/11 – Baseline A: CSP + LDA"
T0=$(date +%s)
uv run python scripts/pipeline/stage_02_baseline_a.py \
    --run-dir "$RUN_DIR" --data-dir "$DATA_DIR" \
    --n-folds "$N_FOLDS" --seed "$SEED" \
    2>&1 | tee -a "$LOG"
_log "Stage 2 done in $(( $(date +%s) - T0 ))s"

# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 – Baseline B: Riemannian + LDA
# ─────────────────────────────────────────────────────────────────────────────
_banner "Stage 3/11 – Baseline B: Riemannian + LDA"
T0=$(date +%s)
uv run python scripts/pipeline/stage_03_baseline_b.py \
    --run-dir "$RUN_DIR" --data-dir "$DATA_DIR" \
    --n-folds "$N_FOLDS" --seed "$SEED" \
    2>&1 | tee -a "$LOG"
_log "Stage 3 done in $(( $(date +%s) - T0 ))s"

# ─────────────────────────────────────────────────────────────────────────────
# Stage 4 – Baseline C: CWT + ViT-Tiny
# ─────────────────────────────────────────────────────────────────────────────
_banner "Stage 4/11 – Baseline C: CWT + ViT-Tiny"
T0=$(date +%s)
uv run python scripts/pipeline/stage_04_baseline_c.py \
    "${PY_BASE[@]}" \
    2>&1 | tee -a "$LOG"
_log "Stage 4 done in $(( $(date +%s) - T0 ))s"

# ─────────────────────────────────────────────────────────────────────────────
# Stage 5 – Dual-branch, attention fusion (within + LOSO)
# ─────────────────────────────────────────────────────────────────────────────
_banner "Stage 5/11 – Dual-branch, attention fusion"
T0=$(date +%s)
uv run python scripts/pipeline/stage_05_dual_attention.py \
    "${PY_BASE[@]}" \
    2>&1 | tee -a "$LOG"
_log "Stage 5 done in $(( $(date +%s) - T0 ))s"

# ─────────────────────────────────────────────────────────────────────────────
# Stage 6 – Dual-branch, concat fusion
# ─────────────────────────────────────────────────────────────────────────────
_banner "Stage 6/11 – Dual-branch, concat fusion"
T0=$(date +%s)
uv run python scripts/pipeline/stage_06_dual_concat.py \
    "${PY_BASE[@]}" \
    2>&1 | tee -a "$LOG"
_log "Stage 6 done in $(( $(date +%s) - T0 ))s"

# ─────────────────────────────────────────────────────────────────────────────
# Stage 7 – Dual-branch, gated fusion
# ─────────────────────────────────────────────────────────────────────────────
_banner "Stage 7/11 – Dual-branch, gated fusion"
T0=$(date +%s)
uv run python scripts/pipeline/stage_07_dual_gated.py \
    "${PY_BASE[@]}" \
    2>&1 | tee -a "$LOG"
_log "Stage 7 done in $(( $(date +%s) - T0 ))s"

# ─────────────────────────────────────────────────────────────────────────────
# Stage 8 – Pretrain ViT on PhysioNet
# ─────────────────────────────────────────────────────────────────────────────
_banner "Stage 8/11 – Pretrain ViT on PhysioNet MMIDB"
T0=$(date +%s)
uv run python scripts/pipeline/stage_08_pretrain.py \
    "${PRETRAIN_FLAGS[@]}" \
    2>&1 | tee -a "$LOG"
_log "Stage 8 done in $(( $(date +%s) - T0 ))s"

# ─────────────────────────────────────────────────────────────────────────────
# Stage 9 – Finetune comparison
# ─────────────────────────────────────────────────────────────────────────────
_banner "Stage 9/11 – Finetune: scratch / imagenet / transfer"
T0=$(date +%s)
uv run python scripts/pipeline/stage_09_finetune.py \
    "${PY_BASE[@]}" --checkpoint "$CKPT" \
    2>&1 | tee -a "$LOG"
_log "Stage 9 done in $(( $(date +%s) - T0 ))s"

# ─────────────────────────────────────────────────────────────────────────────
# Stage 10 – Reduced-data experiment
# ─────────────────────────────────────────────────────────────────────────────
_banner "Stage 10/11 – Reduced-data transfer learning experiment"
T0=$(date +%s)
uv run python scripts/pipeline/stage_10_reduced_data.py \
    "${PY_BASE[@]}" --checkpoint "$CKPT" --n-repeats "$N_REPEATS" \
    2>&1 | tee -a "$LOG"
_log "Stage 10 done in $(( $(date +%s) - T0 ))s"

# ─────────────────────────────────────────────────────────────────────────────
# Stage 11 – Phase 4: Compile + visualize + stats
# ─────────────────────────────────────────────────────────────────────────────
_banner "Stage 11/11 – Compile results, figures, and stats"
T0=$(date +%s)
uv run python scripts/pipeline/stage_11_phase4.py \
    --run-dir "$RUN_DIR" \
    2>&1 | tee -a "$LOG"
_log "Stage 11 done in $(( $(date +%s) - T0 ))s"

# ─────────────────────────────────────────────────────────────────────────────
# Done
# ─────────────────────────────────────────────────────────────────────────────
TOTAL=$(( $(date +%s) - T_ALL ))
H=$(( TOTAL / 3600 ))
M=$(( (TOTAL % 3600) / 60 ))
S=$(( TOTAL % 60 ))

_banner "ALL STAGES COMPLETE  (${H}h ${M}m ${S}s)"
_log "Run directory : $RUN_DIR"
_log "Backbone      : $BACKBONE  (short: $BACKBONE_SHORT)"
_log "Results       : $RUN_DIR/results/"
_log "Plots         : $RUN_DIR/plots/"
_log "Figures       : $RUN_DIR/figures/"
_log "Log           : $LOG"
