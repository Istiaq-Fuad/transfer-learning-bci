#!/usr/bin/env bash
# =============================================================================
# run_experiment.sh  –  BCI Thesis Experiment Runner
# =============================================================================
#
# Run the full overnight pipeline OR individual stages one at a time.
#
# QUICK START
# -----------
#   # Full overnight run (all 11 stages):
#   bash scripts/run_experiment.sh
#
#   # Single stage (e.g. just Baseline A):
#   bash scripts/run_experiment.sh baseline_a
#
#   # Multiple specific stages:
#   bash scripts/run_experiment.sh baseline_a baseline_b dual_attention phase4
#
#   # Resume a previous run (skips completed stages):
#   bash scripts/run_experiment.sh --run-dir runs/2024-01-15_143022
#
#   # Resume and run only one stage from that dir:
#   bash scripts/run_experiment.sh --run-dir runs/2024-01-15_143022 baseline_c
#
#   # Override hyperparameters for a quick test:
#   bash scripts/run_experiment.sh --epochs 10 --batch-size 8 baseline_a
#
# AVAILABLE STAGES
# ----------------
#   download          Stage 1  – Download/verify BCI IV-2a + PhysioNet
#   baseline_a        Stage 2  – CSP + LDA (within-subject + LOSO)
#   baseline_b        Stage 3  – Riemannian + LDA (within-subject + LOSO)
#   baseline_c        Stage 4  – CWT + ViT-Tiny (within-subject)
#   dual_attention    Stage 5  – Dual-branch, attention fusion (within + LOSO)
#   dual_concat       Stage 6  – Dual-branch, concat fusion (within-subject)
#   dual_gated        Stage 7  – Dual-branch, gated fusion (within-subject)
#   pretrain          Stage 8  – Pretrain ViT on PhysioNet
#   finetune          Stage 9  – Finetune: scratch / imagenet / transfer
#   reduced_data      Stage 10 – Reduced-data transfer learning experiment
#   phase4            Stage 11 – Compile results + figures + stats
#   all               Run all 11 stages in order (default when no stage given)
#
# OPTIONS
#   --run-dir DIR     Reuse/resume an existing run directory
#   --data-dir DIR    MNE data directory  (default: ~/mne_data)
#   --device DEVICE   pytorch device     (default: auto)
#   --epochs N        Max epochs/fold    (default: 50)
#   --batch-size N    Batch size         (default: 32)
#   --n-folds N       CV folds           (default: 5)
#   --n-repeats N     Repeats for reduced-data experiment (default: 3)
#   --pretrain-subjects N  PhysioNet subjects for pretraining (default: all)
#   --seed N          Random seed        (default: 42)
#   --dry-run         Verify data then exit without training
#
# NOTES
# -----
#   • Must be run from the project root (where pyproject.toml lives).
#   • Requires `uv` to be installed and `uv sync` already run.
#   • Each stage writes results under a timestamped run directory
#     (e.g. runs/2024-01-15_143022/). If you resume with --run-dir,
#     completed stages are automatically skipped (the Python script
#     checks for existing result files).
#   • All output (stdout + stderr) is also written to
#     <run_dir>/shell.log alongside the Python experiment.log.
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
RUN_DIR=""
DATA_DIR="~/mne_data"
DEVICE="auto"
EPOCHS=50
BATCH_SIZE=32
N_FOLDS=5
N_REPEATS=3
PRETRAIN_SUBJECTS=""   # empty = all 109 subjects
SEED=42
DRY_RUN=0

# ---------------------------------------------------------------------------
# Parse options and stage names
# ---------------------------------------------------------------------------
STAGES=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-dir)           RUN_DIR="$2";            shift 2 ;;
        --data-dir)          DATA_DIR="$2";            shift 2 ;;
        --device)            DEVICE="$2";              shift 2 ;;
        --epochs)            EPOCHS="$2";              shift 2 ;;
        --batch-size)        BATCH_SIZE="$2";          shift 2 ;;
        --n-folds)           N_FOLDS="$2";             shift 2 ;;
        --n-repeats)         N_REPEATS="$2";           shift 2 ;;
        --pretrain-subjects) PRETRAIN_SUBJECTS="$2";   shift 2 ;;
        --seed)              SEED="$2";                shift 2 ;;
        --dry-run)           DRY_RUN=1;                shift   ;;
        -*)
            echo "Unknown option: $1" >&2
            echo "Run: bash scripts/run_experiment.sh --help  for usage." >&2
            exit 1
            ;;
        *)
            STAGES+=("$1")
            shift
            ;;
    esac
done

# Default: run all stages
if [[ ${#STAGES[@]} -eq 0 ]]; then
    STAGES=(all)
fi

# ---------------------------------------------------------------------------
# Verify we're in the project root
# ---------------------------------------------------------------------------
if [[ ! -f "pyproject.toml" ]]; then
    echo "ERROR: Run this script from the project root (where pyproject.toml lives)." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Create / reuse run directory
# ---------------------------------------------------------------------------
if [[ -n "$RUN_DIR" ]]; then
    mkdir -p "$RUN_DIR"
    echo "Resuming run in: $RUN_DIR"
else
    TS=$(date +"%Y-%m-%d_%H%M%S")
    RUN_DIR="runs/$TS"
    mkdir -p "$RUN_DIR"
    echo "New run directory: $RUN_DIR"
fi

SHELL_LOG="$RUN_DIR/shell.log"

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------
_log() {
    local msg="[$(date '+%H:%M:%S')] $*"
    echo "$msg"
    echo "$msg" >> "$SHELL_LOG"
}

_banner() {
    local line="======================================================================"
    _log "$line"
    _log "  $*"
    _log "$line"
}

_skip() {
    local line="----------------------------------------------------------------------"
    _log "$line"
    _log "  SKIP: $*"
    _log "$line"
}

# ---------------------------------------------------------------------------
# Build shared Python flags
# ---------------------------------------------------------------------------
PY_COMMON=(
    --run-dir  "$RUN_DIR"
    --data-dir "$DATA_DIR"
    --device   "$DEVICE"
    --seed     "$SEED"
    --n-folds  "$N_FOLDS"
    --epochs   "$EPOCHS"
    --batch-size "$BATCH_SIZE"
    --n-repeats  "$N_REPEATS"
)
if [[ -n "$PRETRAIN_SUBJECTS" ]]; then
    PY_COMMON+=(--pretrain-subjects "$PRETRAIN_SUBJECTS")
fi
if [[ $DRY_RUN -eq 1 ]]; then
    PY_COMMON+=(--dry-run)
fi

# The individual-stage flags for run_full_experiment.py
SKIP_DOWNLOAD=1
SKIP_BASELINES=1
SKIP_DUAL=1
SKIP_TRANSFER=1
SKIP_PHASE4=1

# ---------------------------------------------------------------------------
# Determine which stages to enable
# ---------------------------------------------------------------------------
for stage in "${STAGES[@]}"; do
    case "$stage" in
        all)
            SKIP_DOWNLOAD=0
            SKIP_BASELINES=0
            SKIP_DUAL=0
            SKIP_TRANSFER=0
            SKIP_PHASE4=0
            ;;
        download)      SKIP_DOWNLOAD=0  ;;
        baseline_a)    SKIP_BASELINES=0 ;;
        baseline_b)    SKIP_BASELINES=0 ;;
        baseline_c)    SKIP_BASELINES=0 ;;
        dual_attention) SKIP_DUAL=0     ;;
        dual_concat)    SKIP_DUAL=0     ;;
        dual_gated)     SKIP_DUAL=0     ;;
        pretrain)      SKIP_TRANSFER=0  ;;
        finetune)      SKIP_TRANSFER=0  ;;
        reduced_data)  SKIP_TRANSFER=0  ;;
        phase4)        SKIP_PHASE4=0    ;;
        *)
            echo "Unknown stage: '$stage'" >&2
            echo "" >&2
            echo "Available stages: download baseline_a baseline_b baseline_c" >&2
            echo "                  dual_attention dual_concat dual_gated" >&2
            echo "                  pretrain finetune reduced_data phase4 all" >&2
            exit 1
            ;;
    esac
done

# Build skip flags for the Python script
PY_SKIP=()
[[ $SKIP_DOWNLOAD  -eq 1 ]] && PY_SKIP+=(--skip-download)
[[ $SKIP_BASELINES -eq 1 ]] && PY_SKIP+=(--skip-baselines)
[[ $SKIP_DUAL      -eq 1 ]] && PY_SKIP+=(--skip-dual-branch)
[[ $SKIP_TRANSFER  -eq 1 ]] && PY_SKIP+=(--skip-transfer)
[[ $SKIP_PHASE4    -eq 1 ]] && PY_SKIP+=(--skip-phase4)

# ---------------------------------------------------------------------------
# Run individual stages one by one (with per-stage banners)
# ---------------------------------------------------------------------------
T_TOTAL_START=$(date +%s)

# Helper: run one named stage by temporarily enabling only that stage's group
run_stage() {
    local label="$1"
    local skip_flag="$2"   # the --skip-* flag that would suppress this group
    shift 2
    # Build flags: all skips EXCEPT the one we want to run
    local flags=()
    [[ $SKIP_DOWNLOAD  -eq 1 ]] && flags+=(--skip-download)
    [[ $SKIP_BASELINES -eq 1 ]] && flags+=(--skip-baselines)
    [[ $SKIP_DUAL      -eq 1 ]] && flags+=(--skip-dual-branch)
    [[ $SKIP_TRANSFER  -eq 1 ]] && flags+=(--skip-transfer)
    [[ $SKIP_PHASE4    -eq 1 ]] && flags+=(--skip-phase4)
    # Remove the flag for this group so it runs
    flags=("${flags[@]/$skip_flag/}")

    _banner "$label"
    T0=$(date +%s)
    uv run python scripts/run_full_experiment.py \
        "${PY_COMMON[@]}" \
        "${flags[@]}" \
        2>&1 | tee -a "$SHELL_LOG"
    T1=$(date +%s)
    _log "$label done in $(( T1 - T0 ))s"
}

# If running all stages at once, just invoke the Python script once with all flags
if [[ ${STAGES[0]} == "all" && ${#STAGES[@]} -eq 1 ]]; then
    _banner "FULL OVERNIGHT RUN (all 11 stages)"
    _log "Settings: epochs=$EPOCHS  batch=$BATCH_SIZE  folds=$N_FOLDS  device=$DEVICE"
    T0=$(date +%s)
    uv run python scripts/run_full_experiment.py \
        "${PY_COMMON[@]}" \
        2>&1 | tee -a "$SHELL_LOG"
    T1=$(date +%s)
    _log "All stages done in $(( T1 - T0 ))s"
else
    # Run only the requested stages in canonical order
    _banner "SELECTIVE RUN: ${STAGES[*]}"
    _log "Settings: epochs=$EPOCHS  batch=$BATCH_SIZE  folds=$N_FOLDS  device=$DEVICE"

    for stage in "${STAGES[@]}"; do
        case "$stage" in
            download)
                _banner "Stage 1/11: Download / verify datasets"
                T0=$(date +%s)
                uv run python scripts/run_full_experiment.py \
                    "${PY_COMMON[@]}" \
                    --skip-baselines --skip-dual-branch \
                    --skip-transfer  --skip-phase4 \
                    2>&1 | tee -a "$SHELL_LOG"
                _log "download done in $(( $(date +%s) - T0 ))s"
                ;;

            baseline_a)
                _banner "Stage 2/11: Baseline A – CSP + LDA"
                T0=$(date +%s)
                uv run python scripts/run_full_experiment.py \
                    "${PY_COMMON[@]}" \
                    --skip-download --skip-dual-branch \
                    --skip-transfer --skip-phase4 \
                    2>&1 | tee -a "$SHELL_LOG"
                _log "baseline_a done in $(( $(date +%s) - T0 ))s"
                ;;

            baseline_b)
                _banner "Stage 3/11: Baseline B – Riemannian + LDA"
                T0=$(date +%s)
                uv run python scripts/run_full_experiment.py \
                    "${PY_COMMON[@]}" \
                    --skip-download --skip-dual-branch \
                    --skip-transfer --skip-phase4 \
                    2>&1 | tee -a "$SHELL_LOG"
                _log "baseline_b done in $(( $(date +%s) - T0 ))s"
                ;;

            baseline_c)
                _banner "Stage 4/11: Baseline C – CWT + ViT-Tiny"
                T0=$(date +%s)
                uv run python scripts/run_full_experiment.py \
                    "${PY_COMMON[@]}" \
                    --skip-download --skip-dual-branch \
                    --skip-transfer --skip-phase4 \
                    2>&1 | tee -a "$SHELL_LOG"
                _log "baseline_c done in $(( $(date +%s) - T0 ))s"
                ;;

            dual_attention)
                _banner "Stage 5/11: Dual-branch, attention fusion (within + LOSO)"
                T0=$(date +%s)
                uv run python scripts/run_full_experiment.py \
                    "${PY_COMMON[@]}" \
                    --skip-download --skip-baselines \
                    --skip-transfer --skip-phase4 \
                    2>&1 | tee -a "$SHELL_LOG"
                _log "dual_attention done in $(( $(date +%s) - T0 ))s"
                ;;

            dual_concat)
                _banner "Stage 6/11: Dual-branch, concat fusion"
                T0=$(date +%s)
                uv run python scripts/run_full_experiment.py \
                    "${PY_COMMON[@]}" \
                    --skip-download --skip-baselines \
                    --skip-transfer --skip-phase4 \
                    2>&1 | tee -a "$SHELL_LOG"
                _log "dual_concat done in $(( $(date +%s) - T0 ))s"
                ;;

            dual_gated)
                _banner "Stage 7/11: Dual-branch, gated fusion"
                T0=$(date +%s)
                uv run python scripts/run_full_experiment.py \
                    "${PY_COMMON[@]}" \
                    --skip-download --skip-baselines \
                    --skip-transfer --skip-phase4 \
                    2>&1 | tee -a "$SHELL_LOG"
                _log "dual_gated done in $(( $(date +%s) - T0 ))s"
                ;;

            pretrain)
                _banner "Stage 8/11: Pretrain ViT on PhysioNet"
                T0=$(date +%s)
                uv run python scripts/run_full_experiment.py \
                    "${PY_COMMON[@]}" \
                    --skip-download --skip-baselines \
                    --skip-dual-branch --skip-phase4 \
                    2>&1 | tee -a "$SHELL_LOG"
                _log "pretrain done in $(( $(date +%s) - T0 ))s"
                ;;

            finetune)
                _banner "Stage 9/11: Finetune comparison (scratch / imagenet / transfer)"
                T0=$(date +%s)
                uv run python scripts/run_full_experiment.py \
                    "${PY_COMMON[@]}" \
                    --skip-download --skip-baselines \
                    --skip-dual-branch --skip-phase4 \
                    2>&1 | tee -a "$SHELL_LOG"
                _log "finetune done in $(( $(date +%s) - T0 ))s"
                ;;

            reduced_data)
                _banner "Stage 10/11: Reduced-data transfer learning experiment"
                T0=$(date +%s)
                uv run python scripts/run_full_experiment.py \
                    "${PY_COMMON[@]}" \
                    --skip-download --skip-baselines \
                    --skip-dual-branch --skip-phase4 \
                    2>&1 | tee -a "$SHELL_LOG"
                _log "reduced_data done in $(( $(date +%s) - T0 ))s"
                ;;

            phase4)
                _banner "Stage 11/11: Phase 4 – Compile + visualize + stats"
                T0=$(date +%s)
                uv run python scripts/run_full_experiment.py \
                    "${PY_COMMON[@]}" \
                    --skip-download --skip-baselines \
                    --skip-dual-branch --skip-transfer \
                    2>&1 | tee -a "$SHELL_LOG"
                _log "phase4 done in $(( $(date +%s) - T0 ))s"
                ;;
        esac
    done
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
T_END=$(date +%s)
TOTAL=$(( T_END - T_TOTAL_START ))
H=$(( TOTAL / 3600 ))
M=$(( (TOTAL % 3600) / 60 ))
S=$(( TOTAL % 60 ))

_banner "DONE  (${H}h ${M}m ${S}s)"
_log "Run directory : $RUN_DIR"
_log "Results       : $RUN_DIR/results/"
_log "Figures       : $RUN_DIR/figures/"
_log "Shell log     : $SHELL_LOG"
_log "Python log    : $RUN_DIR/experiment.log"
