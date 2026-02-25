# Transfer Learning Based Motor Imagery EEG Classification

Thesis project: **Transfer Learning Based Motor Imagery EEG Classification with Reduced Data**

Binary classification of **Left vs Right Hand Motor Imagery** EEG signals using a dual-branch deep learning architecture that combines a Vision Transformer (ViT) processing CWT spectrograms with a handcrafted feature branch (CSP + Riemannian geometry). Transfer learning from an EEG-domain pretrained ViT is the central thesis contribution, demonstrating superior accuracy under data scarcity.

---

## Architecture

```
Branch A: CWT Spectrogram (224×224×3) ──> ViT-Tiny ──> 192-dim features
Branch B: CSP(6) + Riemannian(253) ──> concat(259) ──> MLP(259→256→128) ──> 128-dim
Fusion:   AttentionFusion(192, 128) ──> 128-dim
Head:     MLP(128→64→2) ──> Softmax ──> Left / Right
Total:    ~5.71M parameters
```

- **ViT:** `vit_tiny_patch16_224` from `timm` (~5.7M params, CPU/GPU compatible)
- **Spectrograms:** Morlet CWT, C3→Red, Cz→Green, C4→Blue, 224×224
- **Handcrafted features:** 6 CSP components + 253 Riemannian tangent-space features = 259-dim
- **Fusion variants:** Attention (thesis default), Concat, Gated

---

## Dataset

**Primary:** BCI Competition IV Dataset 2a (BNCI2014-001)
- 9 subjects, 288 trials each (144 left, 144 right hand MI)
- 22 EEG channels, 4s trials at 250 Hz → resampled to 128 Hz via MOABB
- Downloaded automatically to `~/mne_data/` via MOABB on first run

**Secondary (pretraining):** PhysioNet MMIDB or large synthetic population (20 subjects)

---

## Project Structure

```
bci_code/
├── src/bci/
│   ├── data/
│   │   ├── download.py          # MOABB data loading
│   │   ├── preprocessing.py     # MNE preprocessing pipeline
│   │   ├── transforms.py        # CWT spectrogram generation
│   │   ├── dataset.py           # PyTorch datasets
│   │   └── dual_branch_builder.py  # Per-fold feature extraction (no leakage)
│   ├── features/
│   │   ├── csp.py               # CSP with Ledoit-Wolf regularization
│   │   └── riemannian.py        # Riemannian tangent space features
│   ├── models/
│   │   ├── vit_branch.py        # ViT-Tiny branch
│   │   ├── math_branch.py       # MLP for handcrafted features
│   │   ├── fusion.py            # Attention / Concat / Gated fusion
│   │   └── dual_branch.py       # Full dual-branch model
│   ├── training/
│   │   ├── trainer.py           # Training loop, early stopping, LR schedule
│   │   ├── cross_validation.py  # Within-subject CV and LOSO CV
│   │   └── evaluation.py        # Accuracy, kappa, F1 metrics
│   └── utils/
│       ├── config.py            # Dataclass configs
│       ├── seed.py              # Reproducibility helpers
│       └── visualization.py     # Plotting utilities
├── scripts/
│   ├── baseline_a_csp_lda.py         # Baseline A: CSP + LDA
│   ├── baseline_b_riemannian.py      # Baseline B: Riemannian + LDA
│   ├── baseline_c_vit.py             # Baseline C: CWT + ViT-Tiny (no math branch)
│   ├── train_dual_branch.py          # Phase 2: Full dual-branch training
│   ├── pretrain_physionet.py         # Phase 3 Step 1: Pretrain ViT on source data
│   ├── finetune_bci_iv2a.py          # Phase 3 Step 2: Finetune (scratch/imagenet/transfer)
│   ├── reduced_data_experiment.py    # Phase 3 Step 3: Accuracy vs data fraction
│   ├── phase4_compile_results.py     # Phase 4 Step 1: Compile all results into table
│   ├── phase4_visualize.py           # Phase 4 Step 2: Generate thesis figures
│   └── phase4_stats.py               # Phase 4 Step 3: Statistical significance tests
├── tests/
│   ├── test_phase1.py    # 18 tests
│   ├── test_phase2.py    # 17 tests
│   └── test_phase3.py    # 14 tests
├── results/              # JSON result files (auto-created by scripts)
├── checkpoints/          # Model checkpoints (auto-created)
├── figures/              # Generated figures (auto-created by visualize script)
└── pyproject.toml
```

---

## Setup

**Requirements:** Python 3.13, `uv` package manager.

```bash
# Install all dependencies
uv sync

# Run tests to verify everything works
uv run pytest tests/ -v
# Expected: 49/49 tests pass
```

**Note:** The `.venv` is not indexed by the editor LSP — import errors shown in the editor are false positives. All imports work correctly at runtime.

---

## Running the Experiments

All scripts support `--data synthetic` (fast, no download) and `--data real` (BCI IV-2a).  
Real data is downloaded automatically on first run via MOABB to `~/mne_data/`.

### Phase 1: Baselines

```bash
# Baseline A: CSP + LDA (fast, ~2 min on real data)
uv run python scripts/baseline_a_csp_lda.py \
    --data real \
    --n-folds 5 \
    --output results/real_baseline_a_csp_lda.json

# Baseline B: Riemannian + LDA (fast, ~1 min on real data)
uv run python scripts/baseline_b_riemannian.py \
    --data real \
    --n-folds 5 \
    --output results/real_baseline_b_riemannian.json

# Baseline C: CWT + ViT-Tiny (slow: ~60-90 min on CPU, ~15 min on GPU)
uv run python scripts/baseline_c_vit.py \
    --data real \
    --n-folds 5 \
    --epochs 50 \
    --output results/real_baseline_c_vit.json
```

**Expected real-data results (Baseline A, already run):**
- Within-subject 5-fold: 79.32% ± 12.53%, κ=0.586
- LOSO cross-subject:    65.78% ± 10.84%, κ=0.316

**Expected real-data results (Baseline B, already run):**
- Within-subject 5-fold: 61.65% ± 9.41%, κ=0.233
- LOSO cross-subject:    63.85% ± 11.14%, κ=0.277

**Expected real-data results (Baseline C, already run):**
- Within-subject 5-fold: 52.58% ± 8.10%, κ=0.052

### Phase 2: Dual-Branch Model + Fusion Ablation

Run all three fusion variants. Each takes ~2-4h on CPU, ~30-60 min on GPU.

```bash
for FUSION in attention concat gated; do
    uv run python scripts/train_dual_branch.py \
        --data real \
        --strategy within_subject \
        --n-folds 5 \
        --epochs 50 \
        --fusion $FUSION \
        --output results/real_dual_branch_${FUSION}.json
done
```

For LOSO (cross-subject) evaluation:
```bash
uv run python scripts/train_dual_branch.py \
    --data real \
    --strategy loso \
    --epochs 50 \
    --fusion attention \
    --output results/real_dual_branch_attention_loso.json
```

### Phase 3: Transfer Learning

**Step 1 – Pretrain ViT on source data (PhysioNet or synthetic):**
```bash
# Using synthetic source population (no extra download needed)
uv run python scripts/pretrain_physionet.py \
    --data synthetic \
    --n-subjects 20 \
    --epochs 50 \
    --no-pretrained \
    --checkpoint checkpoints/vit_pretrained_eeg.pt \
    --output results/pretrain_eeg.json
```

**Step 2 – Compare scratch / ImageNet / EEG-pretrained ViT on target:**
```bash
uv run python scripts/finetune_bci_iv2a.py \
    --data real \
    --checkpoint checkpoints/vit_pretrained_eeg.pt \
    --conditions scratch imagenet transfer \
    --n-folds 5 \
    --epochs 50 \
    --output-dir results/
# Outputs: results/real_finetune_scratch.json
#          results/real_finetune_imagenet.json
#          results/real_finetune_transfer.json
```

**Step 3 – Reduced-data experiment (core thesis result):**
```bash
uv run python scripts/reduced_data_experiment.py \
    --data real \
    --checkpoint checkpoints/vit_pretrained_eeg.pt \
    --conditions scratch transfer \
    --fractions 0.10 0.25 0.50 0.75 1.00 \
    --n-folds 5 \
    --n-repeats 3 \
    --epochs 50 \
    --output results/real_reduced_data_results.json
```

### Phase 4: Analysis (run after all experiments)

**Step 1 – Compile all results into a unified table:**
```bash
# For real-data results (prefix "real_")
uv run python scripts/phase4_compile_results.py \
    --results-dir results/ \
    --prefix real_ \
    --output results/phase4_summary.json
```

**Step 2 – Generate all thesis figures:**
```bash
uv run python scripts/phase4_visualize.py \
    --summary results/phase4_summary.json \
    --data real \
    --output-dir figures/
# Outputs: figures/fig1_cwt_spectrograms.png
#          figures/fig2_reduced_data_curves.png
#          figures/fig3_fusion_ablation.png
#          figures/fig4_per_subject_heatmap.png
#          figures/fig5_baseline_comparison.png
```

**Step 3 – Statistical significance tests:**
```bash
uv run python scripts/phase4_stats.py \
    --summary results/phase4_summary.json \
    --output results/phase4_stats.json
# Runs Wilcoxon signed-rank and paired t-tests across subjects
# Reports Cohen's d effect sizes and significance stars
```

---

## Quick Smoke Tests (Synthetic Data)

These run in seconds/minutes and verify the full pipeline without downloading anything:

```bash
# Phase 1 baselines
uv run python scripts/baseline_a_csp_lda.py --n-subjects 3 --n-folds 2
uv run python scripts/baseline_b_riemannian.py --n-subjects 3 --n-folds 2
uv run python scripts/baseline_c_vit.py --n-subjects 2 --n-folds 2 --epochs 3 --no-pretrained

# Phase 2 dual-branch
uv run python scripts/train_dual_branch.py \
    --n-subjects 2 --n-folds 2 --epochs 3 --no-pretrained --batch-size 4

# Phase 3 transfer learning
uv run python scripts/pretrain_physionet.py \
    --data synthetic --n-subjects 5 --epochs 5 --no-pretrained \
    --checkpoint checkpoints/test_ckpt.pt
uv run python scripts/finetune_bci_iv2a.py \
    --data synthetic --checkpoint checkpoints/test_ckpt.pt \
    --n-subjects 2 --n-folds 2 --epochs 5
uv run python scripts/reduced_data_experiment.py \
    --data synthetic --checkpoint checkpoints/test_ckpt.pt \
    --n-subjects 2 --n-folds 2 --n-repeats 1 --epochs 5 \
    --fractions 0.50 1.00
```

---

## Reproducibility

- All random seeds are fixed via `--seed 42` (default for all scripts)
- No data leakage: CSP and Riemannian estimators are fit exclusively on training folds inside `DualBranchFoldBuilder.build_fold()`
- Data splits use `StratifiedKFold` for class-balanced folds

---

## Results Summary (Real BCI IV-2a Data)

Results from experiments already completed on this machine:

| Model | Within-Subject Acc | κ |
|---|---|---|
| Baseline A: CSP+LDA | 79.32% ± 12.53% | 0.586 |
| Baseline B: Riemannian+LDA | 61.65% ± 9.41% | 0.233 |
| Baseline C: CWT+ViT-Tiny | 52.58% ± 8.10% | 0.052 |
| Dual-Branch (Attention) | — *run on GPU* | — |
| Dual-Branch (Concat) | — *run on GPU* | — |
| Dual-Branch (Gated) | — *run on GPU* | — |
| Transfer: Scratch | — *run on GPU* | — |
| Transfer: ImageNet | — *run on GPU* | — |
| Transfer: EEG-Pretrained | — *run on GPU* | — |

LOSO cross-subject:

| Model | LOSO Acc | κ |
|---|---|---|
| Baseline A: CSP+LDA | 65.78% ± 10.84% | 0.316 |
| Baseline B: Riemannian+LDA | 63.85% ± 11.14% | 0.277 |

*Fill in GPU results after running the remaining experiments.*

---

## Phase Reports

- [PHASE_0_REPORT.md](PHASE_0_REPORT.md) — Project setup and infrastructure
- [PHASE_1_REPORT.md](PHASE_1_REPORT.md) — Baseline results
- [PHASE_2_REPORT.md](PHASE_2_REPORT.md) — Dual-branch model and fusion ablation
- [PHASE_3_REPORT.md](PHASE_3_REPORT.md) — Transfer learning results
