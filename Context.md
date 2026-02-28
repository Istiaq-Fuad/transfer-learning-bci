# BCI Transfer Learning — Codebase Context

This document describes the full project for an LLM reader: goal, data flow, datasets, preprocessing, models, pipeline stages, test suite, and key constants.

---

## Project Goal

Classify motor imagery (MI) EEG signals — specifically left-hand vs. right-hand imagined movements — using a **dual-branch deep learning architecture** that combines:

1. **A ViT-Tiny image branch** operating on 9-channel CWT spectrograms of EEG.
2. **A handcrafted math branch** (CSP + Riemannian tangent-space features → MLP).

The central thesis question is whether **domain transfer** (pre-training the ViT backbone on the large PhysioNet MMIDB dataset before fine-tuning on the smaller BCI Competition IV-2a dataset) improves classification accuracy, especially under data-scarce conditions.

**Target dataset**: BCI Competition IV-2a (9 subjects, ~288 trials/subject).  
**Source dataset**: PhysioNet Motor Movement/Imagery Database (109 subjects).

---

## Directory Layout

```
transfer-learning-bci/
├── configs/
│   ├── default.yaml           # Default experiment config (19-channel MOABB load + model params)
│   └── dataset/
│       ├── bci_iv2a.yaml      # BCI IV-2a baseline config (no ICA, h_freq=30 Hz)
│       └── physionet.yaml     # PhysioNet config (19 channels, 60 Hz notch)
├── scripts/
│   ├── run_all.sh             # PRIMARY orchestrator: runs all 8 stages in order (bash)
│   ├── run_full_experiment.py # Python orchestrator: same pipeline, adds --dry-run/--skip-*
│   ├── smoke_test.py          # Quick sanity check (no full data needed)
│   ├── __init__.py
│   └── pipeline/
│       ├── __init__.py
│       ├── stage_01_download.py
│       ├── stage_02_baseline_a.py
│       ├── stage_03_baseline_b.py
│       ├── stage_04_pretrain_vit.py
│       ├── stage_05_vit_baseline.py
│       ├── stage_06_dual_branch.py
│       ├── stage_07_reduced_data.py
│       └── stage_08_results.py
├── src/bci/
│   ├── data/
│   │   ├── download.py          # MOABB download + .npz caching + spectrogram I/O
│   │   ├── preprocessing.py     # EEG bandpass, epoch rejection, z-score, euclidean alignment
│   │   ├── transforms.py        # CWTSpectrogramTransform (raw EEG → 9-ch spectrogram tensor)
│   │   ├── augmentation.py      # EEG + spectrogram augmentation (noise, crop, SpecAugment)
│   │   ├── dual_branch_builder.py # Builds (spec_tensor, math_features, labels) TensorDataset
│   │   └── __init__.py
│   ├── features/
│   │   ├── csp.py               # CSPFeatureExtractor (MNE CSP, Ledoit-Wolf, 6 components)
│   │   ├── riemannian.py        # RiemannianFeatureExtractor (OAS covariance, Riemannian metric)
│   │   └── __init__.py
│   ├── models/
│   │   ├── vit_branch.py        # ViTBranch: timm ViT-Tiny, in_chans=9, feature extractor or classifier
│   │   ├── math_branch.py       # MathBranch: MLP(input_dim → [256,128] → 128)
│   │   ├── fusion.py            # ConcatFusion, AttentionFusion, GatedFusion
│   │   ├── dual_branch.py       # DualBranchModel: ViTBranch + MathBranch + Fusion + ClassifierHead
│   │   └── __init__.py
│   ├── training/
│   │   ├── trainer.py           # Trainer: AdamW + cosine LR + warm-up + early stopping
│   │   ├── cross_validation.py  # within_subject_cv_all, loso_cv; CVResult, FoldResult dataclasses
│   │   └── evaluation.py        # compute_metrics → {accuracy, kappa, f1_macro}
│   └── utils/
│       ├── config.py            # All dataclass configs + SpectrogramConfig (9-channel list)
│       ├── logging.py           # setup_stage_logging helper
│       ├── visualization.py     # save_fold_plots (training curves + confusion matrix)
│       └── seed.py              # set_seed, get_device
├── tests/
│   ├── test_phase1.py           # Unit tests for Stage 02/03 helpers (CSP+LDA, Riemannian+LDA)
│   ├── test_phase2.py           # Unit tests for Stage 04/05 helpers (ViTBranch, CWT transform)
│   └── test_phase3.py           # Unit tests for Stage 06/07 helpers (DualBranch, reduced-data)
└── pyproject.toml               # uv/ruff/pytest config; pythonpath = ["src", "scripts"]
```

---

## Key Constants

| Constant | Value | Where defined |
|---|---|---|
| Bandpass filter | 8–32 Hz | All pipeline stages; `SpectrogramConfig.freq_min/max` |
| Spectrogram channels | `["C3","C1","Cz","C2","C4","FC3","FC4","CP3","CP4"]` | `SpectrogramConfig.spectrogram_channels` in `config.py`; hardcoded in stage_08 figure |
| Number of spec channels | 9 | `ModelConfig.in_chans = 9`; all stages use `in_chans=9` |
| Spectrogram image size | 224 × 224 | `SpectrogramConfig.image_size` |
| CWT wavelet | Morlet (`"morl"`) | `SpectrogramConfig.wavelet` |
| CSP components | 6 | `ModelConfig.csp_n_components`; `CSPFeatureExtractor(n_components=6)` |
| CSP regularisation | Ledoit-Wolf | `CSPFeatureExtractor(reg="ledoit_wolf")` |
| Riemannian covariance estimator | OAS | `RiemannianFeatureExtractor(estimator="oas")` |
| Riemannian metric | `"riemann"` | `RiemannianFeatureExtractor(metric="riemann")` |
| LDA solver | `lsqr`, `shrinkage="auto"` | stages 02, 03 |
| Sampling frequency | 128 Hz | `DualBranchFoldBuilder(sfreq=128.0)` |
| ViT backbone | `vit_tiny_patch16_224` | `ModelConfig.vit_model_name` |
| ViT feature dim | 192 | `FEATURE_DIM` in `vit_branch.py` |
| Fusion output dim | 128 | `DEFAULT_FUSED_DIM` in `config.py` |
| Classifier hidden dim | 64 | `DEFAULT_CLS_HIDDEN` in `config.py` |
| MathBranch hidden dims | `[256, 128]` | `ModelConfig.math_hidden_dims` |
| Optimizer | AdamW | `Trainer` |
| LR scheduler | Cosine with warm-up | `Trainer` |
| Label smoothing | 0.1 | `Trainer` |
| Default seed | 42 | all stages |
| BCI IV-2a subjects | 9 (S01–S09) | MOABB |
| PhysioNet subjects | up to 109 | MOABB |
| Classes | left hand, right hand | binary 2-class MI |

---

## Datasets

### BCI Competition IV-2a (target domain)
- 9 healthy subjects, 4-class MI paradigm (left hand, right hand, feet, tongue).
- **Project uses only left-hand vs. right-hand** (binary).
- ~288 trials per subject (≈144 per class) in training session.
- 22 EEG channels at 250 Hz, downsampled to 128 Hz.
- Loaded via MOABB (`BNCI2014_001`).
- Epoch window: 0.5–3.5 s after cue (3 s, 384 samples at 128 Hz).

### PhysioNet MMIDB (source domain)
- 109 subjects, imagined L/R hand + feet MI tasks.
- 64 EEG channels at 160 Hz, downsampled to 128 Hz.
- Loaded via MOABB (`PhysionetMI`).
- Used only for **pre-training the ViT backbone** (Stage 04). Not used in evaluation.
- 60 Hz notch filter (US power line frequency).

---

## Data Flow

```
Stage 01: MOABB download → preprocess → .npz epoch cache + spectrogram cache
                                               │
           ┌───────────────────────────────────┤
           ▼                                   ▼
Stage 02: Epoch cache (.npz)          Stage 04: Spectrogram cache (PhysioNet)
          → CSP+LDA CV                         → pretrain ViT backbone → checkpoint
Stage 03: Epoch cache (.npz)                   │
          → Riemannian+LDA CV         Stage 05: BCI spectrogram cache + checkpoint
                                               → ViT-only fine-tune CV
                                      Stage 06: Epoch cache + spec cache + checkpoint
                                               → DualBranch ablation (attention / gated)
                                      Stage 07: Epoch cache + spec cache + checkpoint
                                               → reduced-data transfer experiment
                                               │
                                               ▼
                                      Stage 08: All result JSONs
                                               → tables + stats + 5 figures
```

All stages read from `data/processed/<dataset>/subject_NN.npz` (epochs) or `subject_NN_spectrograms.npz` (spectrograms), both written by Stage 01. All result JSONs go to `<run-dir>/results/`. Checkpoints go to `<run-dir>/checkpoints/`.

---

## Preprocessing (Stage 01 / `src/bci/data/`)

1. **MOABB download**: epochs loaded via `process_and_cache()` in `download.py`. Applies 8–32 Hz bandpass + 50/60 Hz notch during MOABB epoch extraction.
2. **Epoch saving**: per-subject `(X, y, channel_names, sfreq)` saved to `.npz`.
3. **Spectrogram generation**: `process_and_cache_spectrograms()` calls `CWTSpectrogramTransform` to compute 9-channel CWT spectrograms for each trial. Saved as `subject_NN_spectrograms.npz` with keys `images` (N, 9, 224, 224) and `y` (N,).
4. **Stats computation**: per-channel mean/std computed across all training subjects; saved as `spectrogram_stats.npz`. Used to normalise spectrograms in later stages.

`CWTSpectrogramTransform` (`transforms.py`):
- Uses PyWavelets `pywt.cwt` with Morlet wavelet over 64 log-spaced frequencies in 8–32 Hz.
- Extracts only the 9 spectrogram channels from the full electrode set.
- Magnitude of complex CWT output; clipped and normalised to [0, 1] per channel ("joint" mode preserves laterality).
- Output shape: `(N, 9, 224, 224)` float32.

---

## Feature Extractors (`src/bci/features/`)

### CSPFeatureExtractor (`csp.py`)
- Wraps MNE's `mne.decoding.CSP`.
- **6 spatial filters**, Ledoit-Wolf covariance regularisation.
- `fit_transform(X, y)` → log-variance features, shape `(N, 2*n_components)` = `(N, 12)`.

### RiemannianFeatureExtractor (`riemannian.py`)
- Uses pyRiemann `TangentSpace` with OAS covariance estimation and Riemannian metric.
- `fit_transform(X, y)` → tangent-space vector, shape `(N, n_channels*(n_channels+1)/2)`.

Both implement `fit`, `transform`, `fit_transform` (sklearn API).

---

## Models (`src/bci/models/`)

### ViTBranch (`vit_branch.py`)
- Wraps a `timm` ViT-Tiny backbone (`vit_tiny_patch16_224`).
- **`in_chans=9`**: patch embedding accepts 9-channel spectrograms (not RGB).
- Two modes:
  - `as_feature_extractor=True`: replaces classification head with `nn.Identity`; outputs 192-dim feature vector.
  - `as_feature_extractor=False`: keeps classification head for standalone training (Stage 04/05).
- `freeze_backbone(unfreeze_last_n_blocks=2)`: freezes all but the last 2 transformer blocks for transfer fine-tuning.
- `ModelConfig.in_chans = 9` is the canonical default.

### MathBranch (`math_branch.py`)
- MLP: `Linear(input_dim → 256) → BN → ReLU → Dropout → Linear(256 → 128) → BN → ReLU → Dropout`.
- Input dim is determined at runtime (CSP features + Riemannian features concatenated).
- Output: 128-dim embedding.

### Fusion (`fusion.py`)
Three strategies:
- **`ConcatFusion`**: concatenate ViT (192) + Math (128) → optional linear projection.
- **`AttentionFusion`**: learns scalar attention weights for each branch; computes weighted sum → Linear(fused_dim=128). Thesis primary.
- **`GatedFusion`**: element-wise sigmoid gate on concatenated features → 128-dim output. Thesis ablation.

### DualBranchModel (`dual_branch.py`)
- Combines `ViTBranch` (feature extractor, 192-dim) + `MathBranch` (128-dim) + Fusion (128-dim) + `ClassifierHead` (64 hidden → 2 classes).
- `forward(imgs, math_feats)` → class logits.
- `freeze_vit_backbone(unfreeze_last_n_blocks=2)`: delegates to `ViTBranch.freeze_backbone`.
- Input: imgs `(B, 9, 224, 224)`, math_feats `(B, math_input_dim)`.

### ClassifierHead (`dual_branch.py`)
- `Linear(fused_dim → 64) → BN → ReLU → Dropout → Linear(64 → 2)`.

---

## Training Infrastructure (`src/bci/training/`)

### Trainer (`trainer.py`)
- AdamW optimizer with optional differential LR for backbone (`backbone_lr_scale=0.1`).
- Cosine annealing LR with linear warm-up (`warmup_epochs`).
- Label smoothing cross-entropy loss.
- Validation split (`val_fraction=0.2`) drawn from training data for early stopping.
- Best model checkpoint saved in memory; restored after training.
- `fit(dataset, forward_fn, model_tag)` → `TrainResult` with `.history`, `.best_epoch`.
- `predict(dataloader, forward_fn)` → `(y_pred, y_prob)`.

### Cross-Validation (`cross_validation.py`)
- `within_subject_cv_all(subject_data, predict_fn, n_folds, seed)` → `CVResult`.
- `loso_cv(subject_data, predict_fn)` → `CVResult` (Leave-One-Subject-Out).
- `FoldResult`: per-fold accuracy, kappa, f1, y_true, y_pred, y_prob.
- `CVResult`: aggregates folds; `.mean_accuracy`, `.std_accuracy`, `.per_subject_accuracy`.

### Evaluation (`evaluation.py`)
- `compute_metrics(y_true, y_pred, y_prob)` → `{"accuracy": %, "kappa": float, "f1_macro": float}`.

---

## Pipeline Stages

### Stage 01 — Download + Cache (`stage_01_download.py`)
**Inputs**: internet (MOABB), `~/mne_data`  
**Outputs**: `data/processed/bci_iv2a/subject_NN.npz`, `subject_NN_spectrograms.npz`, `spectrogram_stats.npz`; same for `physionet/`

Downloads BCI IV-2a and PhysioNet via MOABB, applies 8–32 Hz bandpass, saves raw epoch cache, generates 9-channel CWT spectrograms, and computes per-channel normalisation stats.

---

### Stage 02 — Baseline A: CSP + LDA (`stage_02_baseline_a.py`)
**Inputs**: `data/processed/bci_iv2a/`  
**Outputs**: `results/real_baseline_a_csp_lda.json`

5-fold within-subject CV and LOSO CV using CSP (6 components, Ledoit-Wolf) + LDA (lsqr, auto shrinkage). Reports accuracy, kappa, F1, per-subject LOSO accuracy.

**Exported function**: `make_predict_fn(n_components, reg)` → `predict_fn(X_train, y_train, X_test)`.

---

### Stage 03 — Baseline B: Riemannian + LDA (`stage_03_baseline_b.py`)
**Inputs**: `data/processed/bci_iv2a/`  
**Outputs**: `results/real_baseline_b_riemannian.json`

Same CV structure as Stage 02, but uses Riemannian tangent-space features (OAS covariance) + LDA.

**Exported function**: `make_predict_fn(estimator, metric)` → `predict_fn(X_train, y_train, X_test)`.

---

### Stage 04 — Pretrain ViT on PhysioNet (`stage_04_pretrain_vit.py`)
**Inputs**: `data/processed/physionet/` spectrogram cache  
**Outputs**: `checkpoints/vit_pretrained_physionet_vit.pt`, `results/real_pretrain_physionet_vit.json`

Loads PhysioNet spectrograms (N, 9, 224, 224), normalises with per-channel stats, trains ViT-Tiny (`in_chans=9`) from ImageNet init as a binary classifier. Saves only `model.backbone.state_dict()` (the patch embed + transformer blocks, no classification head).

**Exported function**: `pretrain_vit(subject_data, channel_names, sfreq, spec_config, ...)` → metrics dict with `"val_accuracy"`. Builds CWT spectrograms on-the-fly from raw EEG (used in tests; production `main()` uses cached spectrograms).

---

### Stage 05 — ViT-Only Baseline (`stage_05_vit_baseline.py`)
**Inputs**: BCI IV-2a spectrogram cache, Stage 04 checkpoint  
**Outputs**: `results/real_baseline_c_vit.json`, `results/real_baseline_c_vit_loso.json`

Fine-tunes ViT-Tiny (`in_chans=9`) on BCI IV-2a spectrograms with PhysioNet-pretrained backbone. Freezes all except last 2 transformer blocks (`freeze_backbone(unfreeze_last_n_blocks=2)`). Runs 5-fold within-subject CV and LOSO CV. Falls back to ImageNet init if checkpoint is missing.

---

### Stage 06 — Dual-Branch Ablation (`stage_06_dual_branch.py`)
**Inputs**: BCI IV-2a epoch cache + spectrogram cache, Stage 04 checkpoint  
**Outputs**: `results/real_dual_branch_{fusion}_vit.json` × 2 fusion methods × (within + LOSO)

Full `DualBranchModel` trained with:
- Branch A: cached spectrograms (normalised) → ViTBranch (PhysioNet pretrained, last 2 blocks unfrozen).
- Branch B: raw EEG → CSP + Riemannian features via `DualBranchFoldBuilder` → MathBranch.
- Fusion methods: `attention` (primary) and `gated` (ablation).
- Differential LR: backbone 10× smaller (`backbone_lr_scale=0.1`).

**Exported functions**:
- `_build_model(condition, math_input_dim, checkpoint_path, unfreeze_last_n, fusion)` — conditions: `"scratch"` (random ViT init), `"imagenet"` (ImageNet ViT, frozen), `"transfer"` (PhysioNet ViT, frozen).
- `_train_and_eval_fold(fold_idx, subject_id, X_train, y_train, X_test, y_test, condition, checkpoint_path, builder, ...)` → `FoldResult`.

---

### Stage 07 — Reduced-Data Experiment (`stage_07_reduced_data.py`)
**Inputs**: BCI IV-2a epoch cache + spectrogram cache, Stage 04 checkpoint  
**Outputs**: `results/real_reduced_data_results_vit.json`, `plots/stage_07_vit/accuracy_vs_fraction.png`

The **core thesis experiment**. For training-data fractions `[10%, 25%, 50%, 75%, 100%]`, compares:
- **`"scratch"`**: DualBranch trained from ImageNet ViT init, no PhysioNet pretraining.
- **`"transfer"`**: DualBranch with PhysioNet-pretrained backbone, partially frozen.

Runs 5-fold within-subject CV × 3 repeats per fraction. Tests transfer learning advantage at low data regimes.

**Exported function**: `_run_one_trial(condition, X_train_full, y_train_full, X_test, y_test, fraction, builder, checkpoint_path, ...)` → accuracy float.

---

### Stage 08 — Results, Plots, Stats (`stage_08_results.py`)
**Inputs**: All JSON files from Stages 02–07  
**Outputs**: 5 thesis figures, `results/phase4_summary.json`, `results/phase4_stats.json`

Loads all result JSONs, prints a comprehensive comparison table (within-subject + LOSO), runs:
- Wilcoxon signed-rank test: DualBranch(Attention) vs each baseline.
- Paired t-test: transfer vs scratch conditions.
- Friedman test: attention vs gated fusion.
- Cohen's d effect sizes.

Generates 5 thesis figures:
1. CWT spectrogram examples (3 channels × 2 classes × 3 trials).
2. Accuracy vs. training data fraction (transfer vs scratch).
3. Fusion ablation bar chart (attention vs gated).
4. Per-subject accuracy heatmap (all models × 9 subjects).
5. Overall model comparison bar chart.

---

## Running the Pipeline

```bash
# Primary (bash, recommended)
bash scripts/run_all.sh --device cuda --epochs 50 --batch-size 32

# Python alternative (adds --dry-run, --skip-stage-N, --run-dir for resume)
uv run python scripts/run_full_experiment.py --device cuda

# Resume after crash (reuses existing run directory)
RUN_DIR=runs/2024-01-15_143022 bash scripts/run_all.sh
```

Both orchestrators:
- Create a timestamped `runs/<timestamp>/` directory.
- Run all 8 stages in order, passing `--run-dir` to each.
- Skip stages whose output files already exist (idempotent / resumable).

`run_all.sh` is the primary recommended entry point. `run_full_experiment.py` is useful for programmatic control (CI, dry-runs, selective skipping).

---

## Test Suite

All 65 tests pass (`uv run pytest tests/ -v`). `pythonpath = ["src", "scripts"]` in `pyproject.toml` puts both on `sys.path`; `scripts/__init__.py` and `scripts/pipeline/__init__.py` make stages importable as packages.

### `tests/test_phase1.py`
Unit tests for Stage 02/03 helpers. Imports from `scripts.pipeline.stage_02_baseline_a` and `scripts.pipeline.stage_03_baseline_b`. Tests `make_predict_fn`, CSP+LDA accuracy on synthetic data, Riemannian+LDA accuracy, `within_subject_cv_all`, `loso_cv`.

### `tests/test_phase2.py`
Unit tests for Stage 04/05 helpers. Tests `CWTSpectrogramTransform`, `ViTBranch` forward pass with `in_chans=9`, `Trainer` on synthetic data, `compute_metrics`, `DualBranchFoldBuilder`.

### `tests/test_phase3.py`
Unit tests for Stage 06/07 helpers. Imports from `scripts.pipeline.stage_06_dual_branch` and `scripts.pipeline.stage_07_reduced_data`. Tests `_build_model` (all 3 conditions), `_train_and_eval_fold`, `_run_one_trial`, `DualBranchModel` forward pass, `AttentionFusion`, `GatedFusion`.

---

## Config Notes

### `configs/default.yaml`
The `dataset.channels` list (19 channels) is the **MOABB load selection** — these are all motor-cortex-adjacent electrodes loaded from the raw recording. The **9-channel spectrogram subset** is separate and defined in `SpectrogramConfig.spectrogram_channels` in `src/bci/utils/config.py`. The pipeline stages do not use `configs/default.yaml` at runtime; they hard-code their parameters.

### `configs/dataset/bci_iv2a.yaml`
Has `h_freq: 30.0` (vs 32.0 elsewhere). This only affects the YAML-loaded config object, not the pipeline stages (which call `process_and_cache(..., fmax=32.0)` directly).

### `src/bci/utils/config.py` — authoritative defaults
All pipeline-relevant defaults live here as Python dataclasses:
- `SpectrogramConfig`: 9 spectrogram channels, 8–32 Hz, 224×224, Morlet, multichannel mode.
- `ModelConfig`: `in_chans=9`, `vit_tiny_patch16_224`, attention fusion, 6 CSP components.
- `PreprocessingConfig`: 128 Hz resample, 0.5–3.5 s epoch crop, z-score normalisation.

---

## Known False-Positive LSP Warnings

These are pre-existing and can be safely ignored:
- `EpochsArray.astype` / `.shape` — MNE type stubs are incomplete.
- `fit_transform` override in feature extractors — sklearn protocol not fully typed.
- `TensorDataset` tuple-unpacking in stage_06/07 — LSP can't infer tensor tuple structure.
