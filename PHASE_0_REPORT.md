# Phase 0 Completion Report
**Project:** Transfer Learning Based Motor Imagery EEG Classification with Reduced Data  
**Phase:** 0 — Project Setup & Foundation  
**Date:** 2026-02-25  
**Status:** COMPLETE — 10/10 smoke tests passing

---

## Objectives

Phase 0 aimed to establish a production-quality Python project structure with all
foundational components implemented and verified end-to-end before any model training begins.

---

## Deliverables

### 1. Project Structure
```
bci_code/
├── pyproject.toml              # uv-managed, hatchling build backend
├── .gitignore                  # EEG/ML-aware (data, checkpoints, *.fif, *.npy)
├── .python-version             # Python 3.13
├── configs/
│   ├── default.yaml            # Master experiment config
│   └── dataset/
│       ├── bci_iv2a.yaml       # BCI IV-2a specific overrides
│       └── physionet.yaml      # PhysioNet MMIDB specific overrides
├── src/bci/
│   ├── data/                   # Loading, preprocessing, transforms, datasets
│   ├── features/               # CSP, Riemannian feature extraction
│   ├── models/                 # ViT branch, Math branch, Fusion, Dual-branch
│   ├── training/               # Trainer stub, evaluation metrics
│   └── utils/                  # Config loader, seed/device, visualization
├── scripts/
│   └── smoke_test.py           # End-to-end integration test
└── tests/                      # Unit test directory (populated in Phase 1)
```

**Total source lines:** 2,822 across 19 Python files + 3 YAML configs

---

### 2. Dependencies (`pyproject.toml`)

| Category | Package | Version | Purpose |
|---|---|---|---|
| Deep Learning | `torch` | ≥2.5.0 | Core framework, GPU acceleration |
| Deep Learning | `torchvision` | ≥0.20.0 | Image transforms, augmentations |
| Deep Learning | `timm` | ≥1.0.0 | Pretrained ViT-Tiny backbone |
| EEG Processing | `mne` | ≥1.8.0 | Filtering, ICA, epoching, CSP |
| EEG Processing | `moabb` | ≥1.2.0 | Standardised dataset access (BCI IV-2a, PhysioNet) |
| Signal Processing | `PyWavelets` | ≥1.8.0 | CWT with Morlet wavelet |
| Riemannian | `pyriemann` | ≥0.7 | SPD covariance, tangent space projection |
| Scientific | `numpy`, `scipy`, `scikit-learn` | latest | Numerics, ML utilities |
| Visualization | `matplotlib`, `seaborn` | latest | Plots, confusion matrices |
| Config | `pyyaml` | ≥6.0 | YAML experiment configs |
| Tracking | `tensorboard` | ≥2.18.0 | Loss/accuracy logging |
| Dev | `ruff`, `pytest`, `pytest-cov` | latest | Linting, testing |

---

### 3. Module Descriptions

#### `src/bci/data/`
| File | Lines | Description |
|---|---|---|
| `download.py` | 234 | MOABB-based downloader for BCI IV-2a (`BNCI2014_001`) and PhysioNet (`PhysionetMI`). CLI: `uv run bci-download` |
| `preprocessing.py` | 315 | Full MNE pipeline: channel selection → bandpass (4–40 Hz) → notch (50/60 Hz) → FastICA → resample (128 Hz) → z-score normalise |
| `transforms.py` | 313 | CWT Morlet spectrogram generator. Modes: `rgb_c3_cz_c4` (C3→R, Cz→G, C4→B), `mosaic`, `single`. Output: uint8 224×224 images |
| `dataset.py` | 136 | PyTorch `Dataset` classes: `SpectrogramDataset`, `EEGFeatureDataset`, `DualBranchDataset` |

#### `src/bci/features/`
| File | Lines | Description |
|---|---|---|
| `csp.py` | 139 | `CSPFeatureExtractor`: wraps MNE CSP, sklearn-compatible (`fit`/`transform`/`fit_transform`). Ledoit-Wolf regularisation by default |
| `riemannian.py` | 142 | `RiemannianFeatureExtractor`: pyriemann pipeline (covariance estimation → Riemannian tangent space projection). Output: `n_ch*(n_ch+1)/2` features |

#### `src/bci/models/`
| File | Lines | Description |
|---|---|---|
| `vit_branch.py` | 131 | `ViTBranch`: ViT-Tiny (5.72M params, 192-dim output). Pretrained ImageNet weights via `timm`. Selective layer freezing for transfer learning |
| `math_branch.py` | 80 | `MathBranch`: MLP(input→256→128) with BatchNorm + Dropout. Processes concatenated CSP + Riemannian features |
| `fusion.py` | 192 | Three fusion strategies: `ConcatFusion`, `AttentionFusion` (default), `GatedFusion`. Factory: `create_fusion()` |
| `dual_branch.py` | 162 | `DualBranchModel`: full pipeline. **5.65M total parameters** |

#### `src/bci/utils/`
| File | Lines | Description |
|---|---|---|
| `config.py` | 220 | Dataclass-based config system (`ExperimentConfig` → `DatasetConfig`, `PreprocessingConfig`, `SpectrogramConfig`, `ModelConfig`, `TrainingConfig`). YAML loading with recursive merge |
| `seed.py` | 73 | `set_seed(42)` for full reproducibility; `get_device("auto")` for CPU/CUDA/MPS selection |
| `visualization.py` | 249 | `plot_spectrogram()`, `plot_spectrogram_rgb()`, `plot_confusion_matrix()`, `plot_training_curves()`, `plot_subject_accuracies()` |

---

### 4. Smoke Test Results

```
[PASS] Config              — YAML loading, dataclass construction
[PASS] Preprocessing       — MNE filtering (4-40 Hz), notch (50 Hz), resample (256→128 Hz)
[PASS] CWT Transform       — Morlet CWT, 32 scales, 5 trials → (5, 64, 64, 3) uint8 images
[PASS] CSP Features        — Ledoit-Wolf CSP, (40, 8, 256) → (40, 3) features
[PASS] Riemannian Features — SCM covariance + Riemann tangent space, → (40, 36) features
[PASS] ViT Branch          — vit_tiny_patch16_224, (2, 3, 224, 224) → (2, 192)
[PASS] Math Branch         — MLP, (4, 48) → (4, 128)
[PASS] Dual Branch Model   — Full pipeline, 5.65M params, (2, 2, 2) → logits (2, 2)
[PASS] PyTorch Datasets    — SpectrogramDataset, EEGFeatureDataset, DualBranchDataset
[PASS] Device Utils        — set_seed(42), get_device("auto") → cpu
```

**Result: 10/10 PASS**

---

### 5. Configuration System

The YAML + dataclass system allows full experiment reproducibility:

```python
# Load default config
config = load_config()

# Load custom config with overrides
config = load_config("configs/default.yaml", overrides={
    "training": {"learning_rate": 5e-5, "batch_size": 16},
    "model": {"fusion_method": "gated"},
})
```

---

### 6. CLI Entry Points

```bash
uv run bci-download --dataset bci_iv2a    # Download BCI IV-2a via MOABB
uv run bci-download --dataset physionet   # Download PhysioNet MMIDB
uv run bci-download --dataset all         # Download both
uv run bci-preprocess --config configs/default.yaml
uv run bci-train --config configs/default.yaml
uv run python scripts/smoke_test.py       # Integration test
```

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| `timm` for ViT | Best pretrained model zoo, simple API, ViT-Tiny is Colab-friendly at 5.7M params |
| `moabb` for data | Standardises download + epoching across all MI-EEG datasets; avoids manual GDF/BDF parsing |
| Dataclass configs over Hydra | Simpler dependency, fully typed, no CLI overhead; easy to extend |
| `rgb_c3_cz_c4` default | C3/Cz/C4 are the canonical motor cortex channels; RGB maps give the ViT 3-channel input without modifying the pretrained stem |
| Ledoit-Wolf CSP regularisation | Prevents ill-conditioned covariance with small trial counts (key for reduced-data scenario) |
| Sklearn-compatible feature extractors | Allows drop-in use with `sklearn.pipeline.Pipeline` for baselines |

---

## Phase 1 Preview

With the foundation in place, Phase 1 will implement and evaluate three independent baselines
before combining them into the dual-branch model:

1. **Baseline A** — CSP + LDA (classical, fast, interpretable)
2. **Baseline B** — Riemannian + LDA (state-of-the-art classical)
3. **Baseline C** — CWT Spectrogram + ViT-Tiny (deep learning branch alone)

Evaluation: within-subject 5-fold CV and cross-subject LOSO on BCI IV-2a.
