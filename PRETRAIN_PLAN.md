# Multi-Dataset Pretraining Plan

## Background

All 9 CSP/Riemannian improvements from `csp_riemannian.md` are complete (87/87 tests passing).
This plan covers the next phase: fixing the channel selection bug in pretrain scripts and adding
multi-dataset pretraining support.

---

## Problem Summary

### Channel Selection Bug (Critical, Pre-existing)

Both `pretrain_physionet.py` and `stage_07_pretrain.py` hardcode
`channel_names=["C3", "Cz", "C4"]` and pass the full MOABB data array
(e.g. 64 channels for PhysioNet) to `CWTSpectrogramTransform`.

`transform_trial_rgb()` does `channel_names.index("C3")` → returns 0,
but actual C3 is at index 8 in PhysioNet's 64-channel layout.
This means the CWT is computed on the wrong EEG channels.

PhysioNet channel order (indices 0–12):
`FC5, FC3, FC1, FCz, FC2, FC4, FC6, C5, C3, C1, Cz, C2, C4, ...`

Fix: load full MOABB data, get real channel names from metadata, extract
only the needed channels by name before passing to CWT transform.

### `stage_07_pretrain.py` Inconsistencies
- Still uses 4–40 Hz (should be 8–32 Hz)
- Saves `model.state_dict()` (full ViTBranch), but `pretrain_physionet.py` and
  `stage_08_finetune.py` expect `model.backbone.state_dict()` — checkpoint format mismatch
- Uses `label_smoothing=0.1` vs `0.05` in `pretrain_physionet.py`

---

## Dataset Survey

Target pretrain datasets (all LeftRightImagery via MOABB):

| Dataset | Subjects | Channels | Has all 9 motor channels? |
|---|---|---|---|
| PhysionetMI | 109 | 64 | Yes |
| Stieger2021 | 62 | 64 | Yes (10-10 montage) |
| Lee2019_MI | 54 | 62 | Yes (likely) |
| Cho2017 | 52 | 64 | Yes (10-10 montage) |
| BNCI2014_004 | 9 | 3 | No — only C3, Cz, C4 (zero-fill missing 6) |

The 9 target motor cortex channels (from `SpectrogramConfig.spectrogram_channels`):
`C3, C1, Cz, C2, C4, FC3, FC4, CP3, CP4`

`transform_trial_multichannel()` already handles missing channels gracefully
(logs a warning and uses zeros), so BNCI2014_004 works without special-casing.

BNCI2014_001 (the target dataset) must NOT be used for pretraining.

---

## Implementation Steps

### Step 1 — Shared multi-dataset loader (`src/bci/data/download.py`)

Add two functions:

```python
def load_moabb_mi_dataset(
    dataset_name: str,
    subjects: list[int] | None = None,
    fmin: float = 8.0,
    fmax: float = 32.0,
    resample: float = 128.0,
    target_channels: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load a named MOABB motor imagery dataset via LeftRightImagery paradigm.

    Returns (X, y, channel_names) where:
      - X shape: (n_trials, n_target_channels, n_times)  [channels extracted by name]
      - y: integer array, left_hand=0, right_hand=1
      - channel_names: list of channel names in X (the extracted subset)
    """
```

```python
PRETRAIN_DATASETS = ["PhysionetMI", "Stieger2021", "Lee2019_MI", "Cho2017", "BNCI2014_004"]

def load_multi_dataset_pretrain(
    dataset_names: list[str] = PRETRAIN_DATASETS,
    target_channels: list[str] | None = None,  # defaults to SpectrogramConfig().spectrogram_channels
    fmin: float = 8.0,
    fmax: float = 32.0,
    resample: float = 128.0,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load and concatenate multiple MOABB MI datasets for pretraining.

    Returns (X, y, channel_names) with X shape (total_trials, 9, n_times).
    Datasets that lack some target channels will have zeros in those positions.
    """
```

Dataset name → MOABB class mapping:
```python
DATASET_REGISTRY = {
    "PhysionetMI":   moabb.datasets.PhysionetMI,
    "Stieger2021":   moabb.datasets.Stieger2021,
    "Lee2019_MI":    moabb.datasets.Lee2019_MI,
    "Cho2017":       moabb.datasets.Cho2017,
    "BNCI2014_004":  moabb.datasets.BNCI2014_004,
}
```

Channel extraction logic:
1. Get full `(X_full, y, metadata)` from MOABB paradigm
2. Get real channel names from the paradigm (via `paradigm.get_data(dataset)` or
   from the returned `metadata` DataFrame which contains a `channels` column, or
   by loading one raw object from `dataset.get_data()`)
3. For each target channel: find its index in the real channel list; if missing,
   fill with zeros
4. Stack selected channel arrays → output X

### Step 2 — Fix `scripts/pretrain_physionet.py`

- Replace inline MOABB loading with `load_multi_dataset_pretrain()`
- Add `--datasets` CLI flag (default: all 5, space-separated)
- Switch `channel_mode` from `"rgb_c3_cz_c4"` to `"multichannel"` (9 channels)
- Remove hardcoded `channel_names=["C3", "Cz", "C4"]` — real names come from loader
- Keep 8–32 Hz (already fixed)

### Step 3 — Fix `scripts/pipeline/stage_07_pretrain.py`

- Line 192: change `fmin=4.0, fmax=40.0` → `fmin=8.0, fmax=32.0`
- Line 222–225: change `SpectrogramConfig(freq_min=4.0, freq_max=40.0)` → `freq_min=8.0, freq_max=32.0`
- Line 274: change `label_smoothing=0.1` → `label_smoothing=0.05`
- Line 303: change `torch.save(model.state_dict(), ...)` → `torch.save(model.backbone.state_dict(), ...)`
- Replace inline MOABB loading with `load_multi_dataset_pretrain()`
- Add `--datasets` CLI flag (default: all 5)
- Switch to `channel_mode="multichannel"`

### Step 4 — Update `scripts/pipeline/stage_01_download.py`

Add download + verification for:
- Stieger2021
- Lee2019_MI
- Cho2017
- BNCI2014_004

Pattern: same as existing PhysioNet download block — instantiate dataset,
call `download()`, log subject count and channel count.

### Step 5 — Update `SpectrogramConfig` defaults (`src/bci/utils/config.py`)

```python
# Before
freq_min: float = 4.0
freq_max: float = 40.0

# After
freq_min: float = 8.0
freq_max: float = 32.0
```

This only affects code that uses the dataclass default without overriding.
All existing scripts that already pass explicit values are unaffected.

### Step 6 — Add tests

In `tests/test_phase3.py` (or a new `tests/test_pretrain.py`), add:

- `test_load_moabb_mi_dataset_returns_correct_shape` — mock MOABB, verify output shape
- `test_load_moabb_mi_dataset_channel_extraction` — verify correct channels selected by name
- `test_load_moabb_mi_dataset_missing_channel_zeroed` — dataset missing a channel → zeros
- `test_load_multi_dataset_pretrain_concatenates` — multiple datasets → combined array
- `test_load_multi_dataset_pretrain_label_encoding` — labels are 0/1 integers
- `test_pretrain_datasets_constant_excludes_target` — PRETRAIN_DATASETS does not contain BNCI2014_001

### Step 7 — Run full test suite

```bash
uv run pytest tests/ -v
```

All 87+ existing tests must continue to pass alongside the new tests.

---

## Files to Edit

| File | Change |
|---|---|
| `src/bci/data/download.py` | Add `PRETRAIN_DATASETS`, `DATASET_REGISTRY`, `load_moabb_mi_dataset()`, `load_multi_dataset_pretrain()` |
| `scripts/pretrain_physionet.py` | Use shared loader, `--datasets` flag, `channel_mode="multichannel"` |
| `scripts/pipeline/stage_07_pretrain.py` | 8–32 Hz, checkpoint format fix, shared loader, `--datasets` flag, label_smoothing |
| `scripts/pipeline/stage_01_download.py` | Add Stieger2021, Lee2019_MI, Cho2017, BNCI2014_004 download |
| `src/bci/utils/config.py` | `SpectrogramConfig.freq_min/freq_max` defaults: 4/40 → 8/32 |
| `tests/test_phase3.py` (or new file) | 6 new tests for shared loader |

## Files NOT to Edit

- `src/bci/data/transforms.py` — `transform_trial_multichannel()` already handles missing channels correctly
- `scripts/pipeline/stage_08_finetune.py` — downstream, already expects `backbone.state_dict()` format
- `src/bci/features/csp.py`, `riemannian.py` — already updated in previous phase
- Any test file other than `test_phase3.py` — should not require changes
