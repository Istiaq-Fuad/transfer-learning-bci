"""Dataset download, loading, and .npz caching via MOABB.

MOABB (Mother of All BCI Benchmarks) provides standardized access to
public MI-EEG datasets with consistent APIs.

Supported datasets:
    - BCI Competition IV-2a (BNCI2014_001): 9 subjects, 22 channels, 4 classes
    - PhysioNet MMIDB (PhysionetMI): 109 subjects, 64 channels

Cache layout under ``data/processed/<dataset_name>/``::

    subject_01.npz                # X=(n,C,T) float32, y=(n,) int, channel_names, sfreq
    subject_01_spectrograms.npz   # images=(n,9,224,224) float32, y=(n,) int
    ...
    spectrogram_stats.npz         # mean=(9,), std=(9,) computed from training split

Loader API (all functions are idempotent / cache-aware):
    process_and_cache()               — download + epoch + save .npz for all subjects
    load_subject()                    — load one subject's epochs from .npz
    load_all_subjects()               — load all subjects' epochs from .npz
    process_and_cache_spectrograms()  — generate 9-ch spectrograms and cache
    compute_spectrogram_stats()       — compute per-channel mean/std from training set
    load_subject_spectrograms()       — load spectrogram cache for one subject
    load_spectrogram_stats()          — load saved mean/std
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import mne
import numpy as np
from moabb.datasets import BNCI2014_001, PhysionetMI
from moabb.paradigms import LeftRightImagery

from bci.utils.config import DATA_DIR, DatasetConfig, SpectrogramConfig

logger = logging.getLogger(__name__)


def _get_moabb_dataset(name: str) -> Any:
    """Return the MOABB dataset object by name."""
    datasets = {
        "bci_iv2a": BNCI2014_001,
        "physionet": PhysionetMI,
    }
    if name not in datasets:
        raise ValueError(f"Unknown dataset: {name}. Choose from {list(datasets.keys())}")
    return datasets[name]()


def download_bci_iv2a(data_dir: str | Path | None = None) -> None:
    """Download BCI Competition IV-2a dataset via MOABB.

    Args:
        data_dir: Optional directory for MOABB cache. Uses MOABB default if None.
    """
    if data_dir:
        mne.set_config("MNE_DATA", str(data_dir))
    dataset = BNCI2014_001()
    dataset.download()
    logger.info("BCI IV-2a dataset downloaded successfully.")


def download_physionet(data_dir: str | Path | None = None) -> None:
    """Download PhysioNet Motor Movement/Imagery dataset via MOABB.

    Args:
        data_dir: Optional directory for MOABB cache. Uses MOABB default if None.
    """
    if data_dir:
        mne.set_config("MNE_DATA", str(data_dir))
    dataset = PhysionetMI()
    dataset.download()
    logger.info("PhysioNet MI dataset downloaded successfully.")


def load_dataset_raw(
    config: DatasetConfig,
) -> dict[int, dict[str, mne.io.BaseRaw]]:
    """Load raw EEG data for all subjects using MOABB.

    Args:
        config: Dataset configuration.

    Returns:
        Dictionary mapping subject_id -> session_name -> Raw object.
        Example: {1: {"0train": Raw, "1test": Raw}, 2: {...}, ...}
    """
    dataset = _get_moabb_dataset(config.name)

    subjects = config.subjects
    if subjects is None:
        subjects = dataset.subject_list

    raw_data: dict[int, dict[str, mne.io.BaseRaw]] = {}
    for subject in subjects:
        logger.info("Loading subject %d from %s...", subject, config.name)
        try:
            sessions = dataset.get_data(subjects=[subject])
            raw_data[subject] = sessions[subject]
        except Exception:
            logger.exception("Failed to load subject %d", subject)
            continue

    logger.info("Loaded %d subjects from %s", len(raw_data), config.name)
    return raw_data


def load_dataset_epochs(
    config: DatasetConfig,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Load epoched data using MOABB's LeftRightImagery paradigm.

    This is a convenience function that handles downloading, channel selection,
    and epoching in one step using MOABB's built-in paradigm.

    Args:
        config: Dataset configuration.

    Returns:
        Dictionary mapping subject_id -> (X, y) where:
            X: ndarray of shape (n_trials, n_channels, n_times)
            y: ndarray of shape (n_trials,) with string labels
    """
    dataset = _get_moabb_dataset(config.name)
    paradigm = LeftRightImagery()

    subjects = config.subjects
    if subjects is None:
        subjects = dataset.subject_list

    subject_data: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for subject in subjects:
        logger.info("Loading epochs for subject %d...", subject)
        try:
            X, y, _metadata = paradigm.get_data(dataset=dataset, subjects=[subject])
            subject_data[subject] = (X, y)
            logger.info(
                "  Subject %d: X=%s, classes=%s",
                subject,
                X.shape,
                np.unique(y),
            )
        except Exception:
            logger.exception("Failed to load subject %d", subject)
            continue

    return subject_data


def get_common_channels(
    dataset_name: str,
    requested_channels: list[str],
) -> list[str]:
    """Get the intersection of requested channels and what the dataset provides.

    Different datasets may use slightly different channel naming conventions.
    This function handles the mapping.

    Args:
        dataset_name: Name of the dataset.
        requested_channels: List of desired channel names.

    Returns:
        List of channel names available in both the dataset and the request.
    """
    # Known channel name mappings between datasets
    # BCI IV-2a uses standard 10-20 names
    # PhysioNet uses "C3.", "C4." etc. (with dots) — MOABB normalizes these
    dataset = _get_moabb_dataset(dataset_name)
    sample_subject = dataset.subject_list[0]

    # Get a sample raw to check channel names
    sessions = dataset.get_data(subjects=[sample_subject])
    first_session = next(iter(sessions[sample_subject].values()))

    if isinstance(first_session, dict):
        # Some datasets nest runs inside sessions
        first_run = next(iter(first_session.values()))
        available_channels = first_run.ch_names
    else:
        available_channels = first_session.ch_names

    # Find intersection
    common = [ch for ch in requested_channels if ch in available_channels]

    if len(common) < len(requested_channels):
        missing = set(requested_channels) - set(common)
        logger.warning(
            "Channels not found in %s: %s. Available: %s",
            dataset_name,
            missing,
            available_channels,
        )

    return common


# ---------------------------------------------------------------------------
# Cache path helpers
# ---------------------------------------------------------------------------

#: Default processed data root
_PROCESSED_DIR = DATA_DIR / "processed"


def _processed_dir(dataset_name: str, data_dir: Path | None = None) -> Path:
    """Return the processed cache directory for a dataset."""
    base = Path(data_dir) if data_dir is not None else _PROCESSED_DIR
    return base / dataset_name


def _epoch_cache_path(dataset_name: str, subject_id: int, data_dir: Path | None = None) -> Path:
    return _processed_dir(dataset_name, data_dir) / f"subject_{subject_id:02d}.npz"


def _spectrogram_cache_path(
    dataset_name: str, subject_id: int, data_dir: Path | None = None
) -> Path:
    return _processed_dir(dataset_name, data_dir) / f"subject_{subject_id:02d}_spectrograms.npz"


def _stats_cache_path(dataset_name: str, data_dir: Path | None = None) -> Path:
    return _processed_dir(dataset_name, data_dir) / "spectrogram_stats.npz"


# ---------------------------------------------------------------------------
# Epoch label encoding
# ---------------------------------------------------------------------------

_LABEL_MAP: dict[str, int] = {"left_hand": 0, "right_hand": 1}


def _encode_labels(y: np.ndarray) -> np.ndarray:
    """Convert string labels (e.g. 'left_hand') to int32 (0/1).

    The default value in dict.get() is evaluated eagerly, so we must not use
    ``int(lbl)`` as the fallback — it would crash on string labels even when
    the key is present.  Use an explicit lookup instead.
    """
    result = []
    for lbl in y:
        key = str(lbl)
        if key in _LABEL_MAP:
            result.append(_LABEL_MAP[key])
        else:
            try:
                result.append(int(lbl))
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    f"Unknown label {lbl!r}. Add it to _LABEL_MAP or extend the paradigm filter."
                ) from exc
    return np.array(result, dtype=np.int32)


# ---------------------------------------------------------------------------
# Phase B public API
# ---------------------------------------------------------------------------


def process_and_cache(
    dataset_name: str,
    data_dir: Path | None = None,
    fmin: float = 8.0,
    fmax: float = 32.0,
    force: bool = False,
) -> None:
    """Download via MOABB and save per-subject epoch .npz files.

    Files are written to ``<data_dir>/<dataset_name>/subject_NN.npz`` with
    keys ``X`` (float32, n_trials × n_channels × n_times), ``y`` (int32),
    ``channel_names`` (list[str]), and ``sfreq`` (float scalar).

    This function is idempotent: existing files are skipped unless
    ``force=True``.

    Args:
        dataset_name: ``"bci_iv2a"`` or ``"physionet"``.
        data_dir:     Root directory for processed cache (default: ``data/processed``).
        fmin:         Low-cut frequency passed to the MOABB paradigm.
        fmax:         High-cut frequency passed to the MOABB paradigm.
        force:        Overwrite existing cache files.
    """
    out_dir = _processed_dir(dataset_name, data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = _get_moabb_dataset(dataset_name)
    paradigm = LeftRightImagery(fmin=fmin, fmax=fmax)

    subjects = dataset.subject_list
    logger.info(
        "process_and_cache: %s — %d subjects, %.1f–%.1f Hz",
        dataset_name,
        len(subjects),
        fmin,
        fmax,
    )

    for subject in subjects:
        cache_path = _epoch_cache_path(dataset_name, subject, data_dir)
        if cache_path.exists() and not force:
            logger.info("  Subject %d: cache exists, skipping", subject)
            continue

        logger.info("  Subject %d: loading from MOABB...", subject)
        try:
            X, y, metadata = paradigm.get_data(dataset=dataset, subjects=[subject])
        except Exception:
            logger.exception("  Subject %d: failed to load, skipping", subject)
            continue

        X_arr = np.asarray(X, dtype=np.float32)
        y_int = _encode_labels(np.asarray(y))

        # Recover channel names from metadata or dataset
        # MOABB paradigm always returns numpy arrays without ch_names;
        # fetch them from the raw object via the dataset directly.
        try:
            sessions = dataset.get_data(subjects=[subject])
            first_session = next(iter(sessions[subject].values()))
            if isinstance(first_session, dict):
                first_run = next(iter(first_session.values()))
                ch_names = first_run.pick_types(eeg=True).ch_names
            else:
                ch_names = first_session.pick_types(eeg=True).ch_names
        except Exception:
            logger.warning("  Subject %d: could not read ch_names, storing empty list", subject)
            ch_names = []

        sfreq = float(paradigm.resample or 128.0)

        np.savez_compressed(
            cache_path,
            X=X_arr,
            y=y_int,
            channel_names=np.array(ch_names, dtype=object),
            sfreq=np.float32(sfreq),
        )
        logger.info(
            "  Subject %d: saved %s — X=%s, classes=%s",
            subject,
            cache_path.name,
            X_arr.shape,
            np.unique(y_int),
        )

    logger.info("process_and_cache: done for %s", dataset_name)


def load_subject(
    dataset_name: str,
    subject_id: int,
    data_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], float]:
    """Load one subject's epochs from the .npz cache.

    Args:
        dataset_name: ``"bci_iv2a"`` or ``"physionet"``.
        subject_id:   Subject number (1-based).
        data_dir:     Root directory for processed cache.

    Returns:
        Tuple ``(X, y, channel_names, sfreq)`` where
        ``X`` has shape ``(n_trials, n_channels, n_times)`` float32 and
        ``y`` has shape ``(n_trials,)`` int32.

    Raises:
        FileNotFoundError: If the cache file does not exist.
    """
    cache_path = _epoch_cache_path(dataset_name, subject_id, data_dir)
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache not found: {cache_path}. Run process_and_cache() first.")
    data = np.load(cache_path, allow_pickle=True)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int32)
    channel_names = list(data["channel_names"])
    sfreq = float(data["sfreq"])
    return X, y, channel_names, sfreq


def load_all_subjects(
    dataset_name: str,
    data_dir: Path | None = None,
) -> tuple[dict[int, tuple[np.ndarray, np.ndarray]], list[str], float]:
    """Load all cached subjects for a dataset.

    Args:
        dataset_name: ``"bci_iv2a"`` or ``"physionet"``.
        data_dir:     Root directory for processed cache.

    Returns:
        Tuple ``(subject_data, channel_names, sfreq)`` where
        ``subject_data`` maps ``subject_id -> (X, y)``.

    Raises:
        FileNotFoundError: If no cache files are found.
    """
    processed = _processed_dir(dataset_name, data_dir)
    cache_files = sorted(processed.glob("subject_[0-9]*.npz"))
    # Exclude spectrogram files
    cache_files = [p for p in cache_files if "_spectrograms" not in p.name]

    if not cache_files:
        raise FileNotFoundError(
            f"No epoch cache found in {processed}. Run process_and_cache() first."
        )

    subject_data: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    channel_names: list[str] = []
    sfreq: float = 128.0

    for path in cache_files:
        # Parse subject id from filename "subject_NN.npz"
        stem = path.stem  # e.g. "subject_01"
        subject_id = int(stem.split("_")[1])
        data = np.load(path, allow_pickle=True)
        X = data["X"].astype(np.float32)
        y = data["y"].astype(np.int32)
        subject_data[subject_id] = (X, y)
        if not channel_names:
            channel_names = list(data["channel_names"])
            sfreq = float(data["sfreq"])

    logger.info("load_all_subjects: loaded %d subjects from %s", len(subject_data), dataset_name)
    return subject_data, channel_names, sfreq


def process_and_cache_spectrograms(
    dataset_name: str,
    data_dir: Path | None = None,
    spec_config: SpectrogramConfig | None = None,
    force: bool = False,
) -> None:
    """Generate 9-channel multichannel spectrograms and write per-subject .npz.

    Reads epoch caches written by :func:`process_and_cache` and generates
    CWT spectrograms with shape ``(n_trials, 9, 224, 224)`` float32.

    Files are written to ``subject_NN_spectrograms.npz`` with keys
    ``images`` (float32) and ``y`` (int32).

    This function is idempotent: existing files are skipped unless
    ``force=True``.

    Args:
        dataset_name: ``"bci_iv2a"`` or ``"physionet"``.
        data_dir:     Root directory for processed cache.
        spec_config:  Spectrogram configuration (default: :class:`SpectrogramConfig`).
        force:        Overwrite existing cache files.
    """
    from bci.data.transforms import CWTSpectrogramTransform

    cfg = spec_config or SpectrogramConfig()
    transform = CWTSpectrogramTransform(cfg)

    processed = _processed_dir(dataset_name, data_dir)
    epoch_files = sorted(processed.glob("subject_[0-9]*.npz"))
    epoch_files = [p for p in epoch_files if "_spectrograms" not in p.name]

    if not epoch_files:
        raise FileNotFoundError(f"No epoch cache in {processed}. Run process_and_cache() first.")

    logger.info("process_and_cache_spectrograms: %s — %d subjects", dataset_name, len(epoch_files))

    for epoch_path in epoch_files:
        stem = epoch_path.stem
        subject_id = int(stem.split("_")[1])
        spec_path = _spectrogram_cache_path(dataset_name, subject_id, data_dir)

        if spec_path.exists() and not force:
            logger.info("  Subject %d: spectrogram cache exists, skipping", subject_id)
            continue

        data = np.load(epoch_path, allow_pickle=True)
        X = data["X"].astype(np.float32)
        y = data["y"].astype(np.int32)
        channel_names = list(data["channel_names"])
        sfreq = float(data["sfreq"])

        logger.info("  Subject %d: generating spectrograms for %d trials...", subject_id, len(X))
        images = transform.transform_epochs(X, channel_names, sfreq)  # (n, 9, H, W) float32

        np.savez_compressed(spec_path, images=images, y=y)
        logger.info(
            "  Subject %d: saved spectrograms %s → %s", subject_id, images.shape, spec_path.name
        )

    logger.info("process_and_cache_spectrograms: done for %s", dataset_name)


def compute_spectrogram_stats(
    dataset_name: str,
    data_dir: Path | None = None,
    train_subject_ids: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-channel mean and std from training subjects' spectrogram cache.

    Results are saved to ``spectrogram_stats.npz`` (keys ``mean``, ``std``)
    and also returned as numpy arrays of shape ``(n_channels,)``.

    Args:
        dataset_name:      ``"bci_iv2a"`` or ``"physionet"``.
        data_dir:          Root directory for processed cache.
        train_subject_ids: Subject IDs to include in stats computation.
                           Defaults to all cached subjects.

    Returns:
        Tuple ``(mean, std)`` each of shape ``(n_channels,)`` float32.
    """
    processed = _processed_dir(dataset_name, data_dir)
    spec_files = sorted(processed.glob("subject_[0-9]*_spectrograms.npz"))

    if train_subject_ids is not None:
        id_set = set(train_subject_ids)
        spec_files = [p for p in spec_files if int(p.stem.split("_")[1]) in id_set]

    if not spec_files:
        raise FileNotFoundError(
            f"No spectrogram cache in {processed}. Run process_and_cache_spectrograms() first."
        )

    logger.info(
        "compute_spectrogram_stats: accumulating over %d files for %s",
        len(spec_files),
        dataset_name,
    )

    # Two-pass: first compute mean, then compute std
    # Pass 1: sum and count
    n_channels: int | None = None
    running_sum: np.ndarray | None = None
    n_total = 0

    for path in spec_files:
        images = np.load(path)["images"]  # (n, C, H, W) float32
        n, C, H, W = images.shape
        if n_channels is None:
            n_channels = C
            running_sum = np.zeros(C, dtype=np.float64)
        assert images.shape[1] == n_channels, "Channel count mismatch across subjects"
        # Mean over (n, H, W) for each channel
        running_sum += images.mean(axis=(0, 2, 3)).astype(np.float64) * n  # type: ignore[operator]
        n_total += n

    mean = (running_sum / n_total).astype(np.float32)  # type: ignore[operator]

    # Pass 2: variance
    running_sq_sum = np.zeros(n_channels, dtype=np.float64)  # type: ignore[arg-type]
    for path in spec_files:
        images = np.load(path)["images"]  # (n, C, H, W)
        n = images.shape[0]
        diff = images - mean[None, :, None, None]
        running_sq_sum += (diff**2).mean(axis=(0, 2, 3)).astype(np.float64) * n

    std = np.sqrt(running_sq_sum / n_total).astype(np.float32)
    std = np.maximum(std, 1e-6)  # avoid division by zero

    stats_path = _stats_cache_path(dataset_name, data_dir)
    np.savez_compressed(stats_path, mean=mean, std=std)
    logger.info(
        "compute_spectrogram_stats: saved → %s (mean=%s, std=%s)", stats_path.name, mean, std
    )
    return mean, std


def load_subject_spectrograms(
    dataset_name: str,
    subject_id: int,
    data_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load spectrogram cache for one subject.

    Args:
        dataset_name: ``"bci_iv2a"`` or ``"physionet"``.
        subject_id:   Subject number (1-based).
        data_dir:     Root directory for processed cache.

    Returns:
        Tuple ``(images, y)`` where ``images`` has shape
        ``(n_trials, n_channels, H, W)`` float32 and ``y`` is int32.

    Raises:
        FileNotFoundError: If the cache file does not exist.
    """
    cache_path = _spectrogram_cache_path(dataset_name, subject_id, data_dir)
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Spectrogram cache not found: {cache_path}. "
            "Run process_and_cache_spectrograms() first."
        )
    data = np.load(cache_path)
    return data["images"].astype(np.float32), data["y"].astype(np.int32)


def load_spectrogram_stats(
    dataset_name: str,
    data_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load pre-computed per-channel mean and std.

    Args:
        dataset_name: ``"bci_iv2a"`` or ``"physionet"``.
        data_dir:     Root directory for processed cache.

    Returns:
        Tuple ``(mean, std)`` each of shape ``(n_channels,)`` float32.

    Raises:
        FileNotFoundError: If stats have not been computed yet.
    """
    stats_path = _stats_cache_path(dataset_name, data_dir)
    if not stats_path.exists():
        raise FileNotFoundError(
            f"Spectrogram stats not found: {stats_path}. Run compute_spectrogram_stats() first."
        )
    data = np.load(stats_path)
    return data["mean"].astype(np.float32), data["std"].astype(np.float32)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for downloading datasets."""
    parser = argparse.ArgumentParser(description="Download MI-EEG datasets")
    parser.add_argument(
        "--dataset",
        choices=["bci_iv2a", "physionet", "all"],
        default="bci_iv2a",
        help="Dataset to download",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DATA_DIR / "raw"),
        help="Directory to store downloaded data",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset in ("bci_iv2a", "all"):
        logger.info("Downloading BCI Competition IV-2a...")
        download_bci_iv2a(data_dir)

    if args.dataset in ("physionet", "all"):
        logger.info("Downloading PhysioNet Motor Imagery...")
        download_physionet(data_dir)

    logger.info("Download complete. Data stored in: %s", data_dir)


if __name__ == "__main__":
    main()
