"""Dataset download and loading via MOABB.

MOABB (Mother of All BCI Benchmarks) provides standardized access to
public MI-EEG datasets with consistent APIs.

Supported datasets:
    - BCI Competition IV-2a (BNCI2014_001): 9 subjects, 22 channels, 4 classes
    - PhysioNet MMIDB (PhysionetMI): 109 subjects, 64 channels
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

from bci.utils.config import DATA_DIR, DatasetConfig

logger = logging.getLogger(__name__)

# MOABB event ID mapping for left/right hand MI
EVENT_ID = {"left_hand": 1, "right_hand": 2}

# Canonical motor cortex channel names for cross-dataset alignment
MOTOR_CHANNELS = [
    "FC5", "FC3", "FC1", "FC2", "FC4", "FC6",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP5", "CP3", "CP1", "CPz", "CP2", "CP4",
]


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
            X, y, _metadata = paradigm.get_data(
                dataset=dataset, subjects=[subject]
            )
            subject_data[subject] = (X, y)
            logger.info(
                "  Subject %d: X=%s, classes=%s",
                subject, X.shape, np.unique(y),
            )
        except Exception:
            logger.exception("Failed to load subject %d", subject)
            continue

    return subject_data


def get_common_channels(
    dataset_name: str, requested_channels: list[str],
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
            dataset_name, missing, available_channels,
        )

    return common


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
