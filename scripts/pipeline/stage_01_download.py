"""Stage 01 – Download datasets and build .npz cache.

Downloads BCI Competition IV-2a and PhysioNet MMIDB via MOABB, epochs
all subjects with an 8–32 Hz bandpass filter, and saves per-subject .npz
files.  Then generates 9-channel multichannel CWT spectrograms (224×224)
and computes per-channel mean/std statistics from the BCI IV-2a training
subjects.

All later pipeline stages load data exclusively from this cache — no further
MOABB calls are required once this stage has run.

Output layout::

    data/processed/bci_iv2a/subject_01.npz          # X, y, channel_names, sfreq
    data/processed/bci_iv2a/subject_01_spectrograms.npz
    ...
    data/processed/bci_iv2a/spectrogram_stats.npz   # mean, std per channel
    data/processed/physionet/subject_001.npz
    data/processed/physionet/subject_001_spectrograms.npz
    ...
    data/processed/physionet/spectrogram_stats.npz

Usage::

    uv run python scripts/pipeline/stage_01_download.py
    uv run python scripts/pipeline/stage_01_download.py \\
        --processed-dir data/processed --mne-data-dir ~/mne_data
    # Force regenerate even if cache exists:
    uv run python scripts/pipeline/stage_01_download.py --force
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 01: Download datasets and build .npz epoch + spectrogram cache.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--processed-dir",
        default=None,
        help="Root directory for processed .npz cache (default: data/processed/)",
    )
    p.add_argument(
        "--mne-data-dir",
        default="~/mne_data",
        help="MNE data directory for MOABB downloads",
    )
    p.add_argument(
        "--run-dir",
        default=None,
        help="Optional run directory for log file",
    )
    p.add_argument(
        "--skip-physionet",
        action="store_true",
        help="Skip PhysioNet download (faster for testing)",
    )
    p.add_argument(
        "--skip-spectrograms",
        action="store_true",
        help="Skip spectrogram generation (only save epoch .npz files)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Regenerate existing cache files",
    )
    return p.parse_args()


def setup_logging(run_dir: Path | None) -> logging.Logger:
    fmt = "%(asctime)s  %(levelname)-8s  %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt, datefmt="%H:%M:%S", stream=sys.stdout)
    for lib in ("mne", "pyriemann", "timm", "matplotlib", "moabb"):
        logging.getLogger(lib).setLevel(logging.ERROR)
    log = logging.getLogger("stage_01")
    if run_dir:
        run_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(run_dir / "stage_01_download.log")
        fh.setFormatter(logging.Formatter(fmt, "%H:%M:%S"))
        log.addHandler(fh)
    return log


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir) if args.run_dir else None
    log = setup_logging(run_dir)

    import mne

    mne.set_log_level("ERROR")
    if args.mne_data_dir:
        mne.set_config("MNE_DATA", str(Path(args.mne_data_dir).expanduser()))

    processed_dir = Path(args.processed_dir) if args.processed_dir else None

    from bci.data.download import (
        compute_spectrogram_stats,
        process_and_cache,
        process_and_cache_spectrograms,
    )
    from bci.utils.config import SpectrogramConfig

    spec_cfg = SpectrogramConfig()  # multichannel, 8–32 Hz, 9 channels, 224×224

    # ── BCI Competition IV-2a ─────────────────────────────────────────────
    log.info("=== BCI Competition IV-2a (9 subjects) ===")
    try:
        process_and_cache(
            "bci_iv2a",
            data_dir=processed_dir,
            fmin=8.0,
            fmax=32.0,
            force=args.force,
        )
    except Exception:
        log.exception("BCI IV-2a epoch caching failed")
        sys.exit(1)

    if not args.skip_spectrograms:
        log.info("Generating BCI IV-2a spectrograms...")
        try:
            process_and_cache_spectrograms(
                "bci_iv2a",
                data_dir=processed_dir,
                spec_config=spec_cfg,
                force=args.force,
            )
            log.info("Computing BCI IV-2a spectrogram stats...")
            mean, std = compute_spectrogram_stats("bci_iv2a", data_dir=processed_dir)
            log.info("BCI IV-2a stats: mean=%s  std=%s", mean.tolist(), std.tolist())
        except Exception:
            log.exception("BCI IV-2a spectrogram generation failed")
            sys.exit(1)

    # ── PhysioNet MMIDB ───────────────────────────────────────────────────
    if args.skip_physionet:
        log.info("Skipping PhysioNet (--skip-physionet)")
    else:
        log.info("=== PhysioNet MMIDB (up to 109 subjects) ===")
        try:
            process_and_cache(
                "physionet",
                data_dir=processed_dir,
                fmin=8.0,
                fmax=32.0,
                force=args.force,
            )
        except Exception:
            log.exception("PhysioNet epoch caching failed")
            sys.exit(1)

        if not args.skip_spectrograms:
            log.info("Generating PhysioNet spectrograms...")
            try:
                process_and_cache_spectrograms(
                    "physionet",
                    data_dir=processed_dir,
                    spec_config=spec_cfg,
                    force=args.force,
                )
                log.info("Computing PhysioNet spectrogram stats...")
                mean_p, std_p = compute_spectrogram_stats("physionet", data_dir=processed_dir)
                log.info("PhysioNet stats: mean=%s  std=%s", mean_p.tolist(), std_p.tolist())
            except Exception:
                log.exception("PhysioNet spectrogram generation failed")
                sys.exit(1)

    log.info("Stage 01 complete.")


if __name__ == "__main__":
    main()
