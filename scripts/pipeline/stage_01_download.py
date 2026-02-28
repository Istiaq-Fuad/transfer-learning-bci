"""Stage 1 – Download / verify datasets.

Downloads BCI Competition IV-2a and PhysioNet MMIDB via MOABB (if not already
present) and prints a summary of available subjects.  Both datasets are loaded
with the 8–32 Hz motor-imagery band and 128 Hz resampling, consistent with all
downstream pipeline stages.  All 109 PhysioNet subjects are verified so the
full corpus is cached before pretraining (Stage 7).

Usage::

    uv run python scripts/pipeline/stage_01_download.py
    uv run python scripts/pipeline/stage_01_download.py --data-dir ~/mne_data
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 1: Download / verify BCI IV-2a and PhysioNet datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir", default="~/mne_data", help="MNE data directory")
    p.add_argument(
        "--run-dir", default=None, help="Run directory for logs (optional; created if absent)"
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
    import numpy as np
    from moabb.datasets import BNCI2014_001, PhysionetMI
    from moabb.paradigms import LeftRightImagery

    mne.set_log_level("ERROR")

    # ── BCI IV-2a ─────────────────────────────────────────────────────────
    log.info("Checking BCI Competition IV-2a...  [filter: 8–32 Hz]")
    try:
        dataset = BNCI2014_001()
        paradigm = LeftRightImagery(fmin=8.0, fmax=32.0, resample=128.0)
        ok_subjects = []
        for sid in dataset.subject_list:
            try:
                X, y_labels, _ = paradigm.get_data(dataset=dataset, subjects=[sid])
                log.info("  Subject %2d: X=%s", sid, X.shape)
                ok_subjects.append(sid)
            except Exception as e:
                log.warning("  Subject %d: FAILED – %s", sid, e)
        log.info("BCI IV-2a: %d/%d subjects OK.", len(ok_subjects), len(dataset.subject_list))
    except Exception as e:
        log.error("BCI IV-2a download/load failed: %s", e)
        sys.exit(1)

    # ── PhysioNet MMIDB ───────────────────────────────────────────────────
    log.info("Checking PhysioNet MMIDB (all subjects)...  [filter: 8–32 Hz]")
    try:
        pdata = PhysionetMI()
        pparadigm = LeftRightImagery(fmin=8.0, fmax=32.0, resample=128.0)
        ok_subjects_p: list[int] = []
        for sid in pdata.subject_list:
            try:
                X, y_labels, _ = pparadigm.get_data(dataset=pdata, subjects=[sid])
                log.info("  Subject %3d: X=%s", sid, X.shape)
                ok_subjects_p.append(sid)
            except Exception as e:
                log.warning("  Subject %d: FAILED – %s", sid, e)
        log.info(
            "PhysioNet MMIDB: %d/%d subjects OK.",
            len(ok_subjects_p),
            len(pdata.subject_list),
        )
    except Exception as e:
        log.error("PhysioNet download/load failed: %s", e)
        sys.exit(1)

    log.info("Stage 1 complete – all datasets downloaded and verified.")


if __name__ == "__main__":
    main()
