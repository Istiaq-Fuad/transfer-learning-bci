"""Shared logging setup for pipeline stage scripts."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

__all__ = ["setup_stage_logging"]

_NOISY_LIBS = ("mne", "pyriemann", "timm", "matplotlib", "moabb")


def setup_stage_logging(
    run_dir: Path,
    stage_name: str,
    log_filename: str | None = None,
) -> logging.Logger:
    """Configure root logging and return a named stage logger.

    Sets up a stdout handler (INFO level) plus a file handler in
    ``run_dir / log_filename`` (or ``run_dir / "<stage_name>.log"`` if
    *log_filename* is omitted).  Silences chatty third-party libraries.

    Args:
        run_dir: Directory where the log file will be written.
            Created automatically if it does not exist.
        stage_name: Logger name (e.g. ``"stage_05"``).
        log_filename: Override the log filename.  Defaults to
            ``"<stage_name>.log"``.

    Returns:
        A configured :class:`logging.Logger` instance.
    """
    fmt = "%(asctime)s  %(levelname)-8s  %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt, datefmt="%H:%M:%S", stream=sys.stdout)

    for lib in _NOISY_LIBS:
        logging.getLogger(lib).setLevel(logging.ERROR)

    log = logging.getLogger(stage_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    fname = log_filename or f"{stage_name}.log"
    fh = logging.FileHandler(run_dir / fname)
    fh.setFormatter(logging.Formatter(fmt, "%H:%M:%S"))
    log.addHandler(fh)

    return log
