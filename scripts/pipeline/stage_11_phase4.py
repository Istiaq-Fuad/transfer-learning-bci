"""Stage 11 – Phase 4: Compile results, generate figures, run statistics.

Calls the three existing phase4 scripts in order:
  1. phase4_compile_results.py  → phase4_summary.json
  2. phase4_visualize.py        → figures/
  3. phase4_stats.py            → phase4_stats.json

All three scripts must exist in the parent scripts/ directory.

Usage::

    uv run python scripts/pipeline/stage_11_phase4.py --run-dir runs/my_run
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 11: Compile results, visualize, and run stats.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--run-dir", required=True, help="Run directory containing results/")
    return p.parse_args()


def setup_logging(run_dir: Path) -> logging.Logger:
    fmt = "%(asctime)s  %(levelname)-8s  %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt, datefmt="%H:%M:%S",
                        stream=sys.stdout)
    log = logging.getLogger("stage_11")
    run_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(run_dir / "stage_11_phase4.log")
    fh.setFormatter(logging.Formatter(fmt, "%H:%M:%S"))
    log.addHandler(fh)
    return log


def run_script(script_path: Path, extra_args: list[str], log) -> None:
    log.info("Running %s ...", script_path.name)
    result = subprocess.run(
        [sys.executable, str(script_path)] + extra_args,
        capture_output=False,
    )
    if result.returncode != 0:
        log.warning("%s exited with code %d", script_path.name, result.returncode)
    else:
        log.info("%s OK", script_path.name)


def main() -> None:
    args   = parse_args()
    run_dir = Path(args.run_dir)
    log = setup_logging(run_dir)

    scripts_dir  = Path(__file__).parent.parent   # scripts/
    results_dir  = run_dir / "results"
    figures_dir  = run_dir / "figures"
    summary_path = results_dir / "phase4_summary.json"
    stats_path   = results_dir / "phase4_stats.json"

    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: compile ───────────────────────────────────────────────────
    run_script(
        scripts_dir / "phase4_compile_results.py",
        ["--results-dir", str(results_dir),
         "--output", str(summary_path),
         "--prefix", "real_"],
        log,
    )

    # ── Step 2: visualize ─────────────────────────────────────────────────
    if summary_path.exists():
        run_script(
            scripts_dir / "phase4_visualize.py",
            ["--summary", str(summary_path),
             "--output-dir", str(figures_dir)],
            log,
        )
    else:
        log.warning("Summary not found at %s; skipping visualization.", summary_path)

    # ── Step 3: stats ─────────────────────────────────────────────────────
    if summary_path.exists():
        run_script(
            scripts_dir / "phase4_stats.py",
            ["--summary", str(summary_path),
             "--output", str(stats_path)],
            log,
        )
    else:
        log.warning("Summary not found at %s; skipping stats.", summary_path)

    log.info("Results summary: %s", summary_path)
    log.info("Figures:         %s", figures_dir)
    log.info("Stats:           %s", stats_path)
    log.info("Stage 11 complete.")


if __name__ == "__main__":
    main()
