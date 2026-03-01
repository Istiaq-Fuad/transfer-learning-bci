"""Result index and manifest helpers."""

from __future__ import annotations

import json
from pathlib import Path


def _index_path(run_dir: Path) -> Path:
    return run_dir / "results" / "index.json"


def update_results_index(run_dir: Path, stage: str, outputs: dict[str, str]) -> None:
    """Merge outputs into the results index under a stage key."""
    path = _index_path(run_dir)
    if path.exists():
        with open(path) as f:
            index = json.load(f)
    else:
        index = {}

    stage_entry = index.get(stage, {})
    stage_outputs = stage_entry.get("outputs", {})
    stage_outputs.update(outputs)
    index[stage] = {"outputs": stage_outputs}

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(index, f, indent=2)


def write_manifest(
    run_dir: Path,
    stage: str,
    outputs: dict[str, str],
    meta: dict | None = None,
) -> Path:
    """Write a stage manifest with outputs and optional metadata."""
    path = run_dir / "results" / "manifests" / f"{stage}.json"
    payload = {"stage": stage, "outputs": outputs}
    if meta:
        payload["meta"] = meta

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path
