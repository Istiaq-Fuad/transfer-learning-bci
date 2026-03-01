"""Split management utilities for reproducible CV runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold


@dataclass
class SplitSpec:
    dataset: str
    n_folds: int
    seed: int
    within_subject: dict[int, list[dict[str, list[int]]]]
    loso_subjects: list[int]

    def to_json(self) -> dict:
        return {
            "dataset": self.dataset,
            "n_folds": self.n_folds,
            "seed": self.seed,
            "within_subject": {str(sid): folds for sid, folds in self.within_subject.items()},
            "loso_subjects": self.loso_subjects,
        }

    @classmethod
    def from_json(cls, data: dict) -> "SplitSpec":
        within = {int(sid): folds for sid, folds in data.get("within_subject", {}).items()}
        return cls(
            dataset=data.get("dataset", "unknown"),
            n_folds=int(data.get("n_folds", 0)),
            seed=int(data.get("seed", 0)),
            within_subject=within,
            loso_subjects=[int(s) for s in data.get("loso_subjects", [])],
        )


def make_within_subject_splits(
    subject_data: dict[int, tuple[np.ndarray, np.ndarray]],
    n_folds: int,
    seed: int,
) -> dict[int, list[dict[str, list[int]]]]:
    """Create stratified k-fold indices per subject."""
    splits: dict[int, list[dict[str, list[int]]]] = {}
    for sid, (_, y) in sorted(subject_data.items()):
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        folds: list[dict[str, list[int]]] = []
        for train_idx, test_idx in skf.split(np.zeros(len(y)), y):
            folds.append(
                {
                    "train_idx": train_idx.tolist(),
                    "test_idx": test_idx.tolist(),
                }
            )
        splits[int(sid)] = folds
    return splits


def make_loso_subjects(subject_data: dict[int, tuple[np.ndarray, np.ndarray]]) -> list[int]:
    """Return deterministic LOSO subject ordering."""
    return sorted(int(sid) for sid in subject_data.keys())


def build_split_spec(
    dataset: str,
    subject_data: dict[int, tuple[np.ndarray, np.ndarray]],
    n_folds: int,
    seed: int,
) -> SplitSpec:
    within = make_within_subject_splits(subject_data, n_folds, seed)
    loso_subjects = make_loso_subjects(subject_data)
    return SplitSpec(
        dataset=dataset,
        n_folds=n_folds,
        seed=seed,
        within_subject=within,
        loso_subjects=loso_subjects,
    )


def splits_path(run_dir: Path, dataset: str) -> Path:
    return run_dir / "splits" / f"{dataset}_splits.json"


def save_splits(run_dir: Path, spec: SplitSpec) -> Path:
    path = splits_path(run_dir, spec.dataset)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(spec.to_json(), f, indent=2)
    return path


def load_splits(run_dir: Path, dataset: str) -> SplitSpec | None:
    path = splits_path(run_dir, dataset)
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    return SplitSpec.from_json(data)


def get_or_create_splits(
    run_dir: Path,
    dataset: str,
    subject_data: dict[int, tuple[np.ndarray, np.ndarray]],
    n_folds: int,
    seed: int,
) -> SplitSpec:
    spec = load_splits(run_dir, dataset)
    if spec is None:
        spec = build_split_spec(dataset, subject_data, n_folds, seed)
        save_splits(run_dir, spec)
        return spec

    if spec.n_folds != n_folds or spec.seed != seed:
        spec = build_split_spec(dataset, subject_data, n_folds, seed)
        save_splits(run_dir, spec)
        return spec

    return spec
