"""Phase 3 – Step 3: Reduced-Data Experiments.

Trains the DualBranchModel at multiple training-set size fractions and measures
how accuracy degrades as data is reduced — comparing:
    scratch   - random ViT init (no pretraining)
    transfer  - EEG-domain pretrained ViT (from pretrain_physionet.py)

This is the core thesis experiment: demonstrating that transfer learning
maintains higher accuracy when labelled data is scarce.

Evaluation strategy:
    For each fraction f in [0.10, 0.25, 0.50, 0.75, 1.00]:
        - Randomly subsample f × n_train trials
        - Train both models, record test accuracy
        - Repeat n_repeats times (different random subsets)
        - Report mean ± std accuracy vs. fraction

Outputs:
    results/reduced_data_results.json   (per-fraction, per-condition)
    results/reduced_data_summary.json   (summary table)

Usage:
    # Fast synthetic smoke-test
    uv run python scripts/reduced_data_experiment.py --data synthetic \\
        --n-subjects 2 --n-repeats 2 --epochs 10 \\
        --checkpoint checkpoints/vit_pretrained_physionet.pt

    # Full run with real data (after download)
    uv run python scripts/reduced_data_experiment.py --data real \\
        --checkpoint checkpoints/vit_pretrained_physionet.pt \\
        --n-repeats 3 --epochs 50
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch.utils.data import DataLoader

from bci.data.dual_branch_builder import DualBranchFoldBuilder
from bci.models.dual_branch import DualBranchModel
from bci.training.cross_validation import (
    FoldResult,
    make_synthetic_subject_data,
)
from bci.training.evaluation import compute_metrics
from bci.training.trainer import Trainer
from bci.utils.config import ModelConfig
from bci.utils.seed import get_device, set_seed

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logging.getLogger("mne").setLevel(logging.ERROR)
logging.getLogger("pyriemann").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Real BCI IV-2a loader (same as finetune script)
# ---------------------------------------------------------------------------

def load_bci_iv2a(
    data_dir: str, sfreq: float = 128.0
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    import mne
    from moabb.datasets import BNCI2014_001
    from moabb.paradigms import LeftRightImagery

    mne.set_log_level("ERROR")
    dataset = BNCI2014_001()
    paradigm = LeftRightImagery(fmin=4.0, fmax=40.0, resample=sfreq)

    subject_data: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for sid in dataset.subject_list:
        try:
            X, y_labels, _ = paradigm.get_data(dataset=dataset, subjects=[sid])
            classes = sorted(np.unique(y_labels))
            label_map = {c: i for i, c in enumerate(classes)}
            y = np.array([label_map[lb] for lb in y_labels], dtype=np.int64)
            subject_data[sid] = (X.astype(np.float32), y)
            logger.info("  Subject %d: X=%s", sid, X.shape)
        except Exception as e:
            logger.warning("  Skipping subject %d: %s", sid, e)
    return subject_data


# ---------------------------------------------------------------------------
# Build model for a condition
# ---------------------------------------------------------------------------

def _build_model(
    condition: str,
    math_input_dim: int,
    checkpoint_path: Path | None,
    unfreeze_last_n: int,
) -> DualBranchModel:
    use_imagenet = condition in ("imagenet", "transfer")
    model_config = ModelConfig(
        vit_model_name="vit_tiny_patch16_224",
        vit_pretrained=use_imagenet,
        vit_drop_rate=0.1,
        csp_n_components=6,
        math_hidden_dims=[256, 128],
        math_drop_rate=0.3,
        fusion_method="attention",
        fused_dim=128,
        classifier_hidden_dim=64,
        n_classes=2,
    )
    model = DualBranchModel(math_input_dim=math_input_dim, config=model_config)

    if condition == "transfer":
        if checkpoint_path is None or not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Transfer condition requires a checkpoint: {checkpoint_path}"
            )
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        backbone_state = {
            k: v for k, v in ckpt.items() if not k.startswith("backbone.head")
        }
        model.vit_branch.backbone.load_state_dict(backbone_state, strict=False)
        model.freeze_vit_backbone(unfreeze_last_n_blocks=unfreeze_last_n)

    elif condition == "imagenet":
        model.freeze_vit_backbone(unfreeze_last_n_blocks=unfreeze_last_n)

    return model


# ---------------------------------------------------------------------------
# Single reduced-data trial
# ---------------------------------------------------------------------------

def _run_one_trial(
    condition: str,
    X_train_full: np.ndarray,
    y_train_full: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    fraction: float,
    builder: DualBranchFoldBuilder,
    checkpoint_path: Path | None,
    device: str | torch.device,
    epochs: int,
    lr: float,
    batch_size: int,
    warmup_epochs: int,
    patience: int,
    unfreeze_last_n: int,
    seed: int,
) -> float:
    """Train on a fraction of training data, return test accuracy."""
    set_seed(seed)

    # Subsample training data
    if fraction < 1.0:
        n_keep = max(2, int(len(y_train_full) * fraction))
        sss = StratifiedShuffleSplit(
            n_splits=1, train_size=n_keep, random_state=seed
        )
        keep_idx, _ = next(sss.split(X_train_full, y_train_full))
        X_train = X_train_full[keep_idx]
        y_train = y_train_full[keep_idx]
    else:
        X_train = X_train_full
        y_train = y_train_full

    # Build fold datasets
    train_ds, test_ds, math_input_dim = builder.build_fold(
        X_train, y_train, X_test, y_test
    )

    # Build model
    model = _build_model(
        condition=condition,
        math_input_dim=math_input_dim,
        checkpoint_path=checkpoint_path,
        unfreeze_last_n=unfreeze_last_n,
    )

    _device = torch.device(device)

    def dual_fwd(batch):
        imgs, feats, labels = batch
        return model(imgs.to(_device), feats.to(_device)), labels.to(_device)

    trainer = Trainer(
        model=model,
        device=device,
        learning_rate=lr,
        weight_decay=1e-4,
        epochs=epochs,
        batch_size=batch_size,
        warmup_epochs=warmup_epochs,
        patience=patience,
        label_smoothing=0.1,
        val_fraction=0.2,
        seed=seed,
        num_workers=0,
    )
    trainer.fit(train_ds, forward_fn=dual_fwd, model_tag=f"{condition}_f{fraction:.0%}")

    test_loader = DataLoader(
        test_ds, batch_size=batch_size * 2, shuffle=False, num_workers=0
    )
    y_pred, y_prob = trainer.predict(test_loader, forward_fn=dual_fwd)
    m = compute_metrics(y_test, y_pred, y_prob)
    return m["accuracy"]


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def run_reduced_data_experiment(
    subject_data: dict[int, tuple[np.ndarray, np.ndarray]],
    conditions: list[str],
    fractions: list[float],
    n_repeats: int,
    n_folds: int,
    checkpoint_path: Path | None,
    builder: DualBranchFoldBuilder,
    device: str | torch.device,
    epochs: int,
    lr: float,
    batch_size: int,
    warmup_epochs: int,
    patience: int,
    unfreeze_last_n: int,
    seed: int,
) -> dict:
    """Run reduced-data experiment across fractions, conditions, subjects.

    Returns a nested dict:
        results[condition][fraction_str] = {"mean": float, "std": float, "runs": list}
    """
    results: dict[str, dict[str, dict]] = {c: {} for c in conditions}

    for fraction in fractions:
        frac_str = f"{fraction:.2f}"
        logger.info("\n=== Fraction %.0f%% ===", fraction * 100)

        # Collect accuracy across all subjects × folds × repeats per condition
        cond_accs: dict[str, list[float]] = {c: [] for c in conditions}

        for subj_id, (X, y) in sorted(subject_data.items()):
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
            for fold_i, (train_idx, test_idx) in enumerate(skf.split(X, y)):
                X_train_full, X_test = X[train_idx], X[test_idx]
                y_train_full, y_test = y[train_idx], y[test_idx]

                for rep in range(n_repeats):
                    trial_seed = seed + subj_id * 1000 + fold_i * 100 + rep

                    for condition in conditions:
                        ckpt = checkpoint_path if condition == "transfer" else None
                        try:
                            acc = _run_one_trial(
                                condition=condition,
                                X_train_full=X_train_full,
                                y_train_full=y_train_full,
                                X_test=X_test,
                                y_test=y_test,
                                fraction=fraction,
                                builder=builder,
                                checkpoint_path=ckpt,
                                device=device,
                                epochs=epochs,
                                lr=lr,
                                batch_size=batch_size,
                                warmup_epochs=warmup_epochs,
                                patience=patience,
                                unfreeze_last_n=unfreeze_last_n,
                                seed=trial_seed,
                            )
                        except Exception as e:
                            logger.warning(
                                "  Trial failed (S%d fold%d rep%d %s): %s",
                                subj_id, fold_i, rep, condition, e,
                            )
                            acc = float("nan")

                        cond_accs[condition].append(acc)
                        logger.info(
                            "  %s | S%02d fold%d rep%d | frac=%.0f%% | acc=%.2f%%",
                            condition.upper(), subj_id, fold_i, rep,
                            fraction * 100, acc,
                        )

        # Store stats for this fraction
        for condition in conditions:
            accs = [a for a in cond_accs[condition] if not np.isnan(a)]
            results[condition][frac_str] = {
                "fraction": fraction,
                "mean": float(np.mean(accs)) if accs else float("nan"),
                "std": float(np.std(accs)) if accs else float("nan"),
                "n_runs": len(accs),
                "runs": accs,
            }
            logger.info(
                "  %s @ %.0f%%: %.2f%% ± %.2f%% (n=%d)",
                condition.upper(), fraction * 100,
                results[condition][frac_str]["mean"],
                results[condition][frac_str]["std"],
                len(accs),
            )

    return results


# ---------------------------------------------------------------------------
# Print + save
# ---------------------------------------------------------------------------

def print_reduced_data_table(results: dict, fractions: list[float]) -> None:
    conditions = list(results.keys())
    header = f"\n{'Fraction':>10}"
    for c in conditions:
        header += f"  {c.upper():>20}"
    print("\n" + "=" * (12 + 22 * len(conditions)))
    print("  PHASE 3: Reduced-Data Accuracy")
    print("=" * (12 + 22 * len(conditions)))
    print(header)
    print("-" * (12 + 22 * len(conditions)))
    for fraction in fractions:
        frac_str = f"{fraction:.2f}"
        row = f"{fraction*100:>9.0f}%"
        for c in conditions:
            d = results[c].get(frac_str, {})
            mean = d.get("mean", float("nan"))
            std = d.get("std", float("nan"))
            row += f"  {mean:>8.2f}% ± {std:>5.2f}%"
        print(row)
    print("=" * (12 + 22 * len(conditions)) + "\n")


def save_results(results: dict, fractions: list[float], output_path: Path) -> None:
    summary = {}
    for cond, frac_data in results.items():
        summary[cond] = {
            frac_str: {
                "fraction_pct": round(d["fraction"] * 100),
                "mean_accuracy": round(d["mean"], 4),
                "std_accuracy": round(d["std"], 4),
                "n_runs": d["n_runs"],
            }
            for frac_str, d in frac_data.items()
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"fractions": fractions, "results": summary}, f, indent=2)
    logger.info("Results saved to %s", output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 3 Step 3: Reduced-Data Transfer Learning Experiment"
    )
    parser.add_argument("--data", choices=["synthetic", "real"], default="synthetic")
    parser.add_argument("--data-dir", default="~/mne_data")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/vit_pretrained_physionet.pt",
        help="EEG-pretrained ViT checkpoint (transfer condition)",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=["scratch", "imagenet", "transfer"],
        default=["scratch", "transfer"],
    )
    parser.add_argument(
        "--fractions",
        nargs="+",
        type=float,
        default=[0.10, 0.25, 0.50, 0.75, 1.00],
        help="Training-set fractions to evaluate",
    )
    parser.add_argument(
        "--n-repeats", type=int, default=3,
        help="Repetitions per (fraction, fold) for variance estimation",
    )
    parser.add_argument("--n-subjects", type=int, default=3)
    parser.add_argument("--n-folds", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--unfreeze-last-n", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        default="results/reduced_data_results.json",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    checkpoint_path = Path(args.checkpoint)

    logger.info("=" * 60)
    logger.info("Phase 3 – Reduced-Data Experiment")
    logger.info("  Target data:  %s", args.data)
    logger.info("  Conditions:   %s", args.conditions)
    logger.info("  Fractions:    %s", args.fractions)
    logger.info("  N subjects:   %d", args.n_subjects)
    logger.info("  N folds:      %d", args.n_folds)
    logger.info("  N repeats:    %d", args.n_repeats)
    logger.info("  Epochs:       %d (patience=%d)", args.epochs, args.patience)
    logger.info("  Checkpoint:   %s", checkpoint_path)
    logger.info("  Device:       %s", device)
    logger.info("=" * 60)

    # Load target data
    if args.data == "synthetic":
        logger.info("Generating synthetic data (%d subjects)...", args.n_subjects)
        subject_data = make_synthetic_subject_data(n_subjects=args.n_subjects)
        channel_names = ["C3", "Cz", "C4"]
        sfreq = 128.0
    else:
        subject_data = load_bci_iv2a(args.data_dir)
        channel_names = ["C3", "Cz", "C4"]
        sfreq = 128.0

    if not subject_data:
        logger.error("No target data loaded. Exiting.")
        return

    builder = DualBranchFoldBuilder(
        csp_n_components=6,
        csp_reg="ledoit_wolf",
        riemann_estimator="lwf",
        riemann_metric="riemann",
        sfreq=sfreq,
        channel_names=channel_names,
    )

    t0 = time.time()
    results = run_reduced_data_experiment(
        subject_data=subject_data,
        conditions=args.conditions,
        fractions=args.fractions,
        n_repeats=args.n_repeats,
        n_folds=args.n_folds,
        checkpoint_path=checkpoint_path,
        builder=builder,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        warmup_epochs=args.warmup_epochs,
        patience=args.patience,
        unfreeze_last_n=args.unfreeze_last_n,
        seed=args.seed,
    )
    elapsed = time.time() - t0
    logger.info("Total experiment time: %.1fs (%.1f min)", elapsed, elapsed / 60)

    print_reduced_data_table(results, args.fractions)
    save_results(results, args.fractions, Path(args.output))


if __name__ == "__main__":
    main()
