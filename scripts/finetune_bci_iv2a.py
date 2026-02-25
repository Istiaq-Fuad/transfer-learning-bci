"""Phase 3 – Step 2: Transfer Learning Fine-tuning on Target Dataset.

Loads a pretrained ViT checkpoint (from pretrain_physionet.py) and fine-tunes
the full DualBranchModel on the target dataset (BCI IV-2a or synthetic).

Compares three training conditions:
    scratch     - DualBranchModel trained from scratch (random ViT init)
    imagenet    - DualBranchModel with ImageNet-pretrained ViT (standard transfer)
    transfer    - DualBranchModel with EEG-domain pretrained ViT (our method)

Evaluation: within-subject k-fold CV (default 5-fold).

Outputs:
    results/finetune_scratch.json
    results/finetune_imagenet.json
    results/finetune_transfer.json

Usage:
    # Synthetic — smoke test
    uv run python scripts/finetune_bci_iv2a.py --data synthetic \\
        --checkpoint checkpoints/vit_pretrained_physionet.pt \\
        --n-subjects 2 --n-folds 2 --epochs 10

    # All three conditions in one run
    uv run python scripts/finetune_bci_iv2a.py --data synthetic \\
        --checkpoint checkpoints/vit_pretrained_physionet.pt \\
        --conditions scratch imagenet transfer \\
        --n-subjects 3 --n-folds 3 --epochs 20
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from bci.data.dual_branch_builder import DualBranchFoldBuilder
from bci.models.dual_branch import DualBranchModel
from bci.models.vit_branch import ViTBranch
from bci.training.cross_validation import (
    CVResult,
    FoldResult,
    make_synthetic_subject_data,
)
from bci.training.evaluation import compute_metrics
from bci.training.trainer import Trainer
from bci.utils.config import ModelConfig, SpectrogramConfig
from bci.utils.seed import get_device, set_seed

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logging.getLogger("mne").setLevel(logging.ERROR)
logging.getLogger("pyriemann").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

MODEL_NAME = "DualBranch-Transfer"


# ---------------------------------------------------------------------------
# Real BCI IV-2a loader
# ---------------------------------------------------------------------------

def load_bci_iv2a(data_dir: str, sfreq: float = 128.0) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    import mne
    from moabb.datasets import BNCI2014_001
    from moabb.paradigms import LeftRightImagery

    mne.set_log_level("ERROR")
    logger.info("Loading BCI IV-2a from %s ...", data_dir)
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
# Condition: how to initialise ViT
# ---------------------------------------------------------------------------

def _build_model(
    condition: str,
    math_input_dim: int,
    checkpoint_path: Path | None,
    unfreeze_last_n: int,
    fusion: str,
) -> DualBranchModel:
    """Build DualBranchModel with the appropriate ViT initialisation.

    Args:
        condition:       "scratch" | "imagenet" | "transfer"
        math_input_dim:  Dimensionality of handcrafted features.
        checkpoint_path: Path to EEG-pretrained ViT weights (transfer only).
        unfreeze_last_n: Transformer blocks to keep unfrozen (transfer/imagenet).
        fusion:          Fusion method name.

    Returns:
        Initialised DualBranchModel.
    """
    use_imagenet = condition in ("imagenet", "transfer")
    model_config = ModelConfig(
        vit_model_name="vit_tiny_patch16_224",
        vit_pretrained=use_imagenet,
        vit_drop_rate=0.1,
        csp_n_components=6,
        math_hidden_dims=[256, 128],
        math_drop_rate=0.3,
        fusion_method=fusion,
        fused_dim=128,
        classifier_hidden_dim=64,
        n_classes=2,
    )
    model = DualBranchModel(math_input_dim=math_input_dim, config=model_config)

    if condition == "transfer":
        if checkpoint_path is None or not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Transfer condition requires a checkpoint. "
                f"Path not found: {checkpoint_path}"
            )
        # Load EEG-pretrained weights into ViT branch
        # The checkpoint was saved from ViTBranch (as_feature_extractor=False),
        # so the head is a 2-class Linear. We load everything except the head
        # to stay compatible with the Identity head in DualBranchModel.
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        # Filter out the classification head (not used in feature-extractor mode)
        backbone_state = {
            k: v for k, v in ckpt.items()
            if not k.startswith("backbone.head")
        }
        missing, unexpected = model.vit_branch.backbone.load_state_dict(
            backbone_state, strict=False
        )
        logger.info(
            "Loaded EEG-pretrained ViT weights: %d missing, %d unexpected keys",
            len(missing), len(unexpected),
        )
        # Freeze backbone (unfreeze last N blocks + head)
        model.freeze_vit_backbone(unfreeze_last_n_blocks=unfreeze_last_n)

    elif condition == "imagenet":
        # Freeze most of ViT, unfreeze last N blocks
        model.freeze_vit_backbone(unfreeze_last_n_blocks=unfreeze_last_n)

    # condition == "scratch": all weights trainable (already default)
    return model


# ---------------------------------------------------------------------------
# Per-fold training
# ---------------------------------------------------------------------------

def _train_and_eval_fold(
    fold_idx: int,
    subject_id: int | None,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    condition: str,
    checkpoint_path: Path | None,
    builder: DualBranchFoldBuilder,
    device: str | torch.device,
    epochs: int,
    lr: float,
    batch_size: int,
    warmup_epochs: int,
    patience: int,
    unfreeze_last_n: int,
    fusion: str,
    seed: int,
) -> FoldResult:
    set_seed(seed + fold_idx)

    # Build fold datasets
    train_ds, test_ds, math_input_dim = builder.build_fold(
        X_train, y_train, X_test, y_test
    )

    # Build model for this condition
    model = _build_model(
        condition=condition,
        math_input_dim=math_input_dim,
        checkpoint_path=checkpoint_path,
        unfreeze_last_n=unfreeze_last_n,
        fusion=fusion,
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
    trainer.fit(
        train_ds,
        forward_fn=dual_fwd,
        model_tag=f"{condition}_fold{fold_idx}",
    )

    test_loader = DataLoader(
        test_ds, batch_size=batch_size * 2, shuffle=False, num_workers=0
    )
    y_pred, y_prob = trainer.predict(test_loader, forward_fn=dual_fwd)
    m = compute_metrics(y_test, y_pred, y_prob)

    result = FoldResult(
        fold=fold_idx,
        subject=subject_id,
        accuracy=m["accuracy"],
        kappa=m["kappa"],
        f1_macro=m["f1_macro"],
        n_train=len(y_train),
        n_test=len(y_test),
        y_true=y_test,
        y_pred=y_pred,
        y_prob=y_prob,
    )
    logger.info(
        "  [%s] Fold %d [S%s]: acc=%.2f%%  kappa=%.3f",
        condition.upper(),
        fold_idx,
        f"{subject_id:02d}" if subject_id is not None else "??",
        result.accuracy, result.kappa,
    )
    return result


# ---------------------------------------------------------------------------
# Within-subject CV for one condition
# ---------------------------------------------------------------------------

def run_condition(
    condition: str,
    subject_data: dict[int, tuple[np.ndarray, np.ndarray]],
    builder: DualBranchFoldBuilder,
    n_folds: int,
    checkpoint_path: Path | None,
    **kwargs,
) -> CVResult:
    all_folds: list[FoldResult] = []
    fold_counter = 0

    for subj_id, (X, y) in sorted(subject_data.items()):
        logger.info(
            "[%s] Within-subject CV: S%02d (%d trials)...",
            condition.upper(), subj_id, len(y),
        )
        skf = StratifiedKFold(
            n_splits=n_folds, shuffle=True, random_state=kwargs["seed"]
        )
        for train_idx, test_idx in skf.split(X, y):
            fr = _train_and_eval_fold(
                fold_idx=fold_counter,
                subject_id=subj_id,
                X_train=X[train_idx], y_train=y[train_idx],
                X_test=X[test_idx],   y_test=y[test_idx],
                condition=condition,
                checkpoint_path=checkpoint_path,
                builder=builder,
                **kwargs,
            )
            all_folds.append(fr)
            fold_counter += 1

        subj_accs = [f.accuracy for f in all_folds if f.subject == subj_id]
        logger.info(
            "  [%s] S%02d: mean=%.2f%% ± %.2f%%",
            condition.upper(), subj_id,
            float(np.mean(subj_accs)), float(np.std(subj_accs)),
        )

    return CVResult(
        strategy="within_subject",
        model_name=f"{MODEL_NAME}-{condition}",
        folds=all_folds,
    )


# ---------------------------------------------------------------------------
# Printing + saving
# ---------------------------------------------------------------------------

def print_comparison_table(results: dict[str, CVResult]) -> None:
    print("\n" + "=" * 70)
    print("  PHASE 3: Transfer Learning Comparison")
    print("=" * 70)
    print(f"\n{'Condition':<15} {'Accuracy':>10} {'±Std':>8} {'Kappa':>8} {'F1':>8}")
    print("-" * 70)
    for cond, r in results.items():
        print(
            f"{cond:<15} "
            f"{r.mean_accuracy:>9.2f}% "
            f"{r.std_accuracy:>7.2f}% "
            f"{r.mean_kappa:>8.4f} "
            f"{r.mean_f1:>8.4f}"
        )
    print("=" * 70 + "\n")


def save_condition_results(
    condition: str,
    result: CVResult,
    output_dir: Path,
) -> Path:
    data = {
        "model": f"{MODEL_NAME}-{condition}",
        "condition": condition,
        "strategy": result.strategy,
        "mean_accuracy": result.mean_accuracy,
        "std_accuracy": result.std_accuracy,
        "mean_kappa": result.mean_kappa,
        "mean_f1": result.mean_f1,
        "n_folds": len(result.folds),
        "per_subject": result.per_subject_accuracy,
    }
    out = output_dir / f"finetune_{condition}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(data, f, indent=2)
    logger.info("Results saved to %s", out)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 3 Step 2: Transfer Learning Fine-tuning"
    )
    parser.add_argument(
        "--data", choices=["synthetic", "real"], default="synthetic"
    )
    parser.add_argument("--data-dir", default="~/mne_data")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/vit_pretrained_physionet.pt",
        help="Path to EEG-pretrained ViT checkpoint (for 'transfer' condition)",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=["scratch", "imagenet", "transfer"],
        default=["scratch", "imagenet", "transfer"],
        help="Training conditions to evaluate",
    )
    parser.add_argument(
        "--fusion",
        choices=["attention", "concat", "gated"],
        default="attention",
    )
    parser.add_argument("--n-subjects", type=int, default=3)
    parser.add_argument("--n-folds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--unfreeze-last-n", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    checkpoint_path = Path(args.checkpoint)

    logger.info("=" * 60)
    logger.info("Phase 3 – Transfer Learning Fine-tuning")
    logger.info("  Target data:   %s", args.data)
    logger.info("  Conditions:    %s", args.conditions)
    logger.info("  Fusion:        %s", args.fusion)
    logger.info("  Device:        %s", device)
    logger.info("  Epochs:        %d (patience=%d)", args.epochs, args.patience)
    logger.info("  Checkpoint:    %s", checkpoint_path)
    logger.info("=" * 60)

    # Load target data
    if args.data == "synthetic":
        logger.info("Generating synthetic target data (%d subjects)...", args.n_subjects)
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

    # Build fold builder (shared across conditions)
    builder = DualBranchFoldBuilder(
        csp_n_components=6,
        csp_reg="ledoit_wolf",
        riemann_estimator="lwf",
        riemann_metric="riemann",
        sfreq=sfreq,
        channel_names=channel_names,
    )

    fold_kwargs = dict(
        builder=builder,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        warmup_epochs=args.warmup_epochs,
        patience=args.patience,
        unfreeze_last_n=args.unfreeze_last_n,
        fusion=args.fusion,
        seed=args.seed,
    )

    # Run each condition
    output_dir = Path(args.output_dir)
    all_results: dict[str, CVResult] = {}
    t_total = time.time()

    for condition in args.conditions:
        logger.info("\n--- Condition: %s ---", condition.upper())
        ckpt = checkpoint_path if condition == "transfer" else None
        t0 = time.time()
        result = run_condition(
            condition=condition,
            subject_data=subject_data,
            n_folds=args.n_folds,
            checkpoint_path=ckpt,
            **fold_kwargs,
        )
        elapsed = time.time() - t0
        logger.info(
            "[%s] Done in %.1fs | acc=%.2f%% ± %.2f%%",
            condition.upper(), elapsed,
            result.mean_accuracy, result.std_accuracy,
        )
        all_results[condition] = result
        save_condition_results(condition, result, output_dir)

    elapsed_total = time.time() - t_total
    logger.info("\nTotal time: %.1fs (%.1f min)", elapsed_total, elapsed_total / 60)

    print_comparison_table(all_results)


if __name__ == "__main__":
    main()
