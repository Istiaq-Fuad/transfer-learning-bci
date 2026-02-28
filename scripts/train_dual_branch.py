"""Phase 2: Full Dual-Branch Model Training Script.

Trains the complete dual-branch architecture:
    Branch A: CWT Spectrogram (224x224x3) -> ViT-Tiny -> 192-dim
    Branch B: CSP (6 components) + Riemannian (253-dim) -> MLP(259->256->128) -> 128-dim
    Fusion:   AttentionFusion(192, 128) -> 128-dim
    Head:     MLP(128->64->2) -> Left / Right hand MI

Evaluation strategies:
    --strategy within_subject   5-fold stratified CV per subject, pooled
    --strategy loso             Leave-One-Subject-Out

Fusion ablation:
    --fusion attention   (default, thesis target)
    --fusion gated

Usage:
    # Synthetic data, fast smoke-test
    uv run python scripts/train_dual_branch.py --n-subjects 2 --n-folds 2 \\
        --epochs 3 --no-pretrained --batch-size 4

    # Synthetic, full run
    uv run python scripts/train_dual_branch.py --epochs 50 --strategy within_subject

    # Real BCI IV-2a (once downloaded)
    uv run python scripts/train_dual_branch.py --data real --data-dir ~/mne_data

    # Fusion ablation (attention vs gated)
    for fusion in attention gated; do
        uv run python scripts/train_dual_branch.py --fusion $fusion \\
            --output results/dual_branch_${fusion}.json
    done
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
from bci.training.cross_validation import (
    CVResult,
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

MODEL_NAME = "DualBranch-ViT+CSP+Riemann"


# ---------------------------------------------------------------------------
# Per-fold training helper
# ---------------------------------------------------------------------------


def _train_and_eval_fold(
    fold_idx: int,
    subject_id: int | None,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    builder: DualBranchFoldBuilder,
    device: str | torch.device,
    epochs: int,
    lr: float,
    batch_size: int,
    warmup_epochs: int,
    patience: int,
    pretrained: bool,
    fusion: str,
    unfreeze_last_n: int,
    checkpoint_dir: Path | None,
    seed: int,
) -> FoldResult:
    """Train the dual-branch model on one fold and return FoldResult."""
    set_seed(seed + fold_idx)  # slight variation per fold for stochasticity

    # --- Build fold datasets (fits CSP+Riemannian on train only) ---
    train_ds, test_ds, math_input_dim = builder.build_fold(X_train, y_train, X_test, y_test)

    # --- Instantiate fresh model ---
    model_config = ModelConfig(
        vit_model_name="vit_tiny_patch16_224",
        vit_pretrained=pretrained,
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

    if pretrained:
        model.freeze_vit_backbone(unfreeze_last_n_blocks=unfreeze_last_n)

    # --- forward_fn for dual-input batches: (image, features, label) ---
    _device = torch.device(device)

    def dual_forward_fn(batch):
        images, features, labels = batch
        images = images.to(_device)
        features = features.to(_device)
        labels = labels.to(_device)
        logits = model(images, features)
        return logits, labels

    # --- Train ---
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
        checkpoint_dir=checkpoint_dir,
        seed=seed,
        num_workers=0,
    )
    trainer.fit(train_ds, forward_fn=dual_forward_fn, model_tag=f"dual_fold{fold_idx}")

    # --- Predict on test set ---
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False, num_workers=0)
    y_pred, y_prob = trainer.predict(test_loader, forward_fn=dual_forward_fn)

    # --- Compute metrics ---
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
        "  Fold %d [S%s]: acc=%.2f%%  kappa=%.3f",
        fold_idx,
        f"{subject_id:02d}" if subject_id is not None else "??",
        result.accuracy,
        result.kappa,
    )
    return result


# ---------------------------------------------------------------------------
# Within-subject CV
# ---------------------------------------------------------------------------


def run_within_subject_cv(
    subject_data: dict[int, tuple[np.ndarray, np.ndarray]],
    builder: DualBranchFoldBuilder,
    n_folds: int,
    **kwargs,
) -> CVResult:
    """Run within-subject k-fold CV for all subjects."""
    all_folds: list[FoldResult] = []
    fold_counter = 0

    for subj_id, (X, y) in sorted(subject_data.items()):
        logger.info("Within-subject CV: subject %d (%d trials)...", subj_id, len(y))
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=kwargs["seed"])

        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            fold_result = _train_and_eval_fold(
                fold_idx=fold_counter,
                subject_id=subj_id,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                builder=builder,
                **kwargs,
            )
            all_folds.append(fold_result)
            fold_counter += 1

        subj_accs = [f.accuracy for f in all_folds if f.subject == subj_id]
        logger.info(
            "  Subject %02d: mean=%.2f%% ± %.2f%%",
            subj_id,
            float(np.mean(subj_accs)),
            float(np.std(subj_accs)),
        )

    return CVResult(
        strategy="within_subject",
        model_name=MODEL_NAME,
        folds=all_folds,
    )


# ---------------------------------------------------------------------------
# LOSO CV
# ---------------------------------------------------------------------------


def run_loso_cv(
    subject_data: dict[int, tuple[np.ndarray, np.ndarray]],
    builder: DualBranchFoldBuilder,
    **kwargs,
) -> CVResult:
    """Run Leave-One-Subject-Out CV."""
    subjects = sorted(subject_data.keys())
    all_folds: list[FoldResult] = []

    for fold_idx, test_subj in enumerate(subjects):
        train_Xs = [subject_data[s][0] for s in subjects if s != test_subj]
        train_ys = [subject_data[s][1] for s in subjects if s != test_subj]
        X_train = np.concatenate(train_Xs, axis=0)
        y_train = np.concatenate(train_ys, axis=0)
        X_test, y_test = subject_data[test_subj]

        logger.info(
            "LOSO fold %d/%d: test=S%02d, train=%d, test=%d",
            fold_idx + 1,
            len(subjects),
            test_subj,
            len(y_train),
            len(y_test),
        )
        fold_result = _train_and_eval_fold(
            fold_idx=fold_idx,
            subject_id=test_subj,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            builder=builder,
            **kwargs,
        )
        all_folds.append(fold_result)

    return CVResult(strategy="loso", model_name=MODEL_NAME, folds=all_folds)


# ---------------------------------------------------------------------------
# Printing + saving results
# ---------------------------------------------------------------------------


def print_results_table(result: CVResult, fusion: str) -> None:
    print("\n" + "=" * 65)
    print(f"  PHASE 2: {MODEL_NAME}")
    print(f"  Fusion: {fusion.upper()}")
    print("=" * 65)
    print(f"\n{'Strategy':<25} {'Accuracy':>10} {'±':>4} {'Std':>8} {'Kappa':>8} {'F1':>8}")
    print("-" * 65)
    print(
        f"{result.strategy:<25} "
        f"{result.mean_accuracy:>9.2f}% "
        f"{'':>4} "
        f"{result.std_accuracy:>7.2f}% "
        f"{result.mean_kappa:>8.4f} "
        f"{result.mean_f1:>8.4f}"
    )
    print("=" * 65)

    if result.per_subject_accuracy:
        print(f"\nPer-subject accuracy ({result.strategy}):")
        for subj, acc in sorted(result.per_subject_accuracy.items()):
            bar = "#" * int(acc / 5)
            print(f"  S{subj:02d}: {acc:6.2f}%  {bar}")
    print()


def save_results(result: CVResult, fusion: str, output_path: Path) -> None:
    data = {
        "model": MODEL_NAME,
        "fusion": fusion,
        "strategy": result.strategy,
        "mean_accuracy": result.mean_accuracy,
        "std_accuracy": result.std_accuracy,
        "mean_kappa": result.mean_kappa,
        "mean_f1": result.mean_f1,
        "n_folds": len(result.folds),
        "per_subject": result.per_subject_accuracy,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info("Results saved to %s", output_path)


# ---------------------------------------------------------------------------
# Real data loader
# ---------------------------------------------------------------------------


def load_real_data(data_dir: str) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    import mne
    from moabb.datasets import BNCI2014_001
    from moabb.paradigms import LeftRightImagery

    mne.set_log_level("ERROR")
    logger.info("Loading BCI IV-2a from %s ...", data_dir)
    dataset = BNCI2014_001()
    paradigm = LeftRightImagery(fmin=4.0, fmax=40.0, resample=128.0)

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
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2: Dual-Branch Model Training")
    parser.add_argument("--data", choices=["synthetic", "real"], default="synthetic")
    parser.add_argument("--data-dir", default="~/mne_data")
    parser.add_argument(
        "--strategy",
        choices=["within_subject", "loso"],
        default="within_subject",
    )
    parser.add_argument(
        "--fusion",
        choices=["attention", "gated"],
        default="attention",
        help="Fusion method (default: attention)",
    )
    parser.add_argument("--n-subjects", type=int, default=9, help="Subjects (synthetic only)")
    parser.add_argument("--n-folds", type=int, default=5, help="Folds (within-subject only)")
    parser.add_argument("--epochs", type=int, default=50, help="Max training epochs per fold")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--unfreeze-last-n", type=int, default=2)
    parser.add_argument("--csp-components", type=int, default=6)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        default="results/dual_branch_attention.json",
        help="Output JSON path",
    )
    parser.add_argument("--checkpoint-dir", default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)

    logger.info("=" * 60)
    logger.info("Phase 2: Dual-Branch Model Training")
    logger.info("  Data:         %s", args.data)
    logger.info("  Strategy:     %s", args.strategy)
    logger.info("  Fusion:       %s", args.fusion)
    logger.info("  Device:       %s", device)
    logger.info("  Pretrained:   %s", not args.no_pretrained)
    logger.info("  Epochs:       %d (patience=%d)", args.epochs, args.patience)
    logger.info("  LR:           %.2e", args.lr)
    logger.info("  Batch size:   %d", args.batch_size)
    logger.info("  CSP comps:    %d", args.csp_components)
    logger.info("  Seed:         %d", args.seed)
    logger.info("=" * 60)

    # --- Load / generate data ---
    if args.data == "synthetic":
        logger.info("Generating synthetic data (%d subjects)...", args.n_subjects)
        subject_data = make_synthetic_subject_data(n_subjects=args.n_subjects)
        channel_names = ["C3", "Cz", "C4"]
        sfreq = 128.0
    else:
        subject_data = load_real_data(args.data_dir)
        channel_names = ["C3", "Cz", "C4"]
        sfreq = 128.0

    if not subject_data:
        logger.error("No data loaded. Exiting.")
        return

    # --- Build fold builder ---
    builder = DualBranchFoldBuilder(
        csp_n_components=args.csp_components,
        riemann_estimator="oas",
        riemann_metric="riemann",
        sfreq=sfreq,
        channel_names=channel_names,
    )

    # Shared kwargs for fold training
    fold_kwargs = dict(
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        warmup_epochs=args.warmup_epochs,
        patience=args.patience,
        pretrained=not args.no_pretrained,
        fusion=args.fusion,
        unfreeze_last_n=args.unfreeze_last_n,
        checkpoint_dir=Path(args.checkpoint_dir) if args.checkpoint_dir else None,
        seed=args.seed,
    )

    # --- Run CV ---
    t0 = time.time()
    if args.strategy == "within_subject":
        logger.info("\nRunning within-subject %d-fold CV...", args.n_folds)
        result = run_within_subject_cv(subject_data, builder, n_folds=args.n_folds, **fold_kwargs)
    else:
        logger.info("\nRunning LOSO CV...")
        result = run_loso_cv(subject_data, builder, **fold_kwargs)

    elapsed = time.time() - t0
    logger.info("Total CV time: %.1fs (%.1f min)", elapsed, elapsed / 60)

    # --- Report ---
    print_results_table(result, fusion=args.fusion)
    save_results(result, fusion=args.fusion, output_path=Path(args.output))


if __name__ == "__main__":
    main()
