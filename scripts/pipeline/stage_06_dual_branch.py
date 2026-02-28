"""Stage 06 – Dual-branch ablation: attention vs gated fusion (within-subject + LOSO).

Loads BCI IV-2a epochs from the .npz cache (Stage 01) and pre-cached
9-channel spectrograms (Stage 01). Uses PhysioNet-pretrained ViT backbone
weights from Stage 04. Trains the full DualBranchModel with both attention
and gated fusion methods as an ablation study.

Runs within-subject 5-fold CV and LOSO CV for each fusion method.

Prerequisite: Stage 01 and Stage 04 must have been run first.

Output::

    <run-dir>/results/real_dual_branch_attention_vit.json
    <run-dir>/results/real_dual_branch_attention_vit_loso.json
    <run-dir>/results/real_dual_branch_gated_vit.json
    <run-dir>/plots/stage_06_vit_attention/
    <run-dir>/plots/stage_06_vit_gated/

Usage::

    uv run python scripts/pipeline/stage_06_dual_branch.py --run-dir runs/my_run
    uv run python scripts/pipeline/stage_06_dual_branch.py \\
        --run-dir runs/my_run --epochs 50 --batch-size 32 --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import bci.models.vit_branch as _vit_mod
from bci.utils.logging import setup_stage_logging
from bci.utils.visualization import save_fold_plots

_BACKBONE_SHORT = {
    _vit_mod.MODEL_NAME: _vit_mod.BACKBONE_SHORT,
}


def _build_model(
    condition: str,
    math_input_dim: int,
    checkpoint_path: Path | None,
    unfreeze_last_n: int = 2,
    fusion: str = "attention",
):
    """Build and return a DualBranchModel for a given transfer condition.

    Parameters
    ----------
    condition:
        One of ``"scratch"``, ``"imagenet"``, or ``"transfer"``.

        - ``"scratch"``: ViT backbone initialised with *random* weights.
        - ``"imagenet"``: ViT backbone initialised from ImageNet pretrained weights.
          The backbone is then partially frozen (all except the last
          *unfreeze_last_n* transformer blocks).
        - ``"transfer"``: Like ``"imagenet"`` but additionally loads
          PhysioNet-pretrained weights from *checkpoint_path* into the backbone.
    math_input_dim:
        Dimensionality of the handcrafted feature vector (CSP + Riemannian
        tangent-space features concatenated).
    checkpoint_path:
        Path to the backbone ``.pt`` checkpoint file saved by Stage 04 /
        :func:`pretrain_vit`.  Required when *condition* is ``"transfer"``.
    unfreeze_last_n:
        Number of trailing transformer blocks to keep trainable when the backbone
        is partially frozen (``"imagenet"`` and ``"transfer"`` conditions).
    fusion:
        Fusion method for the dual-branch model: ``"attention"`` or ``"gated"``.

    Returns
    -------
    DualBranchModel
    """
    import torch

    from bci.models.dual_branch import DualBranchModel
    from bci.utils.config import ModelConfig

    use_imagenet = condition in ("imagenet", "transfer")
    cfg = ModelConfig(
        vit_model_name="vit_tiny_patch16_224",
        vit_pretrained=use_imagenet,
        vit_drop_rate=0.1,
        in_chans=9,
        csp_n_components=6,
        math_hidden_dims=[256, 128],
        math_drop_rate=0.3,
        fusion_method=fusion,
        fused_dim=128,
        classifier_hidden_dim=64,
        n_classes=2,
    )
    model = DualBranchModel(math_input_dim=math_input_dim, config=cfg)

    if condition == "transfer":
        if checkpoint_path is None or not Path(checkpoint_path).exists():
            raise FileNotFoundError(
                f"Transfer condition requires a valid checkpoint; got: {checkpoint_path}"
            )
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        backbone_state = {
            k: v
            for k, v in ckpt.items()
            if not (k.startswith("head") or k.startswith("classifier"))
        }
        model.vit_branch.backbone.load_state_dict(backbone_state, strict=False)

    if condition in ("imagenet", "transfer"):
        model.freeze_vit_backbone(unfreeze_last_n_blocks=unfreeze_last_n)

    return model


def _train_and_eval_fold(
    fold_idx: int,
    subject_id: int,
    X_train,
    y_train,
    X_test,
    y_test,
    condition: str,
    checkpoint_path: Path | None,
    builder,
    device: str = "cpu",
    epochs: int = 10,
    lr: float = 1e-4,
    batch_size: int = 16,
    warmup_epochs: int = 2,
    patience: int = 10,
    unfreeze_last_n: int = 2,
    fusion: str = "attention",
    seed: int = 0,
):
    """Train and evaluate the dual-branch model for one CV fold.

    Parameters
    ----------
    fold_idx:
        Zero-based fold index (used for seeding and logging).
    subject_id:
        Subject identifier (for logging).
    X_train, y_train:
        Raw EEG training data of shape ``(n_train, n_channels, n_times)`` and
        labels ``(n_train,)``.
    X_test, y_test:
        Raw EEG test data and labels.
    condition:
        Transfer condition passed to :func:`_build_model`.
    checkpoint_path:
        ViT backbone checkpoint path (required for ``"transfer"`` condition).
    builder:
        A :class:`bci.data.dual_branch_builder.DualBranchFoldBuilder` instance
        used to build the TensorDataset pair for this fold.
    device:
        PyTorch device string.
    epochs, lr, batch_size, warmup_epochs, patience:
        Trainer hyper-parameters.
    unfreeze_last_n:
        Passed to :func:`_build_model`.
    fusion:
        Fusion method passed to :func:`_build_model`.
    seed:
        Random seed.

    Returns
    -------
    FoldResult
        A :class:`bci.training.cross_validation.FoldResult` with ``accuracy``,
        ``kappa``, ``f1_macro``, ``n_train``, ``n_test``, ``y_true``,
        ``y_pred``, ``y_prob`` populated.
    """
    import torch
    from torch.utils.data import DataLoader

    from bci.training.cross_validation import FoldResult
    from bci.training.evaluation import compute_metrics
    from bci.training.trainer import Trainer
    from bci.utils.seed import set_seed

    set_seed(seed + fold_idx)
    _device = torch.device(device)

    train_ds, test_ds, math_input_dim = builder.build_fold(X_train, y_train, X_test, y_test)
    model = _build_model(
        condition=condition,
        math_input_dim=math_input_dim,
        checkpoint_path=checkpoint_path,
        unfreeze_last_n=unfreeze_last_n,
        fusion=fusion,
    )

    def fwd(batch, _m=model):
        imgs, feats, labels = batch
        return _m(imgs.to(_device), feats.to(_device)), labels.to(_device)

    backbone_scale = 0.1 if condition in ("imagenet", "transfer") else None
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
        backbone_lr_scale=backbone_scale,
    )
    trainer.fit(train_ds, forward_fn=fwd, model_tag=f"{condition}_fold{fold_idx}")

    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False, num_workers=0)
    y_pred, y_prob = trainer.predict(test_loader, forward_fn=fwd)
    m = compute_metrics(y_test, y_pred, y_prob)
    return FoldResult(
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


_FUSED_DIM = {
    _vit_mod.MODEL_NAME: _vit_mod.DEFAULT_FUSED_DIM,
}
_CLASSIFIER_HIDDEN = {
    _vit_mod.MODEL_NAME: _vit_mod.DEFAULT_CLS_HIDDEN,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 06: Dual-branch fusion ablation (attention vs gated).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--run-dir", required=True)
    p.add_argument(
        "--processed-dir",
        default=None,
        help="Root of processed .npz cache (default: data/processed/)",
    )
    p.add_argument("--device", default="auto")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--backbone",
        default="vit_tiny_patch16_224",
        choices=list(_BACKBONE_SHORT.keys()),
    )
    p.add_argument(
        "--checkpoint",
        default=None,
        help="Path to PhysioNet-pretrained backbone checkpoint from Stage 04 "
        "(default: <run-dir>/checkpoints/vit_pretrained_physionet_vit.pt)",
    )
    p.add_argument(
        "--fusions",
        nargs="+",
        default=["attention", "gated"],
        choices=["attention", "gated"],
        help="Fusion methods to evaluate",
    )
    p.add_argument(
        "--strategies",
        nargs="+",
        default=["within_subject", "loso"],
        choices=["within_subject", "loso"],
    )
    return p.parse_args()


def run_dual_branch(
    fusion: str,
    strategy: str,
    backbone: str,
    bshort: str,
    subject_data: dict,
    subject_spec_data: dict,
    spec_mean,
    spec_std,
    checkpoint_path: Path,
    run_dir: Path,
    n_folds: int,
    epochs: int,
    batch_size: int,
    device: str,
    seed: int,
    log,
) -> Path:
    """Run dual-branch CV for one (fusion, strategy) pair and save results."""
    import numpy as np
    import torch
    from sklearn.model_selection import StratifiedKFold
    from torch.utils.data import DataLoader, TensorDataset

    from bci.data.dual_branch_builder import DualBranchFoldBuilder
    from bci.models.dual_branch import DualBranchModel
    from bci.training.cross_validation import CVResult, FoldResult
    from bci.training.evaluation import compute_metrics
    from bci.training.trainer import Trainer
    from bci.utils.config import ModelConfig
    from bci.utils.seed import set_seed

    MODEL_NAME = f"DualBranch-{bshort.upper()}+CSP+Riemann"
    tag_base = f"real_dual_branch_{fusion}_{bshort}"
    if strategy == "loso":
        tag_base += "_loso"
    out_path = run_dir / "results" / f"{tag_base}.json"
    plots_dir = run_dir / "plots" / f"stage_06_{bshort}_{fusion}_{strategy}"

    if out_path.exists():
        log.info("Already exists: %s – skipping.", out_path)
        return out_path

    _device = torch.device(device)
    fused_dim = _FUSED_DIM.get(backbone, 256)
    cls_hidden = _CLASSIFIER_HIDDEN.get(backbone, 128)

    builder = DualBranchFoldBuilder(
        csp_n_components=6,
        riemann_estimator="oas",
        riemann_metric="riemann",
        sfreq=128.0,
    )

    def build_model(math_input_dim: int) -> DualBranchModel:
        cfg = ModelConfig(
            vit_model_name=backbone,
            vit_pretrained=True,
            vit_drop_rate=0.1,
            in_chans=9,
            csp_n_components=6,
            math_hidden_dims=[256, 128],
            math_drop_rate=0.3,
            fusion_method=fusion,
            fused_dim=fused_dim,
            classifier_hidden_dim=cls_hidden,
            n_classes=2,
        )
        model = DualBranchModel(math_input_dim=math_input_dim, config=cfg)
        if checkpoint_path.exists():
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            backbone_state = {
                k: v
                for k, v in ckpt.items()
                if not (k.startswith("head") or k.startswith("classifier"))
            }
            model.vit_branch.backbone.load_state_dict(backbone_state, strict=False)
        else:
            log.warning("Checkpoint not found at %s; using ImageNet init only.", checkpoint_path)
        model.freeze_vit_backbone(unfreeze_last_n_blocks=2)
        return model

    def normalise_specs(imgs: np.ndarray) -> np.ndarray:
        return (imgs - spec_mean[None, :, None, None]) / spec_std[None, :, None, None]

    t0 = time.time()
    all_folds: list[FoldResult] = []
    subjects = sorted(subject_data.keys())

    if strategy == "within_subject":
        fold_counter = 0
        for sid in subjects:
            X, y = subject_data[sid]
            spec_imgs, _ = subject_spec_data[sid]
            log.info("Subject %d (%d trials)...", sid, len(y))
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
            for train_idx, test_idx in skf.split(X, y):
                set_seed(seed + fold_counter)
                # Build handcrafted feature fold (uses raw EEG)
                train_ds_base, test_ds_base, math_input_dim = builder.build_fold(
                    X[train_idx], y[train_idx], X[test_idx], y[test_idx]
                )
                # Replace image tensors with cached normalised spectrograms
                spec_train = normalise_specs(spec_imgs[train_idx]).astype(np.float32)
                spec_test = normalise_specs(spec_imgs[test_idx]).astype(np.float32)
                # Extract feature tensors from builder output
                _, feats_train, labels_train = [t for t in zip(*train_ds_base.tensors)]
                _, feats_test, labels_test = [t for t in zip(*test_ds_base.tensors)]
                train_ds = TensorDataset(torch.from_numpy(spec_train), feats_train, labels_train)
                test_ds = TensorDataset(torch.from_numpy(spec_test), feats_test, labels_test)

                model = build_model(math_input_dim)

                def fwd(batch, _m=model):
                    imgs, feats, labels = batch
                    return _m(imgs.to(_device), feats.to(_device)), labels.to(_device)

                trainer = Trainer(
                    model=model,
                    device=device,
                    learning_rate=1e-4,
                    weight_decay=1e-4,
                    epochs=epochs,
                    batch_size=batch_size,
                    warmup_epochs=5,
                    patience=10,
                    label_smoothing=0.1,
                    val_fraction=0.2,
                    seed=seed,
                    num_workers=0,
                    backbone_lr_scale=0.1,
                )
                train_result = trainer.fit(
                    train_ds,
                    forward_fn=fwd,
                    model_tag=f"dual_{fusion}_{bshort}_f{fold_counter}",
                )
                test_loader = DataLoader(
                    test_ds, batch_size=batch_size * 2, shuffle=False, num_workers=0
                )
                y_pred, y_prob = trainer.predict(test_loader, forward_fn=fwd)
                m = compute_metrics(y[test_idx], y_pred, y_prob)
                fr = FoldResult(
                    fold=fold_counter,
                    subject=sid,
                    accuracy=m["accuracy"],
                    kappa=m["kappa"],
                    f1_macro=m["f1_macro"],
                    n_train=len(train_idx),
                    n_test=len(test_idx),
                    y_true=y[test_idx],
                    y_pred=y_pred,
                    y_prob=y_prob,
                )
                log.info(
                    "  Fold %d [S%02d]: acc=%.2f%%  kappa=%.3f",
                    fold_counter,
                    sid,
                    fr.accuracy,
                    fr.kappa,
                )
                all_folds.append(fr)
                tag = f"S{sid:02d}_fold{fold_counter:03d}"
                try:
                    save_fold_plots(train_result, y[test_idx], y_pred, plots_dir, tag)
                except Exception as e:
                    log.warning("Plot save failed for %s: %s", tag, e)
                fold_counter += 1

    else:  # loso
        for fold_idx, test_sid in enumerate(subjects):
            train_sids = [s for s in subjects if s != test_sid]
            X_train = np.concatenate([subject_data[s][0] for s in train_sids])
            y_train = np.concatenate([subject_data[s][1] for s in train_sids])
            X_test, y_test = subject_data[test_sid]
            spec_train = np.concatenate([subject_spec_data[s][0] for s in train_sids])
            spec_test, _ = subject_spec_data[test_sid]

            log.info("LOSO fold %d/%d: test=S%02d", fold_idx + 1, len(subjects), test_sid)
            set_seed(seed + fold_idx)

            train_ds_base, test_ds_base, math_input_dim = builder.build_fold(
                X_train, y_train, X_test, y_test
            )
            spec_train_n = normalise_specs(spec_train).astype(np.float32)
            spec_test_n = normalise_specs(spec_test).astype(np.float32)
            _, feats_train, labels_train = [t for t in zip(*train_ds_base.tensors)]
            _, feats_test, labels_test = [t for t in zip(*test_ds_base.tensors)]
            train_ds = TensorDataset(torch.from_numpy(spec_train_n), feats_train, labels_train)
            test_ds = TensorDataset(torch.from_numpy(spec_test_n), feats_test, labels_test)

            model = build_model(math_input_dim)

            def fwd(batch, _m=model):
                imgs, feats, labels = batch
                return _m(imgs.to(_device), feats.to(_device)), labels.to(_device)

            trainer = Trainer(
                model=model,
                device=device,
                learning_rate=1e-4,
                weight_decay=1e-4,
                epochs=epochs,
                batch_size=batch_size,
                warmup_epochs=5,
                patience=10,
                label_smoothing=0.1,
                val_fraction=0.2,
                seed=seed,
                num_workers=0,
                backbone_lr_scale=0.1,
            )
            train_result = trainer.fit(
                train_ds,
                forward_fn=fwd,
                model_tag=f"dual_{fusion}_{bshort}_loso_f{fold_idx}",
            )
            test_loader = DataLoader(
                test_ds, batch_size=batch_size * 2, shuffle=False, num_workers=0
            )
            y_pred, y_prob = trainer.predict(test_loader, forward_fn=fwd)
            m = compute_metrics(y_test, y_pred, y_prob)
            fr = FoldResult(
                fold=fold_idx,
                subject=test_sid,
                accuracy=m["accuracy"],
                kappa=m["kappa"],
                f1_macro=m["f1_macro"],
                n_train=len(y_train),
                n_test=len(y_test),
                y_true=y_test,
                y_pred=y_pred,
                y_prob=y_prob,
            )
            log.info(
                "  LOSO fold %d [S%02d]: acc=%.2f%%  kappa=%.3f",
                fold_idx,
                test_sid,
                fr.accuracy,
                fr.kappa,
            )
            all_folds.append(fr)
            tag = f"S{test_sid:02d}_loso_fold{fold_idx:03d}"
            try:
                save_fold_plots(train_result, y_test, y_pred, plots_dir, tag)
            except Exception as e:
                log.warning("Plot save failed for %s: %s", tag, e)

    elapsed = time.time() - t0
    result = CVResult(strategy=strategy, model_name=MODEL_NAME, folds=all_folds)
    log.info(
        "%s [%s] done in %.1fs: %.2f%% ± %.2f%%",
        fusion,
        strategy,
        elapsed,
        result.mean_accuracy,
        result.std_accuracy,
    )

    # Per-subject summary plot (within-subject only)
    if strategy == "within_subject":
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import seaborn as sns

            per_subj = result.per_subject_accuracy
            sids = sorted(per_subj.keys())
            accs = [per_subj[s] for s in sids]
            mean_a = sum(accs) / len(accs)
            fig, ax = plt.subplots(figsize=(10, 5))
            colors = sns.color_palette("viridis", n_colors=len(sids))
            bars = ax.bar([f"S{s}" for s in sids], accs, color=colors)
            ax.axhline(y=mean_a, color="red", linestyle="--", label=f"Mean: {mean_a:.1f}%")
            for bar, acc in zip(bars, accs):
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() + 0.5,
                    f"{acc:.1f}",
                    ha="center",
                    fontsize=9,
                )
            ax.set_xlabel("Subject")
            ax.set_ylabel("Accuracy (%)")
            ax.set_title(f"Stage 06 [{fusion}] – {MODEL_NAME} per-subject accuracy")
            ax.set_ylim(0, 105)
            ax.legend()
            ax.grid(True, alpha=0.3, axis="y")
            plt.tight_layout()
            plots_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(plots_dir / "per_subject_accuracy.png", dpi=120, bbox_inches="tight")
            plt.close(fig)
        except Exception as e:
            log.warning("Per-subject plot failed: %s", e)

    data = {
        "model": MODEL_NAME,
        "backbone": backbone,
        "fusion": fusion,
        "strategy": strategy,
        "mean_accuracy": result.mean_accuracy,
        "std_accuracy": result.std_accuracy,
        "mean_kappa": result.mean_kappa,
        "mean_f1": result.mean_f1,
        "n_folds": len(all_folds),
        "per_subject": result.per_subject_accuracy,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    log.info("Saved: %s", out_path)
    return out_path


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    backbone = args.backbone
    bshort = _BACKBONE_SHORT.get(backbone, backbone)
    log = setup_stage_logging(run_dir, "stage_06", f"stage_06_dual_branch_{bshort}.log")
    log.info("Backbone: %s  (short: %s)", backbone, bshort)

    checkpoint_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else run_dir / "checkpoints" / "vit_pretrained_physionet_vit.pt"
    )
    processed_dir = Path(args.processed_dir) if args.processed_dir else None

    import numpy as np

    from bci.data.download import (
        load_all_subjects,
        load_spectrogram_stats,
        load_subject_spectrograms,
    )
    from bci.data.download import _processed_dir as _get_processed_dir
    from bci.utils.seed import get_device, set_seed

    device = get_device(args.device)
    log.info("Device: %s", device)
    set_seed(args.seed)

    # ── Load epoch cache ───────────────────────────────────────────────────
    log.info("Loading BCI IV-2a epoch cache...")
    try:
        subject_data, channel_names, sfreq = load_all_subjects("bci_iv2a", data_dir=processed_dir)
    except FileNotFoundError as e:
        log.error("%s  Run Stage 01 first.", e)
        sys.exit(1)
    log.info(
        "Loaded %d subjects (sfreq=%.0f Hz, %d channels)",
        len(subject_data),
        sfreq,
        len(channel_names),
    )

    # ── Load spectrogram cache ─────────────────────────────────────────────
    log.info("Loading BCI IV-2a spectrogram cache...")
    try:
        spec_mean, spec_std = load_spectrogram_stats("bci_iv2a", data_dir=processed_dir)
    except FileNotFoundError as e:
        log.error("%s  Run Stage 01 first.", e)
        sys.exit(1)

    pdir = _get_processed_dir("bci_iv2a", processed_dir)
    spec_files = sorted(pdir.glob("subject_[0-9]*_spectrograms.npz"))
    subject_ids = [int(p.stem.split("_")[1]) for p in spec_files]

    subject_spec_data: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for sid in subject_ids:
        try:
            imgs, y = load_subject_spectrograms("bci_iv2a", sid, data_dir=processed_dir)
            subject_spec_data[sid] = (imgs, y)
        except Exception as e:
            log.warning("  Subject %d spectrograms skipped: %s", sid, e)

    # Only keep subjects that have both epoch and spectrogram data
    common_sids = sorted(set(subject_data.keys()) & set(subject_spec_data.keys()))
    subject_data = {s: subject_data[s] for s in common_sids}
    subject_spec_data = {s: subject_spec_data[s] for s in common_sids}
    log.info("Using %d subjects with both epoch and spectrogram data.", len(common_sids))

    if not common_sids:
        log.error("No subjects with complete data. Exiting.")
        sys.exit(1)

    # ── Run ablation ───────────────────────────────────────────────────────
    for fusion in args.fusions:
        for strategy in args.strategies:
            log.info("=== Fusion: %s | Strategy: %s ===", fusion.upper(), strategy)
            run_dual_branch(
                fusion=fusion,
                strategy=strategy,
                backbone=backbone,
                bshort=bshort,
                subject_data=subject_data,
                subject_spec_data=subject_spec_data,
                spec_mean=spec_mean,
                spec_std=spec_std,
                checkpoint_path=checkpoint_path,
                run_dir=run_dir,
                n_folds=args.n_folds,
                epochs=args.epochs,
                batch_size=args.batch_size,
                device=device,
                seed=args.seed,
                log=log,
            )

    log.info("Stage 06 complete.")


if __name__ == "__main__":
    main()
