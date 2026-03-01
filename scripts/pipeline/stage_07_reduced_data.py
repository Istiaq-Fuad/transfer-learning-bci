"""Reduced-data transfer learning experiment.

    Loads BCI IV-2a epochs and cached 9-channel spectrograms.
    Uses PhysioNet-pretrained backbone from pretraining.

For each training-data fraction (10%, 25%, 50%, 75%, 100%), trains the
DualBranchModel with attention fusion using the PhysioNet-pretrained
backbone and records accuracy. This is the core thesis experiment.

    Prerequisite: Download and pretraining must have been run first.

Output::

    <run-dir>/results/real_reduced_data_results_vit.json
    <run-dir>/plots/reduced_data_vit/  (accuracy-vs-fraction summary plot)

Usage::

    uv run python scripts/pipeline/stage_07_reduced_data.py --run-dir runs/my_run
    uv run python scripts/pipeline/stage_07_reduced_data.py \\
        --run-dir runs/my_run --fractions 0.10 0.25 0.50 0.75 1.00 \\
        --n-repeats 3 --epochs 50 --batch-size 32 --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import bci.models.vit_branch as _vit_mod
from bci.utils.logging import setup_stage_logging

_BACKBONE_SHORT = {
    _vit_mod.MODEL_NAME: _vit_mod.BACKBONE_SHORT,
}
_FUSED_DIM = {
    _vit_mod.MODEL_NAME: _vit_mod.DEFAULT_FUSED_DIM,
}
_CLASSIFIER_HIDDEN = {
    _vit_mod.MODEL_NAME: _vit_mod.DEFAULT_CLS_HIDDEN,
}


def _run_one_trial(
    X_train_full,
    y_train_full,
    X_test,
    y_test,
    fraction: float,
    builder,
    checkpoint_path: Path | None,
    device: str = "cpu",
    epochs: int = 10,
    lr: float = 1e-4,
    batch_size: int = 16,
    warmup_epochs: int = 2,
    patience: int = 10,
    seed: int = 0,
) -> float:
    """Run one reduced-data trial and return test accuracy.

    Subsamples *fraction* of the training data using stratified sampling, builds
    the DualBranchModel via :func:`scripts.pipeline.stage_06_dual_branch._build_model`,
    trains for up to *epochs* with early stopping, and returns the test accuracy.

    Parameters
    ----------
    X_train_full, y_train_full:
        Full training split (before sub-sampling).
    X_test, y_test:
        Test split.
    fraction:
        Fraction of training data to keep (e.g. ``0.5`` keeps 50 %).
        A value of ``1.0`` uses all training data.
    builder:
        :class:`bci.data.dual_branch_builder.DualBranchFoldBuilder` instance.
    checkpoint_path:
        ViT backbone checkpoint path (required for ``"transfer"`` condition).
    device:
        PyTorch device string.
    epochs, lr, batch_size, warmup_epochs, patience:
        Trainer hyper-parameters.
    seed:
        Random seed for both sub-sampling and training.

    Returns
    -------
    float
        Test accuracy in ``[0, 100]``.
    """
    import numpy as np
    import torch
    from sklearn.model_selection import StratifiedShuffleSplit
    from torch.utils.data import DataLoader

    from bci.training.evaluation import compute_metrics
    from bci.training.trainer import Trainer
    from bci.utils.seed import set_seed

    set_seed(seed)
    _device = torch.device(device)

    # Subsample training data
    if fraction < 1.0:
        n_keep = max(2, int(len(y_train_full) * fraction))
        sss = StratifiedShuffleSplit(n_splits=1, train_size=n_keep, random_state=seed)
        keep_idx, _ = next(sss.split(X_train_full, y_train_full))
        X_tr = X_train_full[keep_idx]
        y_tr = y_train_full[keep_idx]
    else:
        X_tr, y_tr = X_train_full, y_train_full

    if checkpoint_path is None or not Path(checkpoint_path).exists():
        raise FileNotFoundError(
            f"PhysioNet checkpoint required for transfer; got: {checkpoint_path}"
        )

    train_ds, test_ds, math_input_dim = builder.build_fold(X_tr, y_tr, X_test, y_test)

    from scripts.pipeline.stage_06_dual_branch import _build_model

    model = _build_model(
        condition="transfer",
        math_input_dim=math_input_dim,
        checkpoint_path=checkpoint_path,
        unfreeze_last_n=2,
        fusion="attention",
    )

    def fwd(batch, _m=model):
        imgs, feats, labels = batch
        return _m(imgs.to(_device), feats.to(_device)), labels.to(_device)

    backbone_scale = 0.1
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
    trainer.fit(train_ds, forward_fn=fwd, model_tag="transfer_trial")

    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False, num_workers=0)
    y_pred, y_prob = trainer.predict(test_loader, forward_fn=fwd)
    m = compute_metrics(y_test, y_pred, y_prob)
    return float(m["accuracy"])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Reduced-data transfer learning experiment.",
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
    p.add_argument("--n-repeats", type=int, default=3, help="Seed repetitions per fraction")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--fractions",
        nargs="+",
        type=float,
        default=[0.10, 0.25, 0.50, 0.75, 1.00],
    )
    p.add_argument(
        "--backbone",
        default="vit_tiny_patch16_224",
        choices=list(_BACKBONE_SHORT.keys()),
    )
    p.add_argument(
        "--checkpoint",
        default=None,
        help="Path to PhysioNet-pretrained backbone checkpoint from pretraining "
        "(default: <run-dir>/checkpoints/vit_pretrained_physionet_vit.pt)",
    )
    return p.parse_args()


def save_fraction_summary_plot(
    results: dict,
    fractions: list[float],
    plots_dir: Path,
    backbone: str,
) -> None:
    """Save accuracy-vs-fraction curve for transfer dual-branch."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plots_dir.mkdir(parents=True, exist_ok=True)
    conditions = list(results.keys())
    colors = {"transfer": "darkorange"}
    markers = {"transfer": "s"}

    fig, ax = plt.subplots(figsize=(8, 5))
    for cond in conditions:
        cond_data = results[cond]
        xs, ys, errs = [], [], []
        for frac in fractions:
            frac_str = f"{frac:.2f}"
            d = cond_data.get(frac_str, {})
            mean = d.get("mean_accuracy", float("nan"))
            std = d.get("std_accuracy", float("nan"))
            if not (mean != mean):  # not NaN
                xs.append(frac * 100)
                ys.append(mean)
                errs.append(std)
        if xs:
            ax.errorbar(
                xs,
                ys,
                yerr=errs,
                label="Transfer (Dual-Branch)",
                color=colors.get(cond),
                marker=markers.get(cond, "o"),
                capsize=4,
                linewidth=2,
                markersize=7,
            )

    ax.axhline(50, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="Chance (50%)")
    ax.set_xlabel("Training Data Used (%)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Reduced-Data Transfer Learning Experiment")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(plots_dir / "accuracy_vs_fraction.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    backbone = args.backbone
    bshort = str(_BACKBONE_SHORT.get(backbone, backbone) or backbone)
    log = setup_stage_logging(run_dir, "reduced_data", f"reduced_data_{bshort}.log")
    log.info("Backbone: %s  (short: %s)", backbone, bshort)

    out_path = run_dir / "results" / f"real_reduced_data_results_{bshort}.json"
    plots_dir = run_dir / "plots" / f"reduced_data_{bshort}"

    if out_path.exists():
        log.info("Result already exists at %s – skipping.", out_path)
        return

    checkpoint_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else run_dir / "checkpoints" / "vit_pretrained_physionet_vit.pt"
    )
    if not checkpoint_path.exists():
        log.error("PhysioNet checkpoint not found at %s. Run pretraining first.", checkpoint_path)
        sys.exit(1)
    processed_dir = Path(args.processed_dir) if args.processed_dir else None

    import numpy as np
    import torch
    from sklearn.model_selection import StratifiedShuffleSplit
    from torch.utils.data import DataLoader, TensorDataset

    from bci.data.download import (
        load_all_subjects,
        load_spectrogram_stats,
        load_subject_spectrograms,
    )
    from bci.data.download import _processed_dir as _get_processed_dir
    from bci.data.dual_branch_builder import DualBranchFoldBuilder
    from bci.models.dual_branch import DualBranchModel
    from bci.training.evaluation import compute_metrics
    from bci.training.splits import get_or_create_splits
    from bci.training.trainer import Trainer
    from bci.utils.config import ModelConfig
    from bci.utils.seed import get_device, set_seed

    device = get_device(args.device)
    log.info("Device: %s", device)
    _device = torch.device(device)
    fused_dim = _FUSED_DIM.get(backbone, 256)
    cls_hidden = _CLASSIFIER_HIDDEN.get(backbone, 128)
    # Must match pretraining resolution (64×64 = 16 patches, ~12× faster)
    TARGET_IMG_SIZE = 64

    # ── Load epoch cache ───────────────────────────────────────────────────
    log.info("Loading BCI IV-2a epoch cache...")
    try:
        subject_data, channel_names, sfreq = load_all_subjects("bci_iv2a", data_dir=processed_dir)
    except FileNotFoundError as e:
        log.error("%s  Run the download step first.", e)
        sys.exit(1)
    log.info("Loaded %d subjects.", len(subject_data))

    # ── Load spectrogram cache ─────────────────────────────────────────────
    log.info("Loading BCI IV-2a spectrogram cache...")
    try:
        spec_mean, spec_std = load_spectrogram_stats("bci_iv2a", data_dir=processed_dir)
    except FileNotFoundError as e:
        log.error("%s  Run the download step first.", e)
        sys.exit(1)

    pdir = _get_processed_dir("bci_iv2a", processed_dir)
    spec_files = sorted(pdir.glob("subject_[0-9]*_spectrograms.npz"))
    subject_spec_data: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for sf in spec_files:
        sid = int(sf.stem.split("_")[1])
        try:
            imgs, y = load_subject_spectrograms("bci_iv2a", sid, data_dir=processed_dir)
            subject_spec_data[sid] = (imgs, y)
        except Exception as e:
            log.warning("  Subject %d spectrograms skipped: %s", sid, e)

    common_sids = sorted(set(subject_data.keys()) & set(subject_spec_data.keys()))
    subject_data = {s: subject_data[s] for s in common_sids}
    subject_spec_data = {s: subject_spec_data[s] for s in common_sids}
    log.info("Using %d subjects.", len(common_sids))

    if not common_sids:
        log.error("No subjects with complete data. Exiting.")
        sys.exit(1)

    builder = DualBranchFoldBuilder(
        csp_n_components=6,
        riemann_estimator="oas",
        riemann_metric="riemann",
        sfreq=128.0,
    )

    split_spec = get_or_create_splits(
        run_dir=run_dir,
        dataset="bci_iv2a",
        subject_data=subject_data,
        n_folds=args.n_folds,
        seed=args.seed,
    )

    def normalise_specs(imgs: np.ndarray) -> np.ndarray:
        # Resize from on-disk 224×224 to TARGET_IMG_SIZE×TARGET_IMG_SIZE
        if imgs.shape[-1] != TARGET_IMG_SIZE:
            t = torch.from_numpy(imgs.astype(np.float32))
            t = torch.nn.functional.interpolate(
                t, size=(TARGET_IMG_SIZE, TARGET_IMG_SIZE), mode="bilinear", align_corners=False
            )
            imgs = t.numpy()
        return (imgs - spec_mean[None, :, None, None]) / spec_std[None, :, None, None]

    def build_model(condition: str, math_input_dim: int) -> DualBranchModel:
        use_imagenet = True  # always start from ImageNet weights
        cfg = ModelConfig(
            vit_model_name=backbone,
            vit_pretrained=use_imagenet,
            vit_drop_rate=0.1,
            in_chans=9,
            csp_n_components=6,
            math_hidden_dims=[256, 128],
            math_drop_rate=0.3,
            fusion_method="attention",
            fused_dim=fused_dim,
            classifier_hidden_dim=cls_hidden,
            n_classes=2,
        )
        model = DualBranchModel(math_input_dim=math_input_dim, config=cfg, img_size=TARGET_IMG_SIZE)
        if checkpoint_path.exists():
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            backbone_state = {
                k: v
                for k, v in ckpt.items()
                if not (k.startswith("head") or k.startswith("classifier"))
            }
            model.vit_branch.backbone.load_state_dict(backbone_state, strict=False)
        else:
            log.warning(
                "Checkpoint not found at %s; transfer condition uses ImageNet init only.",
                checkpoint_path,
            )
        model.freeze_vit_backbone(unfreeze_last_n_blocks=2)
        return model

    results: dict[str, dict] = {"transfer": {}}
    t_total = time.time()

    for fraction in args.fractions:
        frac_str = f"{fraction:.2f}"
        log.info("=== Fraction %.0f%% ===", fraction * 100)
        cond_accs: dict[str, list[float]] = {"transfer": []}

        for sid in common_sids:
            X, y = subject_data[sid]
            spec_imgs, _ = subject_spec_data[sid]
            folds = split_spec.within_subject.get(sid, [])
            for fold_i, fold in enumerate(folds):
                train_idx = np.array(fold["train_idx"], dtype=int)
                test_idx = np.array(fold["test_idx"], dtype=int)
                X_train_full = X[train_idx]
                y_train_full = y[train_idx]
                spec_train_full = spec_imgs[train_idx]
                X_test, y_test = X[test_idx], y[test_idx]
                spec_test = spec_imgs[test_idx]

                for rep in range(args.n_repeats):
                    trial_seed = args.seed + sid * 1000 + fold_i * 100 + rep

                    if fraction < 1.0:
                        n_keep = max(2, int(len(y_train_full) * fraction))
                        sss = StratifiedShuffleSplit(
                            n_splits=1, train_size=n_keep, random_state=trial_seed
                        )
                        keep_idx, _ = next(sss.split(X_train_full, y_train_full))
                        X_tr = X_train_full[keep_idx]
                        y_tr = y_train_full[keep_idx]
                        spec_tr = spec_train_full[keep_idx]
                    else:
                        X_tr, y_tr, spec_tr = X_train_full, y_train_full, spec_train_full

                    set_seed(trial_seed)
                    try:
                        features_train, features_test, math_input_dim = builder.build_math_features(
                            X_tr,
                            y_tr,
                            X_test,
                            y_test,
                            cache_path=run_dir
                            / "cache"
                            / "math_features"
                            / f"reduced_data_transfer_s{sid:02d}_f{fold_i:02d}_r{rep:02d}_frac{fraction:.2f}.npz",
                        )
                        # Replace image tensors with cached normalised spectrograms
                        spec_tr_n = normalise_specs(spec_tr).astype(np.float32)
                        spec_test_n = normalise_specs(spec_test).astype(np.float32)
                        train_ds = TensorDataset(
                            torch.from_numpy(spec_tr_n),
                            torch.from_numpy(features_train),
                            torch.from_numpy(y_tr.astype(np.int64)),
                        )
                        test_ds = TensorDataset(
                            torch.from_numpy(spec_test_n),
                            torch.from_numpy(features_test),
                            torch.from_numpy(y_test.astype(np.int64)),
                        )

                        model = build_model("transfer", math_input_dim)

                        def fwd(batch, _m=model):
                            imgs, feats, labels = batch
                            return (
                                _m(imgs.to(_device), feats.to(_device)),
                                labels.to(_device),
                            )

                        trainer = Trainer(
                            model=model,
                            device=device,
                            learning_rate=1e-4,
                            weight_decay=1e-4,
                            epochs=args.epochs,
                            batch_size=args.batch_size,
                            warmup_epochs=3,
                            patience=8,
                            label_smoothing=0.1,
                            val_fraction=0.2,
                            seed=trial_seed,
                            num_workers=0,
                            backbone_lr_scale=0.1,
                        )
                        trainer.fit(
                            train_ds,
                            forward_fn=fwd,
                            model_tag=f"transfer_{bshort}_f{fraction:.0%}",
                        )
                        test_loader = DataLoader(
                            test_ds,
                            batch_size=args.batch_size * 2,
                            shuffle=False,
                            num_workers=0,
                        )
                        y_pred, y_prob = trainer.predict(test_loader, forward_fn=fwd)
                        m = compute_metrics(y_test, y_pred, y_prob)
                        acc = m["accuracy"]
                    except Exception as e:
                        log.warning(
                            "Trial failed (S%d fold%d rep%d transfer): %s",
                            sid,
                            fold_i,
                            rep,
                            e,
                        )
                        acc = float("nan")

                    cond_accs["transfer"].append(acc)
                    log.info(
                        "  TRANSFER | S%02d fold%d rep%d | frac=%.0f%% | acc=%.2f%%",
                        sid,
                        fold_i,
                        rep,
                        fraction * 100,
                        acc,
                    )

        valid_accs = [a for a in cond_accs["transfer"] if a == a]
        results["transfer"][frac_str] = {
            "fraction_pct": round(fraction * 100),
            "mean_accuracy": round(float(np.mean(valid_accs)), 4) if valid_accs else float("nan"),
            "std_accuracy": round(float(np.std(valid_accs)), 4) if valid_accs else float("nan"),
            "n_runs": len(valid_accs),
        }
        log.info(
            "  TRANSFER @ %.0f%%: %.2f%% ± %.2f%% (n=%d)",
            fraction * 100,
            results["transfer"][frac_str]["mean_accuracy"],
            results["transfer"][frac_str]["std_accuracy"],
            len(valid_accs),
        )

    elapsed = time.time() - t_total
    log.info("Reduced-data experiment done in %.1fs (%.1f min)", elapsed, elapsed / 60)

    # ── Summary plot ───────────────────────────────────────────────────────
    try:
        save_fraction_summary_plot(results, args.fractions, plots_dir, backbone)
        log.info("Summary plot saved: %s", plots_dir / "accuracy_vs_fraction.png")
    except Exception as e:
        log.warning("Fraction summary plot failed: %s", e)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "backbone": backbone,
                "fractions": args.fractions,
                "results": results,
            },
            f,
            indent=2,
        )
    try:
        from bci.utils.results_index import update_results_index, write_manifest

        outputs = {"reduced_data": str(out_path)}
        update_results_index(run_dir, "reduced_data", outputs)
        write_manifest(
            run_dir,
            "reduced_data",
            outputs,
            meta={"fractions": args.fractions, "n_repeats": args.n_repeats},
        )
    except Exception as e:
        log.warning("Failed to update results index: %s", e)
    log.info("Saved: %s", out_path)
    log.info("Reduced-data experiment complete.")


if __name__ == "__main__":
    main()
