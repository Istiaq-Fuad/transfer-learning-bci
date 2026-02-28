"""Stage 07 – Reduced-data transfer learning experiment.

Loads BCI IV-2a epochs and cached 9-channel spectrograms (Stage 01).
Uses PhysioNet-pretrained backbone from Stage 04.

For each training-data fraction (10%, 25%, 50%, 75%, 100%) and for both
'scratch' (ImageNet init) and 'transfer' (PhysioNet-pretrained) conditions,
trains the DualBranchModel with attention fusion multiple times and records
accuracy. This is the core thesis experiment.

Prerequisite: Stage 01 and Stage 04 must have been run first.

Output::

    <run-dir>/results/real_reduced_data_results_vit.json
    <run-dir>/plots/stage_07_vit/  (accuracy-vs-fraction summary plot)

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
    condition: str,
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
    unfreeze_last_n: int = 2,
    seed: int = 0,
) -> float:
    """Run one reduced-data trial and return test accuracy.

    Subsamples *fraction* of the training data using stratified sampling, builds
    the DualBranchModel via :func:`scripts.pipeline.stage_06_dual_branch._build_model`,
    trains for up to *epochs* with early stopping, and returns the test accuracy.

    Parameters
    ----------
    condition:
        ``"scratch"`` or ``"transfer"`` — passed to :func:`_build_model`.
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
    unfreeze_last_n:
        Passed to :func:`_build_model` when backbone is frozen.
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

    from scripts.pipeline.stage_06_dual_branch import _build_model

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

    train_ds, test_ds, math_input_dim = builder.build_fold(X_tr, y_tr, X_test, y_test)
    model = _build_model(
        condition=condition,
        math_input_dim=math_input_dim,
        checkpoint_path=checkpoint_path,
        unfreeze_last_n=unfreeze_last_n,
        fusion="attention",
    )

    def fwd(batch, _m=model):
        imgs, feats, labels = batch
        return _m(imgs.to(_device), feats.to(_device)), labels.to(_device)

    backbone_scale = 0.1 if condition == "transfer" else None
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
    trainer.fit(train_ds, forward_fn=fwd, model_tag=f"{condition}_trial")

    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False, num_workers=0)
    y_pred, y_prob = trainer.predict(test_loader, forward_fn=fwd)
    m = compute_metrics(y_test, y_pred, y_prob)
    return float(m["accuracy"])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 07: Reduced-data transfer learning experiment.",
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
        help="Path to PhysioNet-pretrained backbone checkpoint from Stage 04 "
        "(default: <run-dir>/checkpoints/vit_pretrained_physionet_vit.pt)",
    )
    return p.parse_args()


def save_fraction_summary_plot(
    results: dict,
    fractions: list[float],
    plots_dir: Path,
    backbone: str,
) -> None:
    """Save accuracy-vs-fraction curves for scratch vs. transfer."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plots_dir.mkdir(parents=True, exist_ok=True)
    conditions = list(results.keys())
    colors = {"scratch": "steelblue", "transfer": "darkorange"}
    markers = {"scratch": "o", "transfer": "s"}

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
                label=cond.capitalize(),
                color=colors.get(cond),
                marker=markers.get(cond, "o"),
                capsize=4,
                linewidth=2,
                markersize=7,
            )

    ax.axhline(50, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="Chance (50%)")
    ax.set_xlabel("Training Data Used (%)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Stage 07 – Reduced-data experiment ({backbone})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(plots_dir / "accuracy_vs_fraction.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    backbone = args.backbone
    bshort = _BACKBONE_SHORT.get(backbone, backbone)
    log = setup_stage_logging(run_dir, "stage_07", f"stage_07_reduced_data_{bshort}.log")
    log.info("Backbone: %s  (short: %s)", backbone, bshort)

    out_path = run_dir / "results" / f"real_reduced_data_results_{bshort}.json"
    plots_dir = run_dir / "plots" / f"stage_07_{bshort}"

    if out_path.exists():
        log.info("Result already exists at %s – skipping.", out_path)
        return

    checkpoint_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else run_dir / "checkpoints" / "vit_pretrained_physionet_vit.pt"
    )
    processed_dir = Path(args.processed_dir) if args.processed_dir else None

    import numpy as np
    import torch
    from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
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
    from bci.training.trainer import Trainer
    from bci.utils.config import ModelConfig
    from bci.utils.seed import get_device, set_seed

    device = get_device(args.device)
    log.info("Device: %s", device)
    _device = torch.device(device)
    fused_dim = _FUSED_DIM.get(backbone, 256)
    cls_hidden = _CLASSIFIER_HIDDEN.get(backbone, 128)

    # ── Load epoch cache ───────────────────────────────────────────────────
    log.info("Loading BCI IV-2a epoch cache...")
    try:
        subject_data, channel_names, sfreq = load_all_subjects("bci_iv2a", data_dir=processed_dir)
    except FileNotFoundError as e:
        log.error("%s  Run Stage 01 first.", e)
        sys.exit(1)
    log.info("Loaded %d subjects.", len(subject_data))

    # ── Load spectrogram cache ─────────────────────────────────────────────
    log.info("Loading BCI IV-2a spectrogram cache...")
    try:
        spec_mean, spec_std = load_spectrogram_stats("bci_iv2a", data_dir=processed_dir)
    except FileNotFoundError as e:
        log.error("%s  Run Stage 01 first.", e)
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

    def normalise_specs(imgs: np.ndarray) -> np.ndarray:
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
        model = DualBranchModel(math_input_dim=math_input_dim, config=cfg)
        if condition == "transfer":
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

    conditions = ["scratch", "transfer"]
    results: dict[str, dict] = {c: {} for c in conditions}
    t_total = time.time()

    for fraction in args.fractions:
        frac_str = f"{fraction:.2f}"
        log.info("=== Fraction %.0f%% ===", fraction * 100)
        cond_accs: dict[str, list[float]] = {c: [] for c in conditions}

        for sid in common_sids:
            X, y = subject_data[sid]
            spec_imgs, _ = subject_spec_data[sid]
            skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
            for fold_i, (train_idx, test_idx) in enumerate(skf.split(X, y)):
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

                    for condition in conditions:
                        set_seed(trial_seed)
                        try:
                            # Build handcrafted features from raw EEG
                            train_ds_base, test_ds_base, math_input_dim = builder.build_fold(
                                X_tr, y_tr, X_test, y_test
                            )
                            # Replace image tensors with cached normalised spectrograms
                            spec_tr_n = normalise_specs(spec_tr).astype(np.float32)
                            spec_test_n = normalise_specs(spec_test).astype(np.float32)
                            _, feats_tr, labels_tr = [t for t in zip(*train_ds_base.tensors)]
                            _, feats_te, labels_te = [t for t in zip(*test_ds_base.tensors)]
                            train_ds = TensorDataset(
                                torch.from_numpy(spec_tr_n), feats_tr, labels_tr
                            )
                            test_ds = TensorDataset(
                                torch.from_numpy(spec_test_n), feats_te, labels_te
                            )

                            model = build_model(condition, math_input_dim)

                            def fwd(batch, _m=model):
                                imgs, feats, labels = batch
                                return (
                                    _m(imgs.to(_device), feats.to(_device)),
                                    labels.to(_device),
                                )

                            backbone_scale = 0.1 if condition == "transfer" else None
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
                                backbone_lr_scale=backbone_scale,
                            )
                            trainer.fit(
                                train_ds,
                                forward_fn=fwd,
                                model_tag=f"{condition}_{bshort}_f{fraction:.0%}",
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
                                "Trial failed (S%d fold%d rep%d %s): %s",
                                sid,
                                fold_i,
                                rep,
                                condition,
                                e,
                            )
                            acc = float("nan")

                        cond_accs[condition].append(acc)
                        log.info(
                            "  %s | S%02d fold%d rep%d | frac=%.0f%% | acc=%.2f%%",
                            condition.upper(),
                            sid,
                            fold_i,
                            rep,
                            fraction * 100,
                            acc,
                        )

        for condition in conditions:
            valid_accs = [a for a in cond_accs[condition] if a == a]  # filter NaN
            results[condition][frac_str] = {
                "fraction_pct": round(fraction * 100),
                "mean_accuracy": round(float(np.mean(valid_accs)), 4)
                if valid_accs
                else float("nan"),
                "std_accuracy": round(float(np.std(valid_accs)), 4) if valid_accs else float("nan"),
                "n_runs": len(valid_accs),
            }
            log.info(
                "  %s @ %.0f%%: %.2f%% ± %.2f%% (n=%d)",
                condition.upper(),
                fraction * 100,
                results[condition][frac_str]["mean_accuracy"],
                results[condition][frac_str]["std_accuracy"],
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
    log.info("Saved: %s", out_path)
    log.info("Stage 07 complete.")


if __name__ == "__main__":
    main()
