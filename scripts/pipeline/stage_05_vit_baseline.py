"""Stage 05 – Baseline C: ViT-only classifier on BCI IV-2a spectrograms.

Loads pre-cached 9-channel multichannel CWT spectrograms for BCI IV-2a
subjects (written by Stage 01) and fine-tunes a ViT-Tiny classifier.
Uses PhysioNet-pretrained backbone weights from Stage 04 checkpoint.
Runs 5-fold within-subject CV and LOSO CV.

Prerequisite: Stage 01 and Stage 04 must have been run first.

Output::

    <run-dir>/results/real_baseline_c_vit.json
    <run-dir>/plots/stage_05_vit/  (per-fold training curves + confusion matrices)

Usage::

    uv run python scripts/pipeline/stage_05_vit_baseline.py --run-dir runs/my_run
    uv run python scripts/pipeline/stage_05_vit_baseline.py \\
        --run-dir runs/my_run --epochs 50 --batch-size 32 --device cuda
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from bci.utils.logging import setup_stage_logging
from bci.utils.visualization import save_confusion_matrix, save_per_subject_accuracy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 05: Baseline C – ViT-only on cached spectrograms.",
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
        "--checkpoint",
        default=None,
        help="Path to PhysioNet-pretrained backbone checkpoint from Stage 04 "
        "(default: <run-dir>/checkpoints/vit_pretrained_physionet_vit.pt)",
    )
    args, _ = p.parse_known_args()
    return args


def run_vit_cv(
    strategy: str,
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
    """Run ViT-only CV (within_subject or loso) and save results."""
    import numpy as np
    import torch
    from sklearn.model_selection import StratifiedKFold
    from torch.utils.data import DataLoader, TensorDataset

    from bci.models.vit_branch import ViTBranch
    from bci.training.cross_validation import CVResult, FoldResult
    from bci.training.evaluation import compute_metrics
    from bci.training.trainer import Trainer
    from bci.utils.config import ModelConfig
    from bci.utils.seed import set_seed

    MODEL_NAME = "CWT+ViT-Tiny (PhysioNet pretrained)"
    BACKBONE = "vit_tiny_patch16_224"
    # Resize to 64×64: ViT-Tiny patch16 at 64×64 = 16 patches vs 196 at 224×224.
    # Must match the img_size used in Stage 04 pretraining.
    TARGET_IMG_SIZE = 64

    tag_base = "real_baseline_c_vit"
    if strategy == "loso":
        tag_base += "_loso"
    out_path = run_dir / "results" / f"{tag_base}.json"
    plots_dir = run_dir / "plots" / f"stage_05_vit_{strategy}"

    if out_path.exists():
        log.info("Already exists: %s – skipping.", out_path)
        return out_path

    _device = torch.device(device)

    def build_model():
        cfg = ModelConfig(
            vit_model_name=BACKBONE,
            vit_pretrained=True,
            vit_drop_rate=0.1,
            in_chans=9,
            n_classes=2,
        )
        model = ViTBranch(config=cfg, as_feature_extractor=False, img_size=TARGET_IMG_SIZE)
        if checkpoint_path.exists():
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            # Strip head keys to load only backbone weights
            backbone_state = {
                k: v
                for k, v in ckpt.items()
                if not (k.startswith("head") or k.startswith("classifier"))
            }
            model.backbone.load_state_dict(backbone_state, strict=False)
        else:
            log.warning("Checkpoint not found at %s; using ImageNet init only.", checkpoint_path)
        model.freeze_backbone(unfreeze_last_n_blocks=2)
        return model

    def make_ds(imgs, y):
        # Resize from on-disk 224×224 to TARGET_IMG_SIZE×TARGET_IMG_SIZE.
        # Reduces ViT forward-pass cost by ~12× with minimal accuracy impact.
        if imgs.shape[-1] != TARGET_IMG_SIZE:
            t = torch.from_numpy(imgs.astype(np.float32))
            t = torch.nn.functional.interpolate(
                t, size=(TARGET_IMG_SIZE, TARGET_IMG_SIZE), mode="bilinear", align_corners=False
            )
            imgs = t.numpy()
        # Normalise per-channel using training-set stats
        imgs_norm = (imgs - spec_mean[None, :, None, None]) / spec_std[None, :, None, None]
        return TensorDataset(
            torch.from_numpy(imgs_norm.astype(np.float32)),
            torch.from_numpy(y.astype(np.int64)),
        )

    t0 = time.time()
    all_folds: list[FoldResult] = []
    subjects = sorted(subject_spec_data.keys())

    if strategy == "within_subject":
        fold_counter = 0
        for sid in subjects:
            imgs, y = subject_spec_data[sid]
            log.info("Subject %d (%d trials)...", sid, len(y))
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
            for train_idx, test_idx in skf.split(imgs, y):
                set_seed(seed + fold_counter)
                train_ds = make_ds(imgs[train_idx], y[train_idx])
                test_ds = make_ds(imgs[test_idx], y[test_idx])
                model = build_model()

                def fwd(batch, _m=model):
                    x, labels = batch
                    return _m(x.to(_device)), labels.to(_device)

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
                    train_ds, forward_fn=fwd, model_tag=f"vit_within_f{fold_counter}"
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
                fold_counter += 1

    else:  # loso
        for fold_idx, test_sid in enumerate(subjects):
            train_imgs = np.concatenate(
                [subject_spec_data[s][0] for s in subjects if s != test_sid]
            )
            train_y = np.concatenate([subject_spec_data[s][1] for s in subjects if s != test_sid])
            test_imgs, test_y = subject_spec_data[test_sid]
            log.info("LOSO fold %d/%d: test=S%02d", fold_idx + 1, len(subjects), test_sid)

            set_seed(seed + fold_idx)
            train_ds = make_ds(train_imgs, train_y)
            test_ds = make_ds(test_imgs, test_y)
            model = build_model()

            def fwd(batch, _m=model):
                x, labels = batch
                return _m(x.to(_device)), labels.to(_device)

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
            train_result = trainer.fit(train_ds, forward_fn=fwd, model_tag=f"vit_loso_f{fold_idx}")
            test_loader = DataLoader(
                test_ds, batch_size=batch_size * 2, shuffle=False, num_workers=0
            )
            y_pred, y_prob = trainer.predict(test_loader, forward_fn=fwd)
            m = compute_metrics(test_y, y_pred, y_prob)
            fr = FoldResult(
                fold=fold_idx,
                subject=test_sid,
                accuracy=m["accuracy"],
                kappa=m["kappa"],
                f1_macro=m["f1_macro"],
                n_train=len(train_y),
                n_test=len(test_y),
                y_true=test_y,
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

    elapsed = time.time() - t0
    result = CVResult(strategy=strategy, model_name=MODEL_NAME, folds=all_folds)
    log.info(
        "%s done in %.1fs: %.2f%% ± %.2f%%",
        strategy,
        elapsed,
        result.mean_accuracy,
        result.std_accuracy,
    )

    # ── Summary plots ──────────────────────────────────────────────────────
    strategy_label = "Within-Subject CV" if strategy == "within_subject" else "LOSO CV"
    try:
        import numpy as _np

        agg_y_true = _np.concatenate([f.y_true for f in all_folds])
        agg_y_pred = _np.concatenate([f.y_pred for f in all_folds])
        save_confusion_matrix(
            agg_y_true,
            agg_y_pred,
            plots_dir,
            filename="confusion_matrix",
            title=f"ViT-Only Baseline ({strategy_label})",
        )
    except Exception as e:
        log.warning("Confusion matrix plot failed: %s", e)

    if strategy == "within_subject":
        try:
            save_per_subject_accuracy(
                result.per_subject_accuracy,
                plots_dir,
                filename="per_subject_accuracy",
                title=f"ViT-Only Baseline \u2013 Per-Subject Accuracy ({strategy_label})",
            )
        except Exception as e:
            log.warning("Per-subject plot failed: %s", e)

    # Build output dict keyed for compatibility with phase4_compile_results.py
    data: dict
    if strategy == "within_subject":
        data = {
            "model": MODEL_NAME,
            "backbone": BACKBONE,
            "within_subject": {
                "mean_accuracy": result.mean_accuracy,
                "std_accuracy": result.std_accuracy,
                "mean_kappa": result.mean_kappa,
                "mean_f1": result.mean_f1,
                "n_folds": len(all_folds),
                "per_subject": result.per_subject_accuracy,
            },
        }
    else:
        data = {
            "model": MODEL_NAME,
            "backbone": BACKBONE,
            "strategy": "loso",
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
    log = setup_stage_logging(run_dir, "stage_05", "stage_05_vit_baseline.log")

    checkpoint_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else run_dir / "checkpoints" / "vit_pretrained_physionet_vit.pt"
    )
    processed_dir = Path(args.processed_dir) if args.processed_dir else None

    import numpy as np

    from bci.data.download import load_spectrogram_stats, load_subject_spectrograms
    from bci.data.download import _processed_dir as _get_processed_dir
    from bci.utils.seed import get_device, set_seed

    device = get_device(args.device)
    log.info("Device: %s", device)
    set_seed(args.seed)

    # ── Load spectrogram stats ─────────────────────────────────────────────
    log.info("Loading BCI IV-2a spectrogram stats...")
    try:
        spec_mean, spec_std = load_spectrogram_stats("bci_iv2a", data_dir=processed_dir)
    except FileNotFoundError as e:
        log.error("%s  Run Stage 01 first.", e)
        sys.exit(1)

    # ── Discover available subjects ────────────────────────────────────────
    pdir = _get_processed_dir("bci_iv2a", processed_dir)
    spec_files = sorted(pdir.glob("subject_[0-9]*_spectrograms.npz"))
    if not spec_files:
        log.error("No BCI IV-2a spectrogram files found in %s. Run Stage 01 first.", pdir)
        sys.exit(1)
    subject_ids = [int(p.stem.split("_")[1]) for p in spec_files]
    log.info("Found %d BCI IV-2a subjects.", len(subject_ids))

    # ── Load all spectrogram data ──────────────────────────────────────────
    subject_spec_data: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for sid in subject_ids:
        try:
            imgs, y = load_subject_spectrograms("bci_iv2a", sid, data_dir=processed_dir)
            subject_spec_data[sid] = (imgs, y)
            log.info("  Subject %d: spectrograms=%s", sid, imgs.shape)
        except Exception as e:
            log.warning("  Subject %d skipped: %s", sid, e)

    if not subject_spec_data:
        log.error("No data loaded. Exiting.")
        sys.exit(1)

    log.info("Loaded %d subjects, checkpoint=%s", len(subject_spec_data), checkpoint_path)

    # ── Within-subject CV ──────────────────────────────────────────────────
    log.info("Running ViT-only within-subject %d-fold CV...", args.n_folds)
    run_vit_cv(
        strategy="within_subject",
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

    # ── LOSO CV ───────────────────────────────────────────────────────────
    log.info("Running ViT-only LOSO CV...")
    run_vit_cv(
        strategy="loso",
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

    log.info("Stage 05 complete.")


if __name__ == "__main__":
    main()
