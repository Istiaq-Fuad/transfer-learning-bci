"""Stage 9 – Reduced-data transfer learning experiment.

The core thesis experiment. For each training-data fraction
(10%, 25%, 50%, 75%, 100%) and for both 'scratch' and 'transfer' conditions,
trains the DualBranchModel multiple times and records accuracy.

Requires the Stage 7 checkpoint for the 'transfer' condition.

Output:
  <run-dir>/results/real_reduced_data_results_vit.json
  <run-dir>/plots/stage_09_vit/  (accuracy-vs-fraction summary plot)

Usage::

    uv run python scripts/pipeline/stage_09_reduced_data.py --run-dir runs/my_run
    uv run python scripts/pipeline/stage_09_reduced_data.py --run-dir runs/my_run \\
        --checkpoint runs/my_run/checkpoints/vit_pretrained_physionet_vit.pt \\
        --fractions 0.10 0.25 0.50 0.75 1.00 \\
        --n-repeats 3 --n-folds 5 --epochs 50 --batch-size 32 --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

# Canonical per-backbone constants – single source of truth lives in the library.
import bci.models.vit_branch as _vit_mod

_BACKBONE_SHORT = {
    _vit_mod.MODEL_NAME: _vit_mod.BACKBONE_SHORT,
}
_FUSED_DIM = {
    _vit_mod.MODEL_NAME: _vit_mod.DEFAULT_FUSED_DIM,
}
_CLASSIFIER_HIDDEN = {
    _vit_mod.MODEL_NAME: _vit_mod.DEFAULT_CLS_HIDDEN,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 9: Reduced-data transfer learning experiment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--run-dir", required=True)
    p.add_argument("--data-dir", default="~/mne_data")
    p.add_argument("--device", default="auto")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--n-repeats", type=int, default=3, help="Random-seed repetitions per fraction")
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
        help="timm backbone model name",
    )
    p.add_argument(
        "--checkpoint",
        default=None,
        help="Path to backbone checkpoint from Stage 7 "
        "(default: <run-dir>/checkpoints/vit_pretrained_physionet_<backbone_short>.pt)",
    )
    p.add_argument(
        "--data",
        default="real",
        choices=["real", "synthetic"],
        help="'real' loads BCI IV-2a; 'synthetic' uses generated data (fast, for smoke tests)",
    )
    return p.parse_args()


def setup_logging(run_dir: Path, log_name: str) -> logging.Logger:
    fmt = "%(asctime)s  %(levelname)-8s  %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt, datefmt="%H:%M:%S", stream=sys.stdout)
    for lib in ("mne", "pyriemann", "timm", "matplotlib", "moabb"):
        logging.getLogger(lib).setLevel(logging.ERROR)
    log = logging.getLogger("stage_09")
    run_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(run_dir / log_name)
    fh.setFormatter(logging.Formatter(fmt, "%H:%M:%S"))
    log.addHandler(fh)
    return log


def load_bci_iv2a(data_dir: str, log) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    import mne
    from moabb.datasets import BNCI2014_001
    from moabb.paradigms import LeftRightImagery

    mne.set_log_level("ERROR")
    log.info("Loading BCI IV-2a (sfreq=128 Hz)...")
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
            log.info("  Subject %d: X=%s", sid, X.shape)
        except Exception as e:
            log.warning("  Subject %d skipped: %s", sid, e)
    log.info("Loaded %d subjects.", len(subject_data))
    return subject_data


def save_fraction_summary_plot(
    results: dict[str, dict],
    fractions: list[float],
    plots_dir: Path,
    backbone: str,
    bshort: str,
) -> None:
    """Save accuracy-vs-fraction curves for scratch vs. transfer."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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
            if frac_str in cond_data and not np.isnan(cond_data[frac_str]["mean"]):
                xs.append(frac * 100)
                ys.append(cond_data[frac_str]["mean"])
                errs.append(cond_data[frac_str]["std"])
        if xs:
            ax.errorbar(
                xs,
                ys,
                yerr=errs,
                label=cond.capitalize(),
                color=colors.get(cond, None),
                marker=markers.get(cond, "o"),
                capsize=4,
                linewidth=2,
                markersize=7,
            )

    ax.set_xlabel("Training data fraction (%)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Stage 9 – Reduced-data experiment ({backbone})")
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
    log = setup_logging(run_dir, f"stage_09_reduced_data_{bshort}.log")
    log.info("Backbone: %s  (short: %s)", backbone, bshort)

    out_path = run_dir / "results" / f"real_reduced_data_results_{bshort}.json"
    plots_dir = run_dir / "plots" / f"stage_09_{bshort}"

    if out_path.exists():
        log.info("Result already exists at %s – skipping.", out_path)
        return

    # Default checkpoint uses backbone-specific name from Stage 7
    checkpoint_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else run_dir / "checkpoints" / f"vit_pretrained_physionet_{bshort}.pt"
    )

    import torch
    from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
    from torch.utils.data import DataLoader

    from bci.data.dual_branch_builder import DualBranchFoldBuilder
    from bci.models.dual_branch import DualBranchModel
    from bci.training.evaluation import compute_metrics
    from bci.training.trainer import Trainer
    from bci.utils.config import ModelConfig
    from bci.utils.seed import get_device, set_seed

    fused_dim = _FUSED_DIM.get(backbone, 256)
    cls_hidden = _CLASSIFIER_HIDDEN.get(backbone, 128)

    device = get_device(args.device)
    log.info("Device: %s", device)
    _device = torch.device(device)

    if args.data == "synthetic":
        from bci.training.cross_validation import make_synthetic_subject_data

        log.info("Using synthetic data (smoke-test mode).")
        subject_data = make_synthetic_subject_data(n_subjects=3, seed=args.seed)
    else:
        subject_data = load_bci_iv2a(args.data_dir, log)
    if not subject_data:
        log.error("No data loaded. Exiting.")
        sys.exit(1)

    builder = DualBranchFoldBuilder(
        csp_n_components=6,
        csp_reg="ledoit_wolf",
        riemann_estimator="lwf",
        riemann_metric="riemann",
        sfreq=128.0,
        channel_names=["C3", "Cz", "C4"],
    )

    def build_model(condition, math_input_dim):
        use_imagenet = condition in ("imagenet", "transfer")
        cfg = ModelConfig(
            vit_model_name=backbone,
            vit_pretrained=use_imagenet,
            vit_drop_rate=0.1,
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
                # Strip classifier head keys (works for both 'head' and 'classifier')
                backbone_state = {
                    k: v
                    for k, v in ckpt.items()
                    if not (k.startswith("backbone.head") or k.startswith("backbone.classifier"))
                }
                model.vit_branch.backbone.load_state_dict(backbone_state, strict=False)
            model.freeze_vit_backbone(unfreeze_last_n_blocks=2)
        return model

    conditions = ["scratch", "transfer"]
    results: dict[str, dict] = {c: {} for c in conditions}
    t_total = time.time()

    for fraction in args.fractions:
        frac_str = f"{fraction:.2f}"
        log.info("=== Fraction %.0f%% ===", fraction * 100)
        cond_accs: dict[str, list[float]] = {c: [] for c in conditions}

        for sid, (X, y) in sorted(subject_data.items()):
            skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
            for fold_i, (train_idx, test_idx) in enumerate(skf.split(X, y)):
                X_train_full = X[train_idx]
                y_train_full = y[train_idx]
                X_test, y_test = X[test_idx], y[test_idx]

                for rep in range(args.n_repeats):
                    trial_seed = args.seed + sid * 1000 + fold_i * 100 + rep

                    for condition in conditions:
                        set_seed(trial_seed)

                        if fraction < 1.0:
                            n_keep = max(2, int(len(y_train_full) * fraction))
                            sss = StratifiedShuffleSplit(
                                n_splits=1,
                                train_size=n_keep,
                                random_state=trial_seed,
                            )
                            keep_idx, _ = next(sss.split(X_train_full, y_train_full))
                            X_tr = X_train_full[keep_idx]
                            y_tr = y_train_full[keep_idx]
                        else:
                            X_tr, y_tr = X_train_full, y_train_full

                        try:
                            train_ds, test_ds, math_input_dim = builder.build_fold(
                                X_tr, y_tr, X_test, y_test
                            )
                            model = build_model(condition, math_input_dim)

                            def fwd(batch, _m=model):
                                imgs, feats, labels = batch
                                return (_m(imgs.to(_device), feats.to(_device)), labels.to(_device))

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
                                backbone_lr_scale=0.1 if condition == "transfer" else None,
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
            accs = [a for a in cond_accs[condition] if not np.isnan(a)]
            results[condition][frac_str] = {
                "fraction": fraction,
                "mean": float(np.mean(accs)) if accs else float("nan"),
                "std": float(np.std(accs)) if accs else float("nan"),
                "n_runs": len(accs),
                "runs": accs,
            }
            log.info(
                "  %s @ %.0f%%: %.2f%% ± %.2f%% (n=%d)",
                condition.upper(),
                fraction * 100,
                results[condition][frac_str]["mean"],
                results[condition][frac_str]["std"],
                len(accs),
            )

    elapsed = time.time() - t_total
    log.info("Reduced-data experiment done in %.1fs (%.1f min)", elapsed, elapsed / 60)

    # Save accuracy-vs-fraction summary plot
    try:
        save_fraction_summary_plot(results, args.fractions, plots_dir, backbone, bshort)
        log.info("Summary plot saved: %s", plots_dir / "accuracy_vs_fraction.png")
    except Exception as e:
        log.warning("Fraction summary plot failed: %s", e)

    # Reformat to match phase4_compile_results.py expectations
    summary: dict[str, dict] = {}
    for cond, frac_data in results.items():
        summary[cond] = {
            fs: {
                "fraction_pct": round(d["fraction"] * 100),
                "mean_accuracy": round(d["mean"], 4),
                "std_accuracy": round(d["std"], 4),
                "n_runs": d["n_runs"],
            }
            for fs, d in frac_data.items()
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "backbone": backbone,
                "fractions": args.fractions,
                "results": summary,
            },
            f,
            indent=2,
        )
    log.info("Saved: %s", out_path)
    log.info("Stage 9 complete.")


if __name__ == "__main__":
    main()
