"""Stage 10 – Reduced-data transfer learning experiment.

The core thesis experiment. For each training-data fraction
(10%, 25%, 50%, 75%, 100%) and for both 'scratch' and 'transfer' conditions,
trains the DualBranchModel multiple times and records accuracy.

Requires the Stage 8 checkpoint for the 'transfer' condition.

Output: <run-dir>/results/real_reduced_data_results.json

Usage::

    uv run python scripts/pipeline/stage_10_reduced_data.py --run-dir runs/my_run
    uv run python scripts/pipeline/stage_10_reduced_data.py --run-dir runs/my_run \\
        --checkpoint runs/my_run/checkpoints/vit_pretrained_physionet.pt \\
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 10: Reduced-data transfer learning experiment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--run-dir",    required=True)
    p.add_argument("--data-dir",   default="~/mne_data")
    p.add_argument("--device",     default="auto")
    p.add_argument("--epochs",     type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--n-folds",    type=int, default=5)
    p.add_argument("--n-repeats",  type=int, default=3,
                   help="Random-seed repetitions per fraction")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument(
        "--fractions", nargs="+", type=float,
        default=[0.10, 0.25, 0.50, 0.75, 1.00],
    )
    p.add_argument(
        "--checkpoint", default=None,
        help="Path to ViT checkpoint from Stage 8 "
             "(default: <run-dir>/checkpoints/vit_pretrained_physionet.pt)",
    )
    return p.parse_args()


def setup_logging(run_dir: Path) -> logging.Logger:
    fmt = "%(asctime)s  %(levelname)-8s  %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt, datefmt="%H:%M:%S",
                        stream=sys.stdout)
    for lib in ("mne", "pyriemann", "timm", "matplotlib", "moabb"):
        logging.getLogger(lib).setLevel(logging.ERROR)
    log = logging.getLogger("stage_10")
    run_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(run_dir / "stage_10_reduced_data.log")
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


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    log = setup_logging(run_dir)

    out_path = run_dir / "results" / "real_reduced_data_results.json"
    if out_path.exists():
        log.info("Result already exists at %s – skipping.", out_path)
        return

    checkpoint_path = (
        Path(args.checkpoint) if args.checkpoint
        else run_dir / "checkpoints" / "vit_pretrained_physionet.pt"
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

    device  = get_device(args.device)
    log.info("Device: %s", device)
    _device = torch.device(device)

    subject_data = load_bci_iv2a(args.data_dir, log)
    if not subject_data:
        log.error("No data loaded. Exiting.")
        sys.exit(1)

    builder = DualBranchFoldBuilder(
        csp_n_components=6, csp_reg="ledoit_wolf",
        riemann_estimator="lwf", riemann_metric="riemann",
        sfreq=128.0, channel_names=["C3", "Cz", "C4"],
    )

    def build_model(condition, math_input_dim):
        use_imagenet = condition in ("imagenet", "transfer")
        cfg = ModelConfig(
            vit_model_name="vit_tiny_patch16_224",
            vit_pretrained=use_imagenet,
            vit_drop_rate=0.1, csp_n_components=6,
            math_hidden_dims=[256, 128], math_drop_rate=0.3,
            fusion_method="attention", fused_dim=128,
            classifier_hidden_dim=64, n_classes=2,
        )
        model = DualBranchModel(math_input_dim=math_input_dim, config=cfg)
        if condition == "transfer":
            if checkpoint_path.exists():
                ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                backbone_state = {k: v for k, v in ckpt.items()
                                  if not k.startswith("backbone.head")}
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
            skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True,
                                  random_state=args.seed)
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
                                n_splits=1, train_size=n_keep,
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
                                return (_m(imgs.to(_device), feats.to(_device)),
                                        labels.to(_device))

                            trainer = Trainer(
                                model=model, device=device,
                                learning_rate=1e-4, weight_decay=1e-4,
                                epochs=args.epochs, batch_size=args.batch_size,
                                warmup_epochs=3, patience=8,
                                label_smoothing=0.1, val_fraction=0.2,
                                seed=trial_seed, num_workers=0,
                            )
                            trainer.fit(train_ds, forward_fn=fwd,
                                        model_tag=f"{condition}_f{fraction:.0%}")
                            test_loader = DataLoader(
                                test_ds, batch_size=args.batch_size * 2,
                                shuffle=False, num_workers=0,
                            )
                            y_pred, y_prob = trainer.predict(test_loader, forward_fn=fwd)
                            m = compute_metrics(y_test, y_pred, y_prob)
                            acc = m["accuracy"]
                        except Exception as e:
                            log.warning("Trial failed (S%d fold%d rep%d %s): %s",
                                        sid, fold_i, rep, condition, e)
                            acc = float("nan")

                        cond_accs[condition].append(acc)
                        log.info(
                            "  %s | S%02d fold%d rep%d | frac=%.0f%% | acc=%.2f%%",
                            condition.upper(), sid, fold_i, rep, fraction * 100, acc,
                        )

        for condition in conditions:
            accs = [a for a in cond_accs[condition] if not np.isnan(a)]
            results[condition][frac_str] = {
                "fraction": fraction,
                "mean": float(np.mean(accs)) if accs else float("nan"),
                "std":  float(np.std(accs))  if accs else float("nan"),
                "n_runs": len(accs),
                "runs": accs,
            }
            log.info("  %s @ %.0f%%: %.2f%% ± %.2f%% (n=%d)",
                     condition.upper(), fraction * 100,
                     results[condition][frac_str]["mean"],
                     results[condition][frac_str]["std"],
                     len(accs))

    elapsed = time.time() - t_total
    log.info("Reduced-data experiment done in %.1fs (%.1f min)", elapsed, elapsed / 60)

    # Reformat to match phase4_compile_results.py expectations
    summary: dict[str, dict] = {}
    for cond, frac_data in results.items():
        summary[cond] = {
            fs: {
                "fraction_pct":  round(d["fraction"] * 100),
                "mean_accuracy": round(d["mean"], 4),
                "std_accuracy":  round(d["std"],  4),
                "n_runs":        d["n_runs"],
            }
            for fs, d in frac_data.items()
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"fractions": args.fractions, "results": summary}, f, indent=2)
    log.info("Saved: %s", out_path)
    log.info("Stage 10 complete.")


if __name__ == "__main__":
    main()
