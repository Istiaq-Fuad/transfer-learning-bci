"""PyTorch training loop for MI-EEG classifiers.

Provides a generic Trainer class with:
    - Training + validation loop with early stopping
    - AdamW optimizer with cosine LR schedule and linear warmup
    - Label smoothing cross-entropy loss
    - Best-model checkpointing
    - Per-epoch metric logging

Used by Baseline C (ViT standalone) and the full dual-branch model (Phase 2).
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, cohen_kappa_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, random_split

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Training result container
# ---------------------------------------------------------------------------

@dataclass
class EpochResult:
    epoch: int
    train_loss: float
    val_loss: float
    val_accuracy: float
    val_kappa: float
    lr: float


@dataclass
class TrainResult:
    """Outcome of a full training run."""
    best_val_accuracy: float
    best_epoch: int
    final_epoch: int
    max_epochs: int
    history: list[EpochResult] = field(repr=False)
    best_checkpoint_path: str | None = None

    @property
    def stopped_early(self) -> bool:
        """True if training stopped before reaching max_epochs."""
        return self.final_epoch < self.max_epochs


# ---------------------------------------------------------------------------
# Learning-rate schedule helpers
# ---------------------------------------------------------------------------

def _cosine_with_warmup_schedule(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    total_epochs: int,
) -> LambdaLR:
    """Linear warmup then cosine annealing LR schedule."""
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / max(warmup_epochs, 1)
        progress = float(epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Core Trainer class
# ---------------------------------------------------------------------------

class Trainer:
    """Generic PyTorch trainer.

    Args:
        model: nn.Module to train.
        device: Torch device string.
        learning_rate: Peak learning rate.
        weight_decay: L2 regularization.
        epochs: Maximum number of epochs.
        batch_size: Training batch size.
        warmup_epochs: Epochs for linear LR warmup.
        patience: Early-stopping patience (epochs without improvement).
        min_delta: Minimum improvement to reset patience counter.
        label_smoothing: Label smoothing for CrossEntropyLoss.
        val_fraction: Fraction of training data to use for validation.
        checkpoint_dir: Where to save the best model checkpoint.
        seed: Random seed for the train/val split.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str | torch.device = "cpu",
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        epochs: int = 100,
        batch_size: int = 32,
        warmup_epochs: int = 5,
        patience: int = 15,
        min_delta: float = 1e-4,
        label_smoothing: float = 0.1,
        val_fraction: float = 0.2,
        checkpoint_dir: str | Path | None = None,
        seed: int = 42,
        num_workers: int = 0,
    ) -> None:
        self.model = model
        self.device = torch.device(device)
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.warmup_epochs = warmup_epochs
        self.patience = patience
        self.min_delta = min_delta
        self.label_smoothing = label_smoothing
        self.val_fraction = val_fraction
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.seed = seed
        self.num_workers = num_workers

        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def _make_optimizer_and_scheduler(self, total_epochs: int):
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = _cosine_with_warmup_schedule(optimizer, self.warmup_epochs, total_epochs)
        return optimizer, scheduler

    def _split_dataset(self, dataset: Dataset) -> tuple[DataLoader, DataLoader]:
        """Split dataset into train and validation DataLoaders."""
        n = len(dataset)  # type: ignore[arg-type]
        n_val = max(1, int(n * self.val_fraction))
        n_train = n - n_val
        generator = torch.Generator().manual_seed(self.seed)
        train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=generator)
        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=n_train > self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.device.type == "cuda",
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.batch_size * 2,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.device.type == "cuda",
        )
        return train_loader, val_loader

    def _train_one_epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        forward_fn: Callable,
    ) -> float:
        """Run one training epoch and return mean loss."""
        self.model.train()
        total_loss = 0.0
        n = 0
        for batch in loader:
            optimizer.zero_grad()
            logits, labels = forward_fn(batch)
            loss = self.criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * len(labels)
            n += len(labels)
        return total_loss / max(n, 1)

    @torch.no_grad()
    def _evaluate(
        self,
        loader: DataLoader,
        forward_fn: Callable,
    ) -> tuple[float, float, float]:
        """Evaluate model on a DataLoader.

        Returns:
            (loss, accuracy_percent, kappa)
        """
        self.model.eval()
        total_loss = 0.0
        n = 0
        all_preds: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []

        for batch in loader:
            logits, labels = forward_fn(batch)
            loss = self.criterion(logits, labels)
            total_loss += loss.item() * len(labels)
            n += len(labels)

            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())

        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_labels)

        acc = accuracy_score(y_true, y_pred) * 100
        kappa = cohen_kappa_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.0
        return total_loss / max(n, 1), acc, kappa

    def fit(
        self,
        dataset: Dataset,
        forward_fn: Callable | None = None,
        model_tag: str = "model",
    ) -> TrainResult:
        """Train the model on a dataset.

        Args:
            dataset: A PyTorch Dataset whose items are batches.
                For single-input models: (input, label)
                For multi-input: (input1, input2, ..., label)
            forward_fn: Function that takes a batch and returns (logits, labels).
                If None, assumes (x, y) batches with model(x).
            model_tag: String tag for checkpoint filenames.

        Returns:
            TrainResult with training history.
        """
        if forward_fn is None:
            def forward_fn(batch):
                inputs, labels = batch
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                return self.model(inputs), labels

        train_loader, val_loader = self._split_dataset(dataset)
        optimizer, scheduler = self._make_optimizer_and_scheduler(self.epochs)

        best_val_acc = 0.0
        best_epoch = 0
        best_ckpt_path: str | None = None
        patience_counter = 0
        history: list[EpochResult] = []

        logger.info(
            "Training %s | device=%s | epochs=%d | lr=%.2e | batch=%d | "
            "train=%d val=%d",
            model_tag, self.device, self.epochs, self.lr, self.batch_size,
            len(train_loader.dataset),  # type: ignore[arg-type]
            len(val_loader.dataset),  # type: ignore[arg-type]
        )

        t_start = time.time()
        for epoch in range(1, self.epochs + 1):
            train_loss = self._train_one_epoch(train_loader, optimizer, forward_fn)
            val_loss, val_acc, val_kappa = self._evaluate(val_loader, forward_fn)
            current_lr = optimizer.param_groups[0]["lr"]
            scheduler.step()

            result = EpochResult(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                val_accuracy=val_acc,
                val_kappa=val_kappa,
                lr=current_lr,
            )
            history.append(result)

            if epoch % max(1, self.epochs // 10) == 0 or epoch <= 5:
                logger.info(
                    "  [%03d/%03d] train_loss=%.4f  val_loss=%.4f  "
                    "val_acc=%.2f%%  kappa=%.3f  lr=%.2e",
                    epoch, self.epochs,
                    train_loss, val_loss, val_acc, val_kappa, current_lr,
                )

            # Early stopping + checkpointing
            if val_acc > best_val_acc + self.min_delta:
                best_val_acc = val_acc
                best_epoch = epoch
                patience_counter = 0

                if self.checkpoint_dir is not None:
                    self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    ckpt_path = self.checkpoint_dir / f"{model_tag}_best.pt"
                    torch.save(self.model.state_dict(), ckpt_path)
                    best_ckpt_path = str(ckpt_path)
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info(
                        "  Early stopping at epoch %d (best=%.2f%% @ epoch %d)",
                        epoch, best_val_acc, best_epoch,
                    )
                    break

        elapsed = time.time() - t_start
        logger.info(
            "Training done in %.1fs | best_val_acc=%.2f%% @ epoch %d",
            elapsed, best_val_acc, best_epoch,
        )

        return TrainResult(
            best_val_accuracy=best_val_acc,
            best_epoch=best_epoch,
            final_epoch=history[-1].epoch,
            max_epochs=self.epochs,
            history=history,
            best_checkpoint_path=best_ckpt_path,
        )

    @torch.no_grad()
    def predict(
        self,
        loader: DataLoader,
        forward_fn: Callable | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run inference on a DataLoader.

        Args:
            loader: DataLoader of (inputs, labels) or multi-input batches.
            forward_fn: Same signature as in fit(). If None, uses (x,y) convention.

        Returns:
            (y_pred, y_prob) numpy arrays.
        """
        if forward_fn is None:
            def forward_fn(batch):
                inputs, labels = batch
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                return self.model(inputs), labels

        self.model.eval()
        all_probs: list[np.ndarray] = []
        all_preds: list[np.ndarray] = []

        for batch in loader:
            logits, _ = forward_fn(batch)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_probs.append(probs)
            all_preds.append(preds)

        return np.concatenate(all_preds), np.concatenate(all_probs)


def main() -> None:
    """CLI entry point — delegates to individual training scripts."""
    import argparse

    parser = argparse.ArgumentParser(description="Train MI-EEG classifier")
    parser.add_argument("--config", type=str, default=None, help="Config YAML path")
    parser.add_argument(
        "--baseline",
        choices=["a", "b", "c", "dual"],
        default=None,
        help="Which baseline/model to run",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.baseline == "a":
        from scripts.baseline_a_csp_lda import main as run_a  # type: ignore[import]
        run_a()
    elif args.baseline == "b":
        from scripts.baseline_b_riemannian import main as run_b  # type: ignore[import]
        run_b()
    elif args.baseline == "c":
        from scripts.baseline_c_vit import main as run_c  # type: ignore[import]
        run_c()
    else:
        logger.info("Use --baseline {a,b,c,dual} to select a training pipeline.")
        logger.info("Or run the baseline scripts directly from scripts/")


if __name__ == "__main__":
    main()
