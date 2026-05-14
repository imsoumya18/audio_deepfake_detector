import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        criterion: nn.Module,
        device: torch.device,
        checkpoint_dir: str,
        log_dir: str,
        patience: int = 10,
    ):
        self.model          = model.to(device)
        self.optimizer      = optimizer
        self.scheduler      = scheduler
        self.criterion      = criterion
        self.device         = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.patience       = patience
        self.writer         = SummaryWriter(log_dir=log_dir)

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.best_eer     = float("inf")
        self.epochs_no_improve = 0

    def train_epoch(self, loader: DataLoader, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0

        for x, labels in tqdm(loader, desc=f"Epoch {epoch} [train]", leave=False):
            x, labels = x.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(x)
            loss   = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(loader)

    def eval_epoch(self, loader: DataLoader, epoch: int) -> tuple[float, float]:
        """Returns (avg_loss, EER)."""
        from src.evaluation.eer import compute_eer

        self.model.eval()
        total_loss = 0.0
        all_scores = []
        all_labels = []

        with torch.no_grad():
            for x, labels in tqdm(loader, desc=f"Epoch {epoch} [eval] ", leave=False):
                x, labels = x.to(self.device), labels.to(self.device)
                logits = self.model(x)
                loss   = self.criterion(logits, labels)
                total_loss += loss.item()

                # Softmax score for class 1 (spoof) used as detection score
                scores = torch.softmax(logits, dim=1)[:, 1]
                all_scores.extend(scores.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        avg_loss = total_loss / len(loader)
        eer      = compute_eer(all_labels, all_scores)
        return avg_loss, eer

    def fit(self, train_loader: DataLoader, dev_loader: DataLoader, epochs: int):
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader, epoch)
            dev_loss, dev_eer = self.eval_epoch(dev_loader, epoch)

            self.scheduler.step()

            lr = self.optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch:3d} | train_loss {train_loss:.4f} | dev_loss {dev_loss:.4f} | dev_EER {dev_eer:.4f} | lr {lr:.2e}")

            # TensorBoard logging
            self.writer.add_scalar("loss/train", train_loss, epoch)
            self.writer.add_scalar("loss/dev",   dev_loss,   epoch)
            self.writer.add_scalar("EER/dev",    dev_eer,    epoch)
            self.writer.add_scalar("lr",         lr,         epoch)

            # Save best checkpoint
            if dev_eer != dev_eer:  # nan check — skip if EER couldn't be computed
                print(f"           ↳ EER is NaN (too few samples) — skipping checkpoint")
                continue
            if dev_eer < self.best_eer:
                self.best_eer = dev_eer
                self.epochs_no_improve = 0
                torch.save(self.model.state_dict(), self.checkpoint_dir / "best.pt")
                print(f"           ↳ new best EER: {dev_eer:.4f} — checkpoint saved")
            else:
                self.epochs_no_improve += 1

            # Early stopping
            if self.epochs_no_improve >= self.patience:
                print(f"Early stopping at epoch {epoch} — no improvement for {self.patience} epochs")
                break

        self.writer.close()
        print(f"\nTraining complete. Best dev EER: {self.best_eer:.4f}")
