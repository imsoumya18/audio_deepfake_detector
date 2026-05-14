import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression

from src.models.lcnn import LCNN
from src.models.rawnet2 import RawNet2
from src.data.dataset import ASVspoofDataset
from src.data.transforms import MelSpectrogramTransform


def collect_scores(model, loader, device, use_mel: bool) -> tuple[list, list]:
    """Run model on a dataloader, return (scores, labels)."""
    model.eval()
    all_scores, all_labels = [], []

    with torch.no_grad():
        for x, labels in loader:
            x = x.to(device)
            logits = model(x)
            scores = torch.softmax(logits, dim=1)[:, 1]
            all_scores.extend(scores.cpu().tolist())
            all_labels.extend(labels.tolist())

    return all_scores, all_labels


def build_ensemble_weights(
    lcnn_scores_dev: list,
    rawnet2_scores_dev: list,
    labels_dev: list,
) -> LogisticRegression:
    """Learn optimal fusion weights from dev set scores."""
    X = np.stack([lcnn_scores_dev, rawnet2_scores_dev], axis=1)
    y = np.array(labels_dev)
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X, y)
    print(f"Learned weights — LCNN: {lr.coef_[0][0]:.3f}, RawNet2: {lr.coef_[0][1]:.3f}")
    return lr


def fuse_scores(
    lcnn_scores: list,
    rawnet2_scores: list,
    fusion: str = "average",
    lr_model: LogisticRegression = None,
) -> list:
    """Fuse scores from two models.

    fusion: 'average' | 'learned'
    """
    lcnn   = np.array(lcnn_scores)
    rawnet = np.array(rawnet2_scores)

    if fusion == "average":
        return ((lcnn + rawnet) / 2).tolist()
    elif fusion == "learned" and lr_model is not None:
        X = np.stack([lcnn, rawnet], axis=1)
        return lr_model.predict_proba(X)[:, 1].tolist()
    else:
        raise ValueError(f"Unknown fusion: {fusion}")
