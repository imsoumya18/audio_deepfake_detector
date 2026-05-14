import torch
import torch.nn as nn
import pandas as pd


def compute_class_weights(df: pd.DataFrame) -> torch.Tensor:
    """Compute inverse-frequency class weights from a protocol DataFrame."""
    counts = df["label"].value_counts()
    total  = len(df)
    n_classes = 2

    weight_bonafide = total / (n_classes * counts["bonafide"])
    weight_spoof    = total / (n_classes * counts["spoof"])

    # Index 0 = bonafide, index 1 = spoof (matches LABEL_MAP in dataset.py)
    return torch.tensor([weight_bonafide, weight_spoof], dtype=torch.float32)


def build_loss(df: pd.DataFrame, device: torch.device) -> nn.CrossEntropyLoss:
    weights = compute_class_weights(df).to(device)
    return nn.CrossEntropyLoss(weight=weights)
