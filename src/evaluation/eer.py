import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve


def compute_eer(labels: list[int], scores: list[float]) -> float:
    """
    Compute Equal Error Rate (EER).

    labels : 0 = bonafide, 1 = spoof
    scores : model's spoof probability for each sample (higher = more likely spoof)

    Returns EER as a value between 0 and 1.
    """
    labels = np.array(labels)
    scores = np.array(scores)

    # Need both classes to compute EER
    if len(np.unique(labels)) < 2:
        return float("nan")

    # ROC curve: FPR = FAR (spoof accepted as bonafide), TPR = 1 - FRR
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr  # FRR = 1 - TPR

    # Find threshold where FAR == FRR using Brent's method (fast root finding)
    try:
        eer = brentq(lambda x: interp1d(fpr, fpr - fnr)(x), fpr[0], fpr[-1])
    except ValueError:
        eer = float("nan")
    return float(eer)


def compute_eer_per_attack(
    labels: list[int],
    scores: list[float],
    attack_types: list[str | None],
) -> dict[str, float]:
    """Compute EER separately for each attack type."""
    labels       = np.array(labels)
    scores       = np.array(scores)
    attack_types = np.array(attack_types)

    results = {}

    # EER on bonafide vs all spoof combined
    results["overall"] = compute_eer(labels.tolist(), scores.tolist())

    # EER for each individual attack type
    bonafide_mask = labels == 0
    for attack in np.unique(attack_types[attack_types != np.array(None)]):
        attack_mask = attack_types == attack
        mask = bonafide_mask | attack_mask

        if mask.sum() < 10:
            continue

        results[attack] = compute_eer(labels[mask].tolist(), scores[mask].tolist())

    return results
