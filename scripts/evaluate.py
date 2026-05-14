import torch
import yaml
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.protocol import parse_protocol
from src.data.dataset import ASVspoofDataset
from src.data.transforms import MelSpectrogramTransform
from src.models.lcnn import LCNN
from src.evaluation.eer import compute_eer, compute_eer_per_attack


def main():
    with open("configs/lcnn.yaml") as f:
        cfg = yaml.safe_load(f)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Load eval set
    data_root = Path(cfg["paths"]["data_root"])
    protocols = data_root / "ASVspoof2019_LA_cm_protocols"

    df_eval = parse_protocol(
        protocols / "ASVspoof2019.LA.cm.eval.trl.txt",
        data_root / "ASVspoof2019_LA_eval" / "flac",
    )

    dataset = ASVspoofDataset(df_eval, transform=MelSpectrogramTransform(augment=False))
    loader  = DataLoader(dataset, batch_size=cfg["data"]["batch_size"],
                         shuffle=False, num_workers=0)

    # Load best checkpoint
    model = LCNN()
    model.load_state_dict(torch.load("checkpoints/best.pt", map_location=device))
    model = model.to(device)
    model.eval()

    all_scores  = []
    all_labels  = []

    with torch.no_grad():
        for x, labels in tqdm(loader, desc="Evaluating"):
            x = x.to(device)
            logits = model(x)
            scores = torch.softmax(logits, dim=1)[:, 1]
            all_scores.extend(scores.cpu().tolist())
            all_labels.extend(labels.tolist())

    # Overall EER
    eer = compute_eer(all_labels, all_scores)
    print(f"\n{'='*45}")
    print(f"  LCNN Eval EER : {eer*100:.4f}%")
    print(f"  Baseline EER  : 8.0900%  (LFCC-GMM)")
    print(f"{'='*45}")

    # Per-attack EER
    attack_types = df_eval["attack_type"].tolist()
    per_attack   = compute_eer_per_attack(all_labels, all_scores, attack_types)

    print(f"\nPer-attack EER:")
    print(f"  {'Attack':<10} {'EER':>8}")
    print(f"  {'-'*20}")
    for attack, attack_eer in sorted(per_attack.items()):
        if attack == "overall":
            continue
        print(f"  {attack:<10} {attack_eer*100:>7.4f}%")


if __name__ == "__main__":
    main()
