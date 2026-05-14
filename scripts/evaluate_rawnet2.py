import torch
import yaml
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.protocol import parse_protocol
from src.data.dataset import ASVspoofDataset
from src.models.rawnet2 import RawNet2
from src.evaluation.eer import compute_eer, compute_eer_per_attack


def main():
    with open("configs/rawnet2.yaml") as f:
        cfg = yaml.safe_load(f)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    data_root = Path(cfg["paths"]["data_root"])
    protocols = data_root / "ASVspoof2019_LA_cm_protocols"

    df_eval = parse_protocol(
        protocols / "ASVspoof2019.LA.cm.eval.trl.txt",
        data_root / "ASVspoof2019_LA_eval" / "flac",
    )

    # No transform — raw waveforms
    dataset = ASVspoofDataset(df_eval, transform=None)
    loader  = DataLoader(dataset, batch_size=cfg["data"]["batch_size"],
                         shuffle=False, num_workers=0)

    model = RawNet2()
    model.load_state_dict(torch.load("checkpoints/rawnet2/best.pt", map_location=device))
    model = model.to(device)
    model.eval()

    all_scores = []
    all_labels = []

    with torch.no_grad():
        for x, labels in tqdm(loader, desc="Evaluating RawNet2"):
            x = x.to(device)
            logits = model(x)
            scores = torch.softmax(logits, dim=1)[:, 1]
            all_scores.extend(scores.cpu().tolist())
            all_labels.extend(labels.tolist())

    eer = compute_eer(all_labels, all_scores)
    print(f"\n{'='*45}")
    print(f"  RawNet2 Eval EER : {eer*100:.4f}%")
    print(f"  LCNN Eval EER    : 7.0724%")
    print(f"  Baseline EER     : 8.0900%  (LFCC-GMM)")
    print(f"{'='*45}")

    attack_types = df_eval["attack_type"].tolist()
    per_attack   = compute_eer_per_attack(all_labels, all_scores, attack_types)

    print(f"\nPer-attack EER:")
    print(f"  {'Attack':<10} {'RawNet2':>10}  {'LCNN':>10}")
    print(f"  {'-'*35}")
    lcnn_eer = {"A07": 0.0, "A08": 0.8158, "A09": 0.1224, "A10": 0.5846,
                "A11": 0.3663, "A12": 0.7750, "A13": 0.7937, "A14": 0.5088,
                "A15": 1.5228, "A16": 0.0, "A17": 36.8457, "A18": 9.7477, "A19": 0.0611}
    for attack, attack_eer in sorted(per_attack.items()):
        if attack == "overall":
            continue
        lcnn = lcnn_eer.get(attack, float("nan"))
        print(f"  {attack:<10} {attack_eer*100:>9.4f}%  {lcnn:>9.4f}%")


if __name__ == "__main__":
    main()
