import torch
import yaml
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.protocol import parse_protocol
from src.data.dataset import ASVspoofDataset
from src.data.transforms import MelSpectrogramTransform
from src.models.lcnn import LCNN
from src.models.rawnet2 import RawNet2
from src.models.ensemble import collect_scores, build_ensemble_weights, fuse_scores
from src.evaluation.eer import compute_eer, compute_eer_per_attack


def get_scores(model, df, transform, checkpoint, device, desc):
    dataset = ASVspoofDataset(df, transform=transform)
    loader  = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model = model.to(device)
    scores, labels = collect_scores(model, loader, device, use_mel=transform is not None)
    return scores, labels


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    data_root = Path("data")
    protocols = data_root / "ASVspoof2019_LA_cm_protocols"

    df_dev  = parse_protocol(protocols / "ASVspoof2019.LA.cm.dev.trl.txt",
                              data_root / "ASVspoof2019_LA_dev" / "flac")
    df_eval = parse_protocol(protocols / "ASVspoof2019.LA.cm.eval.trl.txt",
                              data_root / "ASVspoof2019_LA_eval" / "flac")

    mel = MelSpectrogramTransform(augment=False)

    print("Collecting dev scores...")
    lcnn_dev,    labels_dev = get_scores(LCNN(),    df_dev,  mel,  "checkpoints/best.pt",          device, "LCNN dev")
    rawnet2_dev, _          = get_scores(RawNet2(), df_dev,  None, "checkpoints/rawnet2/best.pt",  device, "RawNet2 dev")

    print("Collecting eval scores...")
    lcnn_eval,    labels_eval = get_scores(LCNN(),    df_eval, mel,  "checkpoints/best.pt",         device, "LCNN eval")
    rawnet2_eval, _           = get_scores(RawNet2(), df_eval, None, "checkpoints/rawnet2/best.pt", device, "RawNet2 eval")

    # Learn fusion weights from dev set
    lr_model = build_ensemble_weights(lcnn_dev, rawnet2_dev, labels_dev)

    # Evaluate all combinations
    results = {
        "LCNN only":          compute_eer(labels_eval, lcnn_eval),
        "RawNet2 only":       compute_eer(labels_eval, rawnet2_eval),
        "Ensemble (average)": compute_eer(labels_eval, fuse_scores(lcnn_eval, rawnet2_eval, "average")),
        "Ensemble (learned)": compute_eer(labels_eval, fuse_scores(lcnn_eval, rawnet2_eval, "learned", lr_model)),
    }

    print(f"\n{'='*50}")
    print(f"  {'Model':<25} {'Eval EER':>10}")
    print(f"  {'-'*40}")
    for name, eer in results.items():
        print(f"  {name:<25} {eer*100:>9.4f}%")
    print(f"  {'Baseline (LFCC-GMM)':<25} {'8.0900%':>10}")
    print(f"{'='*50}")

    # Per-attack for best ensemble
    best_name  = min(results, key=results.get)
    best_scores = fuse_scores(lcnn_eval, rawnet2_eval, "learned", lr_model) \
                  if "learned" in best_name else lcnn_eval

    attack_types = df_eval["attack_type"].tolist()
    per_attack   = compute_eer_per_attack(labels_eval, best_scores, attack_types)

    print(f"\nPer-attack EER ({best_name}):")
    print(f"  {'Attack':<10} {'Ensemble':>10}  {'LCNN':>10}  {'RawNet2':>10}")
    print(f"  {'-'*45}")
    lcnn_eer    = {"A07":0.0,"A08":0.8158,"A09":0.1224,"A10":0.5846,"A11":0.3663,
                   "A12":0.7750,"A13":0.7937,"A14":0.5088,"A15":1.5228,"A16":0.0,
                   "A17":36.8457,"A18":9.7477,"A19":0.0611}
    rawnet2_eer = {"A07":1.2644,"A08":6.9613,"A09":1.0582,"A10":1.2237,"A11":1.1557,
                   "A12":1.6859,"A13":2.6252,"A14":1.6044,"A15":1.4652,"A16":2.1571,
                   "A17":40.3807,"A18":40.8221,"A19":12.6988}
    for attack, eer in sorted(per_attack.items()):
        if attack == "overall":
            continue
        l = lcnn_eer.get(attack, float("nan"))
        r = rawnet2_eer.get(attack, float("nan"))
        print(f"  {attack:<10} {eer*100:>9.4f}%  {l:>9.4f}%  {r:>9.4f}%")


if __name__ == "__main__":
    main()
