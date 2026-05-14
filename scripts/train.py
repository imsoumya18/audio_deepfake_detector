import torch
import yaml
from pathlib import Path
from torch.utils.data import DataLoader

from src.data.protocol import parse_protocol, print_stats
from src.data.dataset import ASVspoofDataset
from src.data.transforms import MelSpectrogramTransform
from src.models.lcnn import LCNN
from src.training.losses import build_loss
from src.training.trainer import Trainer


def main():
    # Load config
    with open("configs/lcnn.yaml") as f:
        cfg = yaml.safe_load(f)

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Dataset
    data_root = Path(cfg["paths"]["data_root"])
    protocols = data_root / "ASVspoof2019_LA_cm_protocols"

    df_train = parse_protocol(
        protocols / "ASVspoof2019.LA.cm.train.trn.txt",
        data_root / "ASVspoof2019_LA_train" / "flac",
    )
    df_dev = parse_protocol(
        protocols / "ASVspoof2019.LA.cm.dev.trl.txt",
        data_root / "ASVspoof2019_LA_dev" / "flac",
    )

    print_stats(df_train, "train")
    print_stats(df_dev,   "dev")

    train_dataset = ASVspoofDataset(df_train, transform=MelSpectrogramTransform(augment=True))
    dev_dataset   = ASVspoofDataset(df_dev,   transform=MelSpectrogramTransform(augment=False))

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )

    # Model, loss, optimizer, scheduler
    model     = LCNN()
    criterion = build_loss(df_train, device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg["scheduler"]["T_max"],
        eta_min=cfg["scheduler"]["eta_min"],
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        checkpoint_dir=cfg["paths"]["checkpoint_dir"],
        log_dir=cfg["paths"]["log_dir"],
        patience=cfg["training"]["patience"],
    )

    trainer.fit(train_loader, dev_loader, epochs=cfg["training"]["epochs"])


if __name__ == "__main__":
    main()
