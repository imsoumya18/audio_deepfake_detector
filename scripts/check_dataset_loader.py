from pathlib import Path
from torch.utils.data import DataLoader
from src.data.protocol import parse_protocol
from src.data.dataset import ASVspoofDataset

DATA = Path("data")
PROTOCOLS = DATA / "ASVspoof2019_LA_cm_protocols"

df_train = parse_protocol(
    PROTOCOLS / "ASVspoof2019.LA.cm.train.trn.txt",
    DATA / "ASVspoof2019_LA_train" / "flac",
)

dataset = ASVspoofDataset(df_train)
print(f"Dataset size : {len(dataset)}")

waveform, label = dataset[0]
print(f"Waveform shape: {waveform.shape}")
print(f"Waveform dtype: {waveform.dtype}")
print(f"Label         : {label}  ({'bonafide' if label == 0 else 'spoof'})")

loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)
batch_waveforms, batch_labels = next(iter(loader))
print(f"\nBatch waveforms shape: {batch_waveforms.shape}")
print(f"Batch labels        : {batch_labels}")
