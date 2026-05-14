from pathlib import Path
from src.data.protocol import parse_protocol, print_stats

DATA = Path("data")
PROTOCOLS = DATA / "ASVspoof2019_LA_cm_protocols"

splits = {
    "train": (PROTOCOLS / "ASVspoof2019.LA.cm.train.trn.txt", DATA / "ASVspoof2019_LA_train" / "flac"),
    "dev":   (PROTOCOLS / "ASVspoof2019.LA.cm.dev.trl.txt",   DATA / "ASVspoof2019_LA_dev"   / "flac"),
    "eval":  (PROTOCOLS / "ASVspoof2019.LA.cm.eval.trl.txt",  DATA / "ASVspoof2019_LA_eval"  / "flac"),
}

for split, (protocol_file, audio_dir) in splits.items():
    df = parse_protocol(protocol_file, audio_dir)
    print_stats(df, split)

print("\nSample rows from train:")
print(parse_protocol(splits["train"][0], splits["train"][1]).head(3).to_string(index=False))
