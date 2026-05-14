from pathlib import Path
import pandas as pd


def parse_protocol(protocol_file: str | Path, audio_dir: str | Path) -> pd.DataFrame:
    """Parse an ASVspoof 2019 LA countermeasure protocol file.

    Returns a DataFrame with one row per utterance, including the resolved
    path to the corresponding flac file.
    """
    protocol_file = Path(protocol_file)
    audio_dir = Path(audio_dir)

    records = []
    with open(protocol_file) as f:
        for line in f:
            parts = line.strip().split()
            speaker_id   = parts[0]
            utterance_id = parts[1]
            attack_type  = parts[3] if parts[3] != "-" else None
            label        = parts[4]  # "bonafide" or "spoof"

            file_path = audio_dir / f"{utterance_id}.flac"
            records.append({
                "speaker_id":   speaker_id,
                "utterance_id": utterance_id,
                "file_path":    str(file_path),
                "label":        label,
                "attack_type":  attack_type,
            })

    return pd.DataFrame(records)


def print_stats(df: pd.DataFrame, split: str) -> None:
    total    = len(df)
    counts   = df["label"].value_counts()
    bonafide = counts.get("bonafide", 0)
    spoof    = counts.get("spoof", 0)

    print(f"\n[{split}] {total} utterances")
    print(f"  bonafide : {bonafide:6d}  ({100 * bonafide / total:.1f}%)")
    print(f"  spoof    : {spoof:6d}  ({100 * spoof / total:.1f}%)")

    attack_counts = df[df["attack_type"].notna()]["attack_type"].value_counts()
    print(f"  attack breakdown: {dict(attack_counts)}")
