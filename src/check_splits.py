from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "data" / "splits"


def load_pairs(filename):
    path = SPLITS / filename
    df = pd.read_csv(path, dtype=str, keep_default_na=False)

    required = ["disease_id", "gene_id"]
    missing = set(required) - set(df.columns)

    if missing:
        raise ValueError(f"{filename}: missing columns {missing}")

    for column in required:
        df[column] = df[column].str.strip()

    if df.empty:
        raise ValueError(f"{filename}: file has no edges")

    if df[required].eq("").any().any():
        raise ValueError(f"{filename}: empty disease/gene IDs found")

    pairs = set(zip(df["disease_id"], df["gene_id"]))

    print(
        f"{filename}: rows={len(df)}, "
        f"unique_pairs={len(pairs)}, "
        f"duplicates={len(df) - len(pairs)}"
    )

    if len(df) != len(pairs):
        raise ValueError(f"{filename}: duplicate positive pairs found")

    return pairs


def main():
    train = load_pairs("train_edges.csv")
    val = load_pairs("val_edges.csv")
    test = load_pairs("test_edges.csv")

    overlaps = {
        "Train–validation": train & val,
        "Train–test": train & test,
        "Validation–test": val & test,
    }

    for name, pairs in overlaps.items():
        print(f"{name} overlap: {len(pairs)}")

    if any(overlaps.values()):
        raise ValueError(
            "FAIL: positive pairs overlap across splits."
        )

    train_diseases = {d for d, g in train}
    heldout_diseases = {d for d, g in val | test}
    missing_training_diseases = heldout_diseases - train_diseases

    print(
        "Held-out diseases without a training association:",
        len(missing_training_diseases)
    )

    if missing_training_diseases:
        raise ValueError(
            "FAIL: some validation/test diseases have no training link."
        )

    print("\nPASS: positive splits are disjoint.")
    print(
        "Use train_edges.csv only for disease–gene "
        "message passing."
    )


if __name__ == "__main__":
    main()