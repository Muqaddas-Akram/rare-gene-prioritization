from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

FILES = {
    "nodes_gene": "nodes_gene.csv",
    "disease_gene": "edges_disease_gene.csv",
    "gene_phenotype": "edges_gene_phenotype.csv",
}


def read_file(filename):
    path = PROCESSED / filename

    df = pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False
    )

    if "gene_id" not in df.columns:
        raise ValueError(
            f"{filename}: gene_id column missing"
        )

    df["gene_id"] = df["gene_id"].str.strip()

    if "gene_symbol" in df.columns:
        df["gene_symbol"] = df["gene_symbol"].str.strip()

    return df


def describe(name, df):
    print(f"\n===== {name.upper()} =====")
    print("Rows:", len(df))
    print("Unique gene IDs:", df["gene_id"].nunique())
    print("Columns:", ", ".join(df.columns))

    unique_ids = df["gene_id"].drop_duplicates()

    print(
        "Numeric-only IDs:",
        int(unique_ids.str.fullmatch(r"\d+").sum())
    )
    print(
        "HGNC-prefixed IDs:",
        int(unique_ids.str.startswith("HGNC:").sum())
    )
    print(
        "NCBIGene-prefixed IDs:",
        int(unique_ids.str.startswith("NCBIGene:").sum())
    )

    columns = [
        col for col in ["gene_id", "gene_symbol"]
        if col in df.columns
    ]

    print("\nFirst 12 unique gene records:")
    print(
        df[columns]
        .drop_duplicates()
        .head(12)
        .to_string(index=False)
    )


def compare_symbols(left_name, left, right_name, right):
    if (
        "gene_symbol" not in left.columns
        or "gene_symbol" not in right.columns
    ):
        print(
            f"\nCannot compare symbols: "
            f"{left_name} or {right_name} lacks gene_symbol."
        )
        return

    a = left[["gene_id", "gene_symbol"]].drop_duplicates()
    b = right[["gene_id", "gene_symbol"]].drop_duplicates()

    a = a[a["gene_symbol"] != ""]
    b = b[b["gene_symbol"] != ""]

    comparison = a.merge(
        b,
        on="gene_id",
        how="inner",
        suffixes=("_left", "_right")
    )

    mismatches = comparison[
        comparison["gene_symbol_left"]
        != comparison["gene_symbol_right"]
    ]

    print(f"\n===== {left_name} vs {right_name} =====")
    print(
        "Shared gene IDs:",
        len(set(a["gene_id"]) & set(b["gene_id"]))
    )
    print(
        "Shared IDs with differing symbol records:",
        len(mismatches)
    )

    if not mismatches.empty:
        print("\nFirst 15 differing symbol records:")
        print(mismatches.head(15).to_string(index=False))

    print(
        "Note: differing symbols are audit flags, not automatic "
        "proof of an incorrect mapping; aliases may exist."
    )


def main():
    frames = {
        name: read_file(filename)
        for name, filename in FILES.items()
    }

    for name, df in frames.items():
        describe(name, df)

    compare_symbols(
        "disease_gene",
        frames["disease_gene"],
        "gene_phenotype",
        frames["gene_phenotype"]
    )

    compare_symbols(
        "nodes_gene",
        frames["nodes_gene"],
        "gene_phenotype",
        frames["gene_phenotype"]
    )

    print(
        "\nAudit complete. No input files were modified."
    )
    print(
        "Matching IDs/symbols alone do not establish the "
        "identifier namespace. Raw-source mapping must also "
        "be checked before training."
    )


if __name__ == "__main__":
    main()