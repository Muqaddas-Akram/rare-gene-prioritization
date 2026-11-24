from pathlib import Path
import hashlib
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

GRAPH = ROOT / "data" / "processed_gene_ids_fixed"
TARGETS = ROOT / "data" / "causal_target_audit"
OUTPUT = ROOT / "data" / "primary_dataset"


def read_csv(path, required):
    df = pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False
    )

    missing = set(required) - set(df.columns)

    if missing:
        raise ValueError(
            f"{path.name}: missing columns {sorted(missing)}"
        )

    for column in required:
        df[column] = df[column].str.strip()

    return df


def main():
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise FileExistsError(
            f"{OUTPUT} is not empty. Rename/archive it "
            "before rerunning."
        )

    targets_path = TARGETS / "eligible_causal_pairs.csv"

    dg = read_csv(
        targets_path,
        ["disease_id", "gene_id", "gene_symbol"]
    ).drop_duplicates(["disease_id", "gene_id"])

    disease_ids = set(dg["disease_id"])
    gene_ids = set(dg["gene_id"])

    diseases = read_csv(
        GRAPH / "nodes_disease.csv",
        ["disease_id", "disease_name"]
    )

    diseases = diseases[
        diseases["disease_id"].isin(disease_ids)
    ].copy()

    genes = read_csv(
        GRAPH / "nodes_gene.csv",
        ["gene_id", "gene_symbol"]
    )

    genes = genes[
        genes["gene_id"].isin(gene_ids)
    ].copy()

    dp = read_csv(
        GRAPH / "edges_disease_phenotype.csv",
        ["disease_id", "hpo_id"]
    )

    dp = dp[
        dp["disease_id"].isin(disease_ids)
    ].drop_duplicates(["disease_id", "hpo_id"])

    phenotype_ids = set(dp["hpo_id"])

    phenotypes = read_csv(
        GRAPH / "nodes_phenotype.csv",
        ["phenotype_id"]
    )

    phenotypes = phenotypes[
        phenotypes["phenotype_id"].isin(phenotype_ids)
    ].copy()

    gene_counts = dg.groupby(
        "disease_id"
    )["gene_id"].nunique()

    phenotype_counts = dp.groupby(
        "disease_id"
    )["hpo_id"].nunique()

    diseases["gene_count"] = (
        diseases["disease_id"]
        .map(gene_counts)
        .fillna(0)
        .astype(int)
    )

    diseases["phenotype_count"] = (
        diseases["disease_id"]
        .map(phenotype_counts)
        .fillna(0)
        .astype(int)
    )

    if dg.empty:
        raise ValueError("No causal target pairs.")

    if diseases["disease_id"].duplicated().any():
        raise ValueError("Duplicate disease nodes.")

    if genes["gene_id"].duplicated().any():
        raise ValueError("Duplicate gene nodes.")

    if set(diseases["disease_id"]) != disease_ids:
        raise ValueError("Missing disease nodes.")

    if set(genes["gene_id"]) != gene_ids:
        raise ValueError("Missing gene nodes.")

    if set(phenotypes["phenotype_id"]) != phenotype_ids:
        raise ValueError("Missing phenotype nodes.")

    if not genes["gene_id"].str.fullmatch(
        r"NCBIGene:\d+"
    ).all():
        raise ValueError("Invalid canonical gene IDs.")

    if (diseases["gene_count"] < 2).any():
        raise ValueError(
            "A disease has fewer than two causal genes."
        )

    if (diseases["phenotype_count"] < 5).any():
        raise ValueError(
            "A disease has fewer than five phenotypes."
        )

    # Block every known mapped association from being
    # sampled as a negative, not only causal positives.
    all_known = read_csv(
        GRAPH / "disease_gene_association_evidence.csv",
        ["disease_id", "gene_id"]
    )

    blocked = all_known[
        all_known["disease_id"].isin(disease_ids)
        & all_known["gene_id"].isin(gene_ids)
    ][["disease_id", "gene_id"]].drop_duplicates()

    target_pairs = set(zip(
        dg["disease_id"], dg["gene_id"]
    ))

    blocked_pairs = set(zip(
        blocked["disease_id"], blocked["gene_id"]
    ))

    if not target_pairs.issubset(blocked_pairs):
        raise ValueError(
            "Negative-block list is missing causal positives."
        )

    OUTPUT.mkdir(parents=True, exist_ok=True)

    diseases.to_csv(
        OUTPUT / "nodes_disease.csv", index=False
    )
    genes.to_csv(
        OUTPUT / "nodes_gene.csv", index=False
    )
    phenotypes.to_csv(
        OUTPUT / "nodes_phenotype.csv", index=False
    )

    dg.to_csv(
        OUTPUT / "edges_disease_gene.csv", index=False
    )
    dp.to_csv(
        OUTPUT / "edges_disease_phenotype.csv",
        index=False
    )
    blocked.to_csv(
        OUTPUT / "negative_block_pairs.csv",
        index=False
    )

    stats = {
        "diseases": len(diseases),
        "genes": len(genes),
        "phenotypes": len(phenotypes),
        "causal_disease_gene_pairs": len(dg),
        "disease_phenotype_edges": len(dp),
        "gene_phenotype_edges_included": 0,
        "phenotype_hierarchy_edges_included": 0,
        "known_pairs_blocked_from_negative_sampling":
            len(blocked),
    }

    pd.DataFrame([stats]).to_csv(
        OUTPUT / "dataset_statistics.csv",
        index=False
    )

    manifest = {
        "stage": "primary candidate dataset; not split",
        "gene_namespace": "NCBIGene",
        "target": "assessed disease-causing germline associations",
        "cohort": "previously selected subset with >=2 causal genes and >=5 phenotypes",
        "gene_phenotype_edges": "excluded because of disease-derived provenance",
        "hierarchy": "excluded",
        "disease_phenotypes": "curated disease-level context; not patient validation",
        "source_target_sha256": hashlib.sha256(
            targets_path.read_bytes()
        ).hexdigest(),
        "negative_block_scope":
            "all mapped association types among retained diseases and genes",
    }

    (OUTPUT / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8"
    )

    print("\n===== PRIMARY DATASET =====")
    for key, value in stats.items():
        print(f"{key}: {value}")

    print("\nSaved to:", OUTPUT)

    print(
        "\nNo splits were generated. "
        "No models were trained."
    )

    print(
        "Do not run the old training scripts against "
        "this dataset yet."
    )


if __name__ == "__main__":
    main()