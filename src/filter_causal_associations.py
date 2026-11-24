from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

SOURCE = ROOT / "data" / "processed_gene_ids_fixed"
OUTPUT = ROOT / "data" / "causal_target_audit"

ALLOWED_TYPES = {
    "Disease-causing germline mutation(s) in",
    "Disease-causing germline mutation(s) (loss of function) in",
    "Disease-causing germline mutation(s) (gain of function) in",
}

MIN_CAUSAL_GENES = 2
MIN_PHENOTYPES = 5


def read_csv(filename, required_columns):
    path = SOURCE / filename

    df = pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False
    )

    missing = set(required_columns) - set(df.columns)

    if missing:
        raise ValueError(
            f"{filename}: missing columns {sorted(missing)}"
        )

    for column in required_columns:
        df[column] = df[column].str.strip()

    return df


def main():
    evidence = read_csv(
        "disease_gene_association_evidence.csv",
        [
            "disease_id",
            "gene_id",
            "association_type",
            "association_status",
        ]
    )

    diseases = read_csv(
        "nodes_disease.csv",
        ["disease_id", "disease_name"]
    )

    phenotypes = read_csv(
        "edges_disease_phenotype.csv",
        ["disease_id", "hpo_id"]
    )

    allowed_type = evidence["association_type"].isin(
        ALLOWED_TYPES
    )

    assessed = (
        evidence["association_status"]
        .str.casefold()
        .eq("assessed")
    )

    retained_evidence = evidence[
        allowed_type & assessed
    ].copy()

    excluded_evidence = evidence[
        ~(allowed_type & assessed)
    ].copy()

    causal_pairs = retained_evidence.drop_duplicates(
        ["disease_id", "gene_id"]
    ).copy()

    if causal_pairs.empty:
        raise ValueError(
            "No assessed disease-causing germline pairs found."
        )

    gene_counts = (
        causal_pairs.groupby("disease_id")["gene_id"]
        .nunique()
    )

    phenotype_counts = (
        phenotypes.groupby("disease_id")["hpo_id"]
        .nunique()
    )

    cohort = diseases[
        ["disease_id", "disease_name"]
    ].drop_duplicates("disease_id").copy()

    cohort["causal_gene_count"] = (
        cohort["disease_id"]
        .map(gene_counts)
        .fillna(0)
        .astype(int)
    )

    cohort["phenotype_count"] = (
        cohort["disease_id"]
        .map(phenotype_counts)
        .fillna(0)
        .astype(int)
    )

    cohort["eligible"] = (
        (cohort["causal_gene_count"] >= MIN_CAUSAL_GENES)
        & (cohort["phenotype_count"] >= MIN_PHENOTYPES)
    )

    eligible = cohort[cohort["eligible"]].copy()

    eligible_pairs = causal_pairs[
        causal_pairs["disease_id"].isin(
            eligible["disease_id"]
        )
    ].copy()

    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise FileExistsError(
            f"{OUTPUT} is not empty. Rename/archive it "
            "before rerunning."
        )

    OUTPUT.mkdir(parents=True, exist_ok=True)

    retained_evidence.to_csv(
        OUTPUT / "retained_causal_evidence.csv",
        index=False
    )

    excluded_evidence.to_csv(
        OUTPUT / "excluded_association_evidence.csv",
        index=False
    )

    cohort.to_csv(
        OUTPUT / "causal_cohort_audit.csv",
        index=False
    )

    eligible.to_csv(
        OUTPUT / "eligible_causal_diseases.csv",
        index=False
    )

    eligible_pairs.to_csv(
        OUTPUT / "eligible_causal_pairs.csv",
        index=False
    )

    original_pairs = set(zip(
        evidence["disease_id"],
        evidence["gene_id"]
    ))

    retained_pairs = set(zip(
        causal_pairs["disease_id"],
        causal_pairs["gene_id"]
    ))

    excluded_only_pairs = original_pairs - retained_pairs

    pd.DataFrame(
        sorted(excluded_only_pairs),
        columns=["disease_id", "gene_id"]
    ).to_csv(
        OUTPUT / "excluded_only_pairs.csv",
        index=False
    )

    summary = {
        "input_evidence_rows": len(evidence),
        "input_unique_pairs": len(original_pairs),
        "retained_causal_evidence_rows": len(
            retained_evidence
        ),
        "retained_unique_causal_pairs": len(causal_pairs),
        "excluded_only_unique_pairs": len(
            excluded_only_pairs
        ),
        "diseases_with_at_least_one_causal_gene": int(
            (cohort["causal_gene_count"] >= 1).sum()
        ),
        "eligible_diseases_min_2_causal_genes": len(
            eligible
        ),
        "eligible_causal_pairs": len(eligible_pairs),
        "eligible_unique_genes": (
            eligible_pairs["gene_id"].nunique()
        ),
    }

    pd.DataFrame([summary]).to_csv(
        OUTPUT / "causal_target_statistics.csv",
        index=False
    )

    print("\n===== CAUSAL TARGET AUDIT =====")
    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\n===== CAUSAL GENE COUNT DISTRIBUTION =====")
    print(
        cohort["causal_gene_count"]
        .value_counts()
        .sort_index()
        .rename_axis("causal_genes")
        .to_frame("diseases")
        .to_string()
    )

    print("\nSaved to:", OUTPUT)

    print(
        "\nNo training files were overwritten. "
        "No splits were generated."
    )

    print(
        "Important: this is a subset of the previously "
        "selected diseases, not a new scan of every "
        "Orphanet disease."
    )

    print(
        "Excluded associations must not automatically "
        "be treated as confirmed negatives."
    )


if __name__ == "__main__":
    main()