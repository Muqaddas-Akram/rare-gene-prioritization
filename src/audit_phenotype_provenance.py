from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

GRAPH = ROOT / "data" / "processed_gene_ids_fixed"
TARGETS = ROOT / "data" / "causal_target_audit"
OUTPUT = ROOT / "data" / "phenotype_provenance_audit"


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
    provenance = read_csv(
        GRAPH / "gene_phenotype_provenance.csv",
        ["gene_id", "hpo_id", "source_disease_id"]
    )

    targets = read_csv(
        TARGETS / "eligible_causal_pairs.csv",
        ["disease_id", "gene_id"]
    ).drop_duplicates(["disease_id", "gene_id"])

    disease_phenotypes = read_csv(
        GRAPH / "edges_disease_phenotype.csv",
        ["disease_id", "hpo_id"]
    ).drop_duplicates(["disease_id", "hpo_id"])

    eligible_diseases = set(targets["disease_id"])
    eligible_genes = set(targets["gene_id"])

    provenance = provenance[
        provenance["gene_id"].isin(eligible_genes)
    ].copy()

    provenance = provenance[
        ["gene_id", "hpo_id", "source_disease_id"]
    ].drop_duplicates()

    provenance["source_namespace"] = (
        provenance["source_disease_id"]
        .apply(
            lambda value: (
                value.split(":", 1)[0]
                if ":" in value
                else "MISSING_OR_UNRECOGNIZED"
            )
        )
    )

    source_counts = (
        provenance.groupby("source_namespace")
        .size()
        .reset_index(name="unique_provenance_records")
    )

    direct_orpha = provenance[
        provenance["source_disease_id"].isin(
            eligible_diseases
        )
    ].copy()

    direct_orpha = direct_orpha.rename(
        columns={"source_disease_id": "disease_id"}
    )

    target_derived = direct_orpha.merge(
        targets[["disease_id", "gene_id"]],
        on=["disease_id", "gene_id"],
        how="inner",
        validate="many_to_one"
    )

    shared_phenotypes = target_derived.merge(
        disease_phenotypes[["disease_id", "hpo_id"]],
        on=["disease_id", "hpo_id"],
        how="inner",
        validate="many_to_one"
    )

    target_derived_counts = (
        target_derived.groupby(["disease_id", "gene_id"])
        ["hpo_id"]
        .nunique()
        .rename("same_source_disease_phenotype_count")
        .reset_index()
    )

    shared_counts = (
        shared_phenotypes.groupby(
            ["disease_id", "gene_id"]
        )["hpo_id"]
        .nunique()
        .rename("also_in_disease_profile_count")
        .reset_index()
    )

    pair_audit = targets[
        ["disease_id", "gene_id"]
    ].copy()

    pair_audit = pair_audit.merge(
        target_derived_counts,
        on=["disease_id", "gene_id"],
        how="left"
    )

    pair_audit = pair_audit.merge(
        shared_counts,
        on=["disease_id", "gene_id"],
        how="left"
    )

    count_columns = [
        "same_source_disease_phenotype_count",
        "also_in_disease_profile_count",
    ]

    for column in count_columns:
        pair_audit[column] = (
            pair_audit[column]
            .fillna(0)
            .astype(int)
        )

    with_source = int(
        (
            pair_audit[
                "same_source_disease_phenotype_count"
            ] > 0
        ).sum()
    )

    with_shared_profile = int(
        (
            pair_audit[
                "also_in_disease_profile_count"
            ] > 0
        ).sum()
    )

    missing_source = int(
        provenance["source_disease_id"].eq("").sum()
    )

    summary = {
        "eligible_target_pairs": len(targets),
        "eligible_diseases": len(eligible_diseases),
        "eligible_genes": len(eligible_genes),
        "unique_provenance_records": len(provenance),
        "records_without_source_disease": missing_source,
        "targets_with_same_ORPHA_source_annotations":
            with_source,
        "targets_with_same_source_and_shared_phenotype":
            with_shared_profile,
        "same_ORPHA_source_percentage": (
            100 * with_source / len(targets)
            if len(targets) else 0
        ),
    }

    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise FileExistsError(
            f"{OUTPUT} is not empty. Rename/archive it "
            "before rerunning."
        )

    OUTPUT.mkdir(parents=True, exist_ok=True)

    source_counts.to_csv(
        OUTPUT / "source_namespace_counts.csv",
        index=False
    )

    pair_audit.to_csv(
        OUTPUT / "target_pair_provenance_audit.csv",
        index=False
    )

    target_derived.to_csv(
        OUTPUT / "same_source_ORPHA_annotations.csv",
        index=False
    )

    pd.DataFrame([summary]).to_csv(
        OUTPUT / "provenance_summary.csv",
        index=False
    )

    print("\n===== PHENOTYPE PROVENANCE AUDIT =====")
    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\n===== SOURCE NAMESPACES =====")
    print(source_counts.to_string(index=False))

    print("\n===== FIRST 10 FLAGGED TARGET PAIRS =====")
    flagged = pair_audit[
        pair_audit[
            "same_source_disease_phenotype_count"
        ] > 0
    ]

    print(
        flagged.head(10).to_string(index=False)
        if not flagged.empty
        else "No direct same-ORPHA-source matches found."
    )

    print("\nSaved to:", OUTPUT)

    print(
        "\nNo graph, split or training files were modified."
    )

    print(
        "This audit checks direct ORPHA identity only. "
        "OMIM-to-ORPHA equivalences are not resolved."
    )

    print(
        "A zero direct-match count does NOT certify "
        "absence of indirect target information."
    )


if __name__ == "__main__":
    main()