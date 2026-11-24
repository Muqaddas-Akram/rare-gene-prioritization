import os
import xml.etree.ElementTree as ET
import pandas as pd
from tqdm import tqdm

RAW = "data/raw"
OUT = "data/processed"
os.makedirs(OUT, exist_ok=True)

MIN_GENES = 2
MIN_PHENOTYPES = 5


def clean_id(value):
    return str(value).strip()


def parse_orphanet_gene_disease():
    path = os.path.join(RAW, "en_product6.xml")
    tree = ET.parse(path)
    root = tree.getroot()

    disease_rows = []
    gene_rows = []
    disease_gene_edges = []

    for disorder in tqdm(root.iter("Disorder"), desc="Parsing Orphanet"):
        orpha_code = disorder.findtext("OrphaCode")
        if not orpha_code:
            continue

        disease_id = f"ORPHA:{clean_id(orpha_code)}"
        disease_name = disorder.findtext("Name") or ""

        disorder_type = disorder.find("DisorderType")
        disorder_group = disorder.find("DisorderGroup")

        disease_rows.append({
            "disease_id": disease_id,
            "disease_name": disease_name.strip(),
            "orpha_code": clean_id(orpha_code),
            "disorder_type": (
                disorder_type.findtext("Name")
                if disorder_type is not None else ""
            ),
            "disorder_group": (
                disorder_group.findtext("Name")
                if disorder_group is not None else ""
            )
        })

        association_list = disorder.find("DisorderGeneAssociationList")

        if association_list is None:
            continue

        for association in association_list.findall("DisorderGeneAssociation"):
            gene = association.find("Gene")

            if gene is None:
                continue

            symbol_node = gene.find("Symbol")
            gene_symbol = (
                symbol_node.text.strip()
                if symbol_node is not None and symbol_node.text else ""
            )

            if not gene_symbol:
                continue

            gene_id = gene_symbol

            external_refs = gene.find("ExternalReferenceList")
            if external_refs is not None:
                for ref in external_refs.findall("ExternalReference"):
                    source = ref.findtext("Source")
                    reference = ref.findtext("Reference")

                    if source == "HGNC" and reference:
                        gene_id = clean_id(reference)
                        break

            association_type = association.find("DisorderGeneAssociationType")
            association_name = (
                association_type.findtext("Name")
                if association_type is not None else ""
            )

            gene_rows.append({
                "gene_id": gene_id,
                "gene_symbol": gene_symbol,
                "source": "Orphanet"
            })

            disease_gene_edges.append({
                "disease_id": disease_id,
                "gene_id": gene_id,
                "gene_symbol": gene_symbol,
                "association_type": association_name,
                "edge_type": "DISEASE_ASSOCIATED_WITH_GENE"
            })

    disease_df = pd.DataFrame(disease_rows).drop_duplicates("disease_id")
    gene_df = pd.DataFrame(gene_rows).drop_duplicates("gene_id")
    dg_edges = pd.DataFrame(disease_gene_edges).drop_duplicates(
        ["disease_id", "gene_id"]
    )

    print(f"Orphanet diseases parsed: {len(disease_df)}")
    print(f"Orphanet genes parsed: {len(gene_df)}")
    print(f"Disease-gene edges parsed: {len(dg_edges)}")

    return disease_df, gene_df, dg_edges


def parse_hpo_disease_phenotype():
    path = os.path.join(RAW, "phenotype.hpoa")
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        header_found = False
        columns = []

        for line in f:
            line = line.strip()

            if not header_found:
                if line.startswith("database_id") or line.startswith("DatabaseID"):
                    columns = line.split("\t")
                    header_found = True
                continue

            if not line or line.startswith("#"):
                continue

            parts = line.split("\t")

            if len(parts) < len(columns):
                continue

            row = dict(zip(columns, parts))

            disease_id = row.get("database_id", row.get("DatabaseID", ""))
            hpo_id = row.get("hpo_id", row.get("HPO_ID", ""))
            qualifier = row.get("qualifier", row.get("Qualifier", ""))
            aspect = row.get("aspect", row.get("Aspect", ""))

            if aspect != "P":
                continue

            if qualifier.upper() == "NOT":
                continue

            if not disease_id or not hpo_id:
                continue

            rows.append({
                "disease_id": disease_id,
                "hpo_id": hpo_id,
                "edge_type": "DISEASE_HAS_PHENOTYPE"
            })

    dp_edges = pd.DataFrame(rows).drop_duplicates(
        ["disease_id", "hpo_id"]
    )

    phenotype_df = pd.DataFrame({
        "phenotype_id": dp_edges["hpo_id"].unique()
    })

    print(f"Disease-phenotype edges parsed: {len(dp_edges)}")

    return phenotype_df, dp_edges


def parse_hpo_gene_phenotype():
    path = os.path.join(RAW, "genes_to_phenotype.txt")
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue

            parts = line.rstrip("\n").split("\t")

            if len(parts) < 4:
                continue

            ncbi_gene_id, gene_symbol, hpo_id, hpo_name = parts[:4]

            if not ncbi_gene_id or not hpo_id:
                continue

            rows.append({
                "gene_id": clean_id(ncbi_gene_id),
                "gene_symbol": gene_symbol,
                "hpo_id": hpo_id,
                "edge_type": "GENE_ASSOCIATED_WITH_PHENOTYPE"
            })

    gp_edges = pd.DataFrame(rows).drop_duplicates(
        ["gene_id", "hpo_id"]
    )

    gene_df = gp_edges[["gene_id", "gene_symbol"]].drop_duplicates("gene_id")
    phenotype_df = pd.DataFrame({
        "phenotype_id": gp_edges["hpo_id"].unique()
    })

    print(f"Gene-phenotype edges parsed: {len(gp_edges)}")

    return gene_df, phenotype_df, gp_edges


def main():
    print("1/5 Parsing Orphanet gene-disease data...")
    orphanet_diseases, orphanet_genes, dg_edges = parse_orphanet_gene_disease()

    if dg_edges.empty:
        raise ValueError(
            "No disease-gene edges were parsed. "
            "Check en_product6.xml structure or re-download the file."
        )

    print("2/5 Parsing HPO disease-phenotype annotations...")
    hpo_phenotypes, dp_edges = parse_hpo_disease_phenotype()

    print("3/5 Parsing HPO gene-phenotype annotations...")
    hpo_genes, hpo_phenotypes_gp, gp_edges = parse_hpo_gene_phenotype()

    print("4/5 Merging and filtering graph...")

    disease_df = orphanet_diseases.copy()

    gene_df = pd.concat(
        [orphanet_genes, hpo_genes],
        ignore_index=True
    ).drop_duplicates("gene_id")

    phenotype_df = pd.concat(
        [hpo_phenotypes, hpo_phenotypes_gp],
        ignore_index=True
    ).drop_duplicates("phenotype_id")

    dg_counts = dg_edges.groupby("disease_id").size().rename("gene_count")
    dp_counts = dp_edges.groupby("disease_id").size().rename("phenotype_count")

    disease_df = disease_df.merge(dg_counts, on="disease_id", how="left")
    disease_df = disease_df.merge(dp_counts, on="disease_id", how="left")

    disease_df["gene_count"] = (
        disease_df["gene_count"].fillna(0).astype(int)
    )
    disease_df["phenotype_count"] = (
        disease_df["phenotype_count"].fillna(0).astype(int)
    )

    eligible_diseases = disease_df[
        (disease_df["gene_count"] >= MIN_GENES)
        & (disease_df["phenotype_count"] >= MIN_PHENOTYPES)
    ].copy()

    eligible_ids = set(eligible_diseases["disease_id"])

    dg_edges = dg_edges[dg_edges["disease_id"].isin(eligible_ids)]
    dp_edges = dp_edges[dp_edges["disease_id"].isin(eligible_ids)]

    used_genes = set(dg_edges["gene_id"])
    used_phenotypes = set(dp_edges["hpo_id"])

    gene_df = gene_df[gene_df["gene_id"].isin(used_genes)]
    phenotype_df = phenotype_df[
        phenotype_df["phenotype_id"].isin(used_phenotypes)
    ]

    gp_edges = gp_edges[
        gp_edges["gene_id"].isin(used_genes)
        & gp_edges["hpo_id"].isin(used_phenotypes)
    ]

    print("5/5 Saving processed files...")

    disease_df.to_csv(f"{OUT}/nodes_disease.csv", index=False)
    gene_df.to_csv(f"{OUT}/nodes_gene.csv", index=False)
    phenotype_df.to_csv(f"{OUT}/nodes_phenotype.csv", index=False)

    dg_edges.to_csv(f"{OUT}/edges_disease_gene.csv", index=False)
    dp_edges.to_csv(f"{OUT}/edges_disease_phenotype.csv", index=False)
    gp_edges.to_csv(f"{OUT}/edges_gene_phenotype.csv", index=False)

    stats = {
        "selected_diseases": len(disease_df),
        "eligible_diseases": len(eligible_diseases),
        "genes": len(gene_df),
        "phenotypes": len(phenotype_df),
        "disease_gene_edges": len(dg_edges),
        "disease_phenotype_edges": len(dp_edges),
        "gene_phenotype_edges": len(gp_edges),
        "average_genes_per_selected_disease": round(
            disease_df["gene_count"].mean(), 2
        ),
        "average_phenotypes_per_selected_disease": round(
            disease_df["phenotype_count"].mean(), 2
        )
    }

    stats_df = pd.DataFrame([stats])
    stats_df.to_csv(f"{OUT}/graph_statistics.csv", index=False)

    print("\n===== GRAPH STATISTICS =====")
    print(stats_df.T)

    print("\nFiles saved in:", OUT)


if __name__ == "__main__":
    main()