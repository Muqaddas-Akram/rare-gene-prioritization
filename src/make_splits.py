import json
import os
import random

import pandas as pd

PROCESSED = "data/processed"
SPLITS = "data/splits"
RAW = "data/raw"

os.makedirs(SPLITS, exist_ok=True)

SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.10
TEST_RATIO = 0.20


def build_phenotype_hierarchy():
    path = os.path.join(RAW, "hp.json")

    with open(path, "r", encoding="utf-8") as f:
        hpo = json.load(f)

    graphs = hpo.get("graphs", [])

    if not graphs:
        raise ValueError("No graphs found in hp.json")

    graph = graphs[0]
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])

    valid_phenotypes = set()

    for node in nodes:
        node_id = node.get("id", "")

        if node_id.startswith("HP:"):
            valid_phenotypes.add(node_id)

    hierarchy_rows = []

    for edge in edges:
        subject = edge.get("sub", "")
        object_node = edge.get("obj", "")
        predicate = edge.get("pred", "")

        if predicate != "is_a":
            continue

        if not subject.startswith("HP:") or not object_node.startswith("HP:"):
            continue

        if subject not in valid_phenotypes:
            continue

        if object_node not in valid_phenotypes:
            continue

        hierarchy_rows.append({
            "child_phenotype_id": subject,
            "parent_phenotype_id": object_node,
            "edge_type": "PHENOTYPE_IS_A_PHENOTYPE"
        })

    hierarchy_df = pd.DataFrame(hierarchy_rows).drop_duplicates(
        ["child_phenotype_id", "parent_phenotype_id"]
    )

    hierarchy_df.to_csv(
        f"{PROCESSED}/edges_phenotype_phenotype.csv",
        index=False
    )

    print(f"Phenotype hierarchy edges saved: {len(hierarchy_df)}")

    return hierarchy_df


def load_disease_gene_edges():
    path = os.path.join(
        PROCESSED,
        "edges_disease_gene.csv"
    )

    edges = pd.read_csv(path)

    required_columns = {
        "disease_id",
        "gene_id"
    }

    if not required_columns.issubset(edges.columns):
        raise ValueError(
            "edges_disease_gene.csv must contain disease_id and gene_id"
        )

    return edges


def make_edge_splits(dg_edges):
    random.seed(SEED)

    edges = dg_edges.copy()
    edges = edges.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    disease_gene_positive = set(
        zip(edges["disease_id"], edges["gene_id"])
    )

    train_rows = []
    val_rows = []
    test_rows = []

    for disease_id, disease_edges in edges.groupby("disease_id"):
        disease_edges = disease_edges.reset_index(drop=True)
        n = len(disease_edges)

        if n < 2:
            train_rows.append(disease_edges)
            continue

        n_test = max(1, int(round(n * TEST_RATIO)))
        n_val = max(1, int(round(n * VAL_RATIO)))

        if n_test + n_val >= n:
            n_test = max(1, n // 5)
            n_val = max(1, n // 10)

        if n_test + n_val >= n:
            train_rows.append(disease_edges)
            continue

        test_rows.append(disease_edges.tail(n_test))
        val_rows.append(
            disease_edges.iloc[n - n_test - n_val:n - n_test]
        )
        train_rows.append(
            disease_edges.iloc[:n - n_test - n_val]
        )

    train_df = pd.concat(train_rows, ignore_index=True)
    val_df = pd.concat(val_rows, ignore_index=True)
    test_df = pd.concat(test_rows, ignore_index=True)

    train_df.to_csv(f"{SPLITS}/train_edges.csv", index=False)
    val_df.to_csv(f"{SPLITS}/val_edges.csv", index=False)
    test_df.to_csv(f"{SPLITS}/test_edges.csv", index=False)

    print(f"Train edges: {len(train_df)}")
    print(f"Validation edges: {len(val_df)}")
    print(f"Test edges: {len(test_df)}")

    return train_df, val_df, test_df


def make_negative_samples(
    dg_edges,
    gene_df,
    train_df,
    val_df,
    test_df,
    negatives_per_positive=3
):
    random.seed(SEED)

    all_genes = gene_df["gene_id"].tolist()
    gene_set = set(all_genes)

    all_positive = set(
        zip(dg_edges["disease_id"], dg_edges["gene_id"])
    )

    split_frames = {
        "train": train_df,
        "val": val_df,
        "test": test_df
    }

    for split_name, split_df in split_frames.items():
        rows = []

        for _, row in split_df.iterrows():
            disease_id = row["disease_id"]
            positive_gene = row["gene_id"]

            candidates = list(gene_set)

            attempts = 0
            added = 0

            while added < negatives_per_positive and attempts < 100:
                attempts += 1
                negative_gene = random.choice(candidates)

                if negative_gene == positive_gene:
                    continue

                if (disease_id, negative_gene) in all_positive:
                    continue

                rows.append({
                    "disease_id": disease_id,
                    "gene_id": negative_gene,
                    "label": 0,
                    "split": split_name
                })

                added += 1

            rows.append({
                "disease_id": disease_id,
                "gene_id": positive_gene,
                "label": 1,
                "split": split_name
            })

        negative_df = pd.DataFrame(rows)
        negative_df.to_csv(
            f"{SPLITS}/{split_name}_classification_samples.csv",
            index=False
        )

        print(
            f"{split_name}: "
            f"{len(negative_df)} samples, "
            f"{negative_df['label'].sum()} positives"
        )


def main():
    print("1/3 Building phenotype hierarchy from hp.json...")
    build_phenotype_hierarchy()

    print("2/3 Creating disease-gene edge splits...")
    dg_edges = load_disease_gene_edges()

    gene_df = pd.read_csv(f"{PROCESSED}/nodes_gene.csv")

    train_df, val_df, test_df = make_edge_splits(dg_edges)

    print("3/3 Creating classification samples with negatives...")
    make_negative_samples(
        dg_edges=dg_edges,
        gene_df=gene_df,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        negatives_per_positive=3
    )

    print("\nSplits saved in:", SPLITS)


if __name__ == "__main__":
    main()