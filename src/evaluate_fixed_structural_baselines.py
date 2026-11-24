from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "warm_connected_seed42"


def read_csv(filename, required):
    path = DATA / filename

    df = pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
    )

    missing = set(required) - set(df.columns)

    if missing:
        raise ValueError(
            f"{filename}: missing columns {sorted(missing)}"
        )

    for column in required:
        df[column] = df[column].str.strip()

    if df[required].eq("").any().any():
        raise ValueError(
            f"{filename}: empty required values"
        )

    return df


def pairset(df):
    return set(zip(df["disease_id"], df["gene_id"]))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="val",
    )

    parser.add_argument(
        "--run-name",
        default="fixed_protocol_controls",
    )

    args = parser.parse_args()

    allowed = (
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789_-"
    )

    if not args.run_name or any(
        character not in allowed
        for character in args.run_name
    ):
        raise ValueError(
            "run-name mein sirf letters, numbers, "
            "underscores aur hyphens use karein."
        )

    output = (
        ROOT
        / "results"
        / "warm_connected"
        / f"{args.run_name}_{args.split}"
    )

    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"{output} already has files. "
            "Use a different --run-name."
        )

    mappings = {}

    for node_type, column in [
        ("disease", "disease_id"),
        ("gene", "gene_id"),
        ("phenotype", "phenotype_id"),
    ]:
        nodes = read_csv(
            f"nodes_{node_type}.csv",
            [column],
        )

        if nodes.empty or nodes[column].duplicated().any():
            raise ValueError(
                f"Invalid {node_type} node table."
            )

        mappings[node_type] = {
            identifier: index
            for index, identifier in enumerate(
                sorted(nodes[column])
            )
        }

    train = read_csv(
        "train_edges.csv",
        ["disease_id", "gene_id"],
    )

    targets = read_csv(
        f"{args.split}_edges.csv",
        ["disease_id", "gene_id"],
    )

    samples = read_csv(
        f"{args.split}_classification_samples.csv",
        ["disease_id", "gene_id", "label"],
    )

    known = read_csv(
        "known_causal_pairs_for_filtering.csv",
        ["disease_id", "gene_id"],
    )

    blocked = read_csv(
        "negative_block_pairs.csv",
        ["disease_id", "gene_id"],
    )

    dp = read_csv(
        "edges_disease_phenotype.csv",
        ["disease_id", "hpo_id"],
    ).drop_duplicates(["disease_id", "hpo_id"])

    samples["label"] = pd.to_numeric(
        samples["label"],
        errors="raise",
    )

    if set(samples["label"].unique()) != {0, 1}:
        raise ValueError("Both labels 0 and 1 are required.")

    for name, frame in [
        ("train", train),
        ("targets", targets),
        ("samples", samples),
    ]:
        if frame.empty or frame.duplicated(
            ["disease_id", "gene_id"]
        ).any():
            raise ValueError(
                f"{name}: empty or duplicate pairs."
            )

    if pairset(train) & pairset(targets):
        raise ValueError("Training/held-out positive overlap.")

    if pairset(
        samples[samples["label"] == 1]
    ) != pairset(targets):
        raise ValueError(
            "Classification positives do not match targets."
        )

    if pairset(
        samples[samples["label"] == 0]
    ) & pairset(blocked):
        raise ValueError(
            "Known associations sampled as negatives."
        )

    if not (
        pairset(train) | pairset(targets)
    ).issubset(pairset(known)):
        raise ValueError(
            "Known-causal ranking filter is incomplete."
        )

    for frame in [train, targets, samples, known]:
        if (
            not frame["disease_id"].isin(
                mappings["disease"]
            ).all()
            or not frame["gene_id"].isin(
                mappings["gene"]
            ).all()
        ):
            raise ValueError(
                "Unknown disease/gene endpoint."
            )

    if (
        not dp["disease_id"].isin(
            mappings["disease"]
        ).all()
        or not dp["hpo_id"].isin(
            mappings["phenotype"]
        ).all()
    ):
        raise ValueError("Unknown phenotype-edge endpoint.")

    number_diseases = len(mappings["disease"])
    number_genes = len(mappings["gene"])
    number_phenotypes = len(mappings["phenotype"])

    # Only training disease-gene links construct scores.
    A = np.zeros(
        (number_diseases, number_genes),
        dtype=np.float64,
    )

    A[
        train["disease_id"].map(mappings["disease"]),
        train["gene_id"].map(mappings["gene"]),
    ] = 1.0

    B = np.zeros(
        (number_diseases, number_phenotypes),
        dtype=np.float64,
    )

    B[
        dp["disease_id"].map(mappings["disease"]),
        dp["hpo_id"].map(mappings["phenotype"]),
    ] = 1.0

    disease_degree = A.sum(axis=1)
    gene_degree = A.sum(axis=0)

    if (
        (disease_degree == 0).any()
        or (gene_degree == 0).any()
    ):
        raise ValueError(
            "Expected training-connected diseases and genes."
        )

    # Disease -> gene -> disease -> gene.
    Pdg = A / disease_degree[:, None]
    Pgd = A.T / gene_degree[:, None]
    walk_scores = Pdg @ Pgd @ Pdg

    # Direct disease-phenotype cosine similarity.
    lengths = np.linalg.norm(B, axis=1)
    normalized = B / np.maximum(lengths[:, None], 1.0)

    similarity = normalized @ normalized.T
    np.fill_diagonal(similarity, 0.0)
    similarity = np.maximum(similarity, 0.0)

    phenotype_scores = similarity @ A
    phenotype_scores /= np.maximum(
        similarity.sum(axis=1, keepdims=True),
        1e-12,
    )

    methods = {
        "gene_degree": np.broadcast_to(
            gene_degree,
            (number_diseases, number_genes),
        ),
        "dg_three_step_walk": walk_scores,
        "dp_cosine_transfer": phenotype_scores,
    }

    known_by_disease = {}

    for disease, gene in pairset(known):
        known_by_disease.setdefault(
            disease, set()
        ).add(mappings["gene"][gene])

    disease_indices = samples["disease_id"].map(
        mappings["disease"]
    ).to_numpy()

    gene_indices = samples["gene_id"].map(
        mappings["gene"]
    ).to_numpy()

    labels = samples["label"].to_numpy(dtype=int)

    results = []
    artifacts = {}

    for method, score_matrix in methods.items():
        if not np.isfinite(score_matrix).all():
            raise ValueError(
                f"{method}: non-finite scores."
            )

        sample_scores = score_matrix[
            disease_indices, gene_indices
        ]

        rank_rows = []

        for _, row in targets.iterrows():
            disease = row["disease_id"]
            gene = row["gene_id"]

            target_index = mappings["gene"][gene]
            scores = score_matrix[
                mappings["disease"][disease]
            ]

            allowed_mask = np.ones(
                number_genes,
                dtype=bool,
            )

            other_known = (
                known_by_disease.get(disease, set())
                - {target_index}
            )

            if other_known:
                allowed_mask[list(other_known)] = False

            candidate_scores = scores[allowed_mask]
            target_score = scores[target_index]

            better = int(
                (candidate_scores > target_score).sum()
            )

            tied = int(
                (candidate_scores == target_score).sum()
            )

            rank = 1 + better + 0.5 * (tied - 1)

            rank_rows.append({
                "disease_id": disease,
                "gene_id": gene,
                "rank": rank,
                "strictly_better_candidates": better,
                "tied_candidates": tied,
                "target_score": target_score,
                "filtered_candidate_count":
                    int(allowed_mask.sum()),
            })

        ranks = pd.DataFrame(rank_rows)
        rank_values = ranks["rank"].to_numpy()

        results.append({
            "setting": (
                "dg_plus_dp"
                if method == "dp_cosine_transfer"
                else "dg_only"
            ),
            "model": method,
            "split": args.split,
            "auroc": roc_auc_score(
                labels, sample_scores
            ),
            "average_precision":
                average_precision_score(
                    labels, sample_scores
                ),
            "hits@10": float(
                np.mean(rank_values <= 10)
            ),
            "hits@20": float(
                np.mean(rank_values <= 20)
            ),
            "mrr": float(
                np.mean(1.0 / rank_values)
            ),
            "mean_rank": float(rank_values.mean()),
            "num_ranking_queries": len(ranks),
            "mean_tied_candidates": float(
                ranks["tied_candidates"].mean()
            ),
            "zero_score_target_fraction": float(
                ranks["target_score"].eq(0).mean()
            ),
        })

        predictions = samples.copy()
        predictions["raw_score"] = sample_scores
        artifacts[method] = (predictions, ranks)

    output.mkdir(parents=True, exist_ok=True)

    report = pd.DataFrame(results)
    report.to_csv(
        output / "baseline_metrics.csv",
        index=False,
    )

    for method, (predictions, ranks) in artifacts.items():
        predictions.to_csv(
            output / f"{method}_{args.split}_predictions.csv",
            index=False,
        )
        ranks.to_csv(
            output / f"{method}_{args.split}_ranks.csv",
            index=False,
        )

    configuration = {
        "split": args.split,
        "candidate_genes": number_genes,
        "score_construction":
            "Training disease-gene links only; "
            "cosine transfer also uses disease phenotypes.",
        "ranking":
            "Filter other known causal pairs; average ties.",
        "deterministic": True,
        "threshold_metrics": "Not computed.",
        "source_hashes": {
            filename: hashlib.sha256(
                (DATA / filename).read_bytes()
            ).hexdigest()
            for filename in [
                "train_edges.csv",
                f"{args.split}_edges.csv",
                f"{args.split}_classification_samples.csv",
                "edges_disease_phenotype.csv",
                "known_causal_pairs_for_filtering.csv",
                "negative_block_pairs.csv",
            ]
        },
    }

    (output / "baseline_config.json").write_text(
        json.dumps(configuration, indent=2),
        encoding="utf-8",
    )

    print(
        f"\n===== STRUCTURAL BASELINES: "
        f"{args.split.upper()} ====="
    )
    print(report.to_string(index=False))
    print("\nSaved to:", output)
    print(
        "Scores constructed from training target links only."
    )
    print(
        "Known-causal pairs used only for ranking filtering."
    )


if __name__ == "__main__":
    main()
