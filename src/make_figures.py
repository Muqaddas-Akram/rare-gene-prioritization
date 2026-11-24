from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

FIGURES.mkdir(exist_ok=True)

MODEL_ORDER = [
    "gene_degree",
    "dg_three_step_walk",
    "dp_cosine_transfer",
    "id_mlp",
    "graphconv",
    "sage",
    "gat",
]

MODEL_LABELS = {
    "gene_degree": "Gene degree",
    "dg_three_step_walk": "3-step walk",
    "dp_cosine_transfer": "Phenotype cosine",
    "id_mlp": "ID-MLP",
    "graphconv": "GraphConv",
    "sage": "GraphSAGE",
    "gat": "GAT",
}

COLORS = {
    "gene_degree": "#bbbbbb",
    "dg_three_step_walk": "#8c6d31",
    "dp_cosine_transfer": "#d62728",
    "id_mlp": "#9e9e9e",
    "graphconv": "#4c72b0",
    "sage": "#55a868",
    "gat": "#c44e52",
}


def load_summary():
    path = RESULTS / "benchmark_summary.csv"
    df = pd.read_csv(path)

    df["model"] = pd.Categorical(
        df["model"],
        categories=MODEL_ORDER,
        ordered=True,
    )

    df = df.sort_values(
        ["setting", "model"],
        kind="stable",
    )

    return df


def make_metric_figure(df, metric, ylabel, filename):
    mean_column = f"{metric}_mean"
    std_column = f"{metric}_std"

    x = np.arange(len(df))
    means = df[mean_column].to_numpy(dtype=float)
    stds = df[std_column].to_numpy(dtype=float)
    stds = np.nan_to_num(stds, nan=0.0)

    plt.figure(figsize=(10, 5))
    bars = plt.bar(
        x,
        means,
        yerr=stds,
        capsize=4,
        color=[COLORS[model] for model in df["model"]],
        edgecolor="black",
        linewidth=0.8,
    )

    plt.xticks(
        x,
        [MODEL_LABELS[model] for model in df["model"]],
        rotation=35,
        ha="right",
        fontsize=10,
    )
    plt.ylabel(ylabel, fontsize=12)
    plt.title(
        f"Fixed-protocol test comparison: {ylabel}",
        fontsize=13,
        fontweight="bold",
    )
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(FIGURES / filename, dpi=300)
    plt.close()


def make_table(df):
    metric_columns = [
        ("auroc", "AUROC"),
        ("average_precision", "AUPRC"),
        ("hits@10", "Hits@10"),
        ("hits@20", "Hits@20"),
        ("mrr", "MRR"),
    ]

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Fixed-protocol test performance of structural baselines and graph neural network models.}",
        r"\label{tab:fixed_protocol_results}",
        r"\begin{tabular}{llccccc}",
        r"\hline",
        r"Setting & Model & AUROC & AUPRC & Hits@10 & Hits@20 & MRR \\",
        r"\hline",
    ]

    for _, row in df.iterrows():
        model = str(row["model"])
        label = MODEL_LABELS.get(model, model)
        setting = str(row["setting"])

        values = []
        for metric, _ in metric_columns:
            value = row[f"{metric}_mean"]
            values.append(f"{float(value):.3f}")

        lines.append(
            " & ".join(
                [setting, label, *values]
            ) + r" \\"
        )

    lines.extend([
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ])

    (FIGURES / "table_fixed_protocol_results.tex").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main():
    df = load_summary()

    make_metric_figure(
        df,
        "auroc",
        "AUROC",
        "figure_auroc_fixed_protocol.png",
    )

    make_metric_figure(
        df,
        "average_precision",
        "Average precision",
        "figure_auprc_fixed_protocol.png",
    )

    make_metric_figure(
        df,
        "hits@10",
        "Hits@10",
        "figure_hits10_fixed_protocol.png",
    )

    make_metric_figure(
        df,
        "hits@20",
        "Hits@20",
        "figure_hits20_fixed_protocol.png",
    )

    make_metric_figure(
        df,
        "mrr",
        "MRR",
        "figure_mrr_fixed_protocol.png",
    )

    make_table(df)

    print("Saved fixed-protocol figures and LaTeX table.")
    print(FIGURES / "table_fixed_protocol_results.tex")


if __name__ == "__main__":
    main()