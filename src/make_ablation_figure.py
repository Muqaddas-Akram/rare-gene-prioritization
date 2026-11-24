import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

RESULTS = "results"
FIGURES = "figures"

os.makedirs(FIGURES, exist_ok=True)

ABLATION_LABELS = {
    "full": "Full graph",
    "no_disease_phenotype": "No disease–phenotype",
    "no_gene_phenotype": "No gene–phenotype",
    "gene_disease_only": "Gene–disease only"
}

ABLATION_ORDER = [
    "full",
    "no_disease_phenotype",
    "no_gene_phenotype",
    "gene_disease_only"
]

COLORS = {
    "full": "#55a868",
    "no_disease_phenotype": "#4c72b0",
    "no_gene_phenotype": "#c44e52",
    "gene_disease_only": "#9e9e9e"
}


def load_ablation_summary():
    path = f"{RESULTS}/ablation_summary.csv"
    df = pd.read_csv(path)

    df["ablation"] = pd.Categorical(
        df["ablation"],
        categories=ABLATION_ORDER,
        ordered=True
    )

    df = df.sort_values("ablation")

    return df


def make_ablation_figure(df):
    metrics = [
        ("auroc", "AUROC"),
        ("auprc", "AUPRC"),
        ("f1", "F1-score")
    ]

    x = np.arange(len(df))
    width = 0.25

    plt.figure(figsize=(9, 5))

    for i, (metric, label) in enumerate(metrics):
        means = df[f"{metric}_mean"].values
        stds = df[f"{metric}_std"].values

        plt.bar(
            x + (i - 1) * width,
            means,
            width,
            yerr=stds,
            capsize=4,
            label=label,
            edgecolor="black",
            linewidth=0.8
        )

    plt.xticks(
        x,
        [ABLATION_LABELS[a] for a in df["ablation"]],
        fontsize=10,
        rotation=15,
        ha="right"
    )

    plt.ylabel("Score", fontsize=12)
    plt.title(
        "Ablation study: contribution of phenotype information",
        fontsize=13,
        fontweight="bold"
    )

    plt.legend(frameon=False)
    plt.grid(
        axis="y",
        linestyle="--",
        alpha=0.4
    )

    plt.gca().set_axisbelow(True)

    for spine in ["top", "right"]:
        plt.gca().spines[spine].set_visible(False)

    plt.tight_layout()
    plt.savefig(
        f"{FIGURES}/figure_ablation_study.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    print(f"Saved: {FIGURES}/figure_ablation_study.png")


def make_ablation_latex_table(df):
    metrics = [
        ("accuracy", "Accuracy"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1-score"),
        ("auroc", "AUROC"),
        ("auprc", "AUPRC")
    ]

    lines = []

    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append(
        "\\caption{Ablation study of phenotype-based graph edges using the GraphSAGE model. Values are mean $\\pm$ standard deviation over three random seeds.}"
    )
    lines.append("\\label{tab:ablation_results}")
    lines.append("\\begin{tabular}{lcccccc}")
    lines.append("\\toprule")
    lines.append(
        "Graph setting & Accuracy & Precision & Recall & F1 & AUROC & AUPRC \\\\"
    )
    lines.append("\\midrule")

    for _, row in df.iterrows():
        setting = ABLATION_LABELS[row["ablation"]]

        values = []

        for metric, _ in metrics:
            mean = row[f"{metric}_mean"]
            std = row[f"{metric}_std"]
            values.append(f"{mean:.3f} $\\pm$ {std:.3f}")

        lines.append(
            f"{setting} & " + " & ".join(values) + " \\\\"
        )

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    latex = "\n".join(lines)

    with open(
        f"{FIGURES}/table_ablation_results.tex",
        "w",
        encoding="utf-8"
    ) as f:
        f.write(latex)

    print(f"Saved: {FIGURES}/table_ablation_results.tex")


def main():
    df = load_ablation_summary()

    make_ablation_figure(df)
    make_ablation_latex_table(df)


if __name__ == "__main__":
    main()