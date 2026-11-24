from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "warm_connected"

gnn = pd.read_csv(
    RESULTS / "test_lr001_seed42" / "summary.csv",
    dtype={"split": str},
)

gnn = gnn[gnn["split"] == "test"].copy()

baselines = pd.read_csv(
    RESULTS / "fixed_protocol_controls_test" / "baseline_metrics.csv"
)

baseline_map = {
    "auroc": "auroc_mean",
    "average_precision": "average_precision_mean",
    "hits@10": "hits@10_mean",
    "hits@20": "hits@20_mean",
    "mrr": "mrr_mean",
    "mean_rank": "mean_rank_mean",
}

for old, new in baseline_map.items():
    baselines[new] = baselines[old]

for column in [
    "auroc_std", "average_precision_std", "f1_mean", "f1_std",
    "hits@10_std", "hits@20_std", "mrr_std", "mean_rank_std",
]:
    baselines[column] = float("nan")

columns = [
    "setting", "model", "split", "auroc_mean", "auroc_std",
    "average_precision_mean", "average_precision_std", "f1_mean", "f1_std",
    "hits@10_mean", "hits@10_std", "hits@20_mean", "hits@20_std",
    "mrr_mean", "mrr_std", "mean_rank_mean", "mean_rank_std",
]

combined = pd.concat([gnn[columns], baselines[columns]], ignore_index=True)

output = RESULTS / "fixed_protocol_test_summary.csv"
combined.to_csv(output, index=False)

print(combined.to_string(index=False))
print("\nSaved:", output)
