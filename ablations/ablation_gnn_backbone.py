"""Ablation: choice of GNN backbone inside the augmented-graph convolution.

Compares GCNConv (default), GATConv and SAGEConv on the same three datasets.
The paper claims that the qualitative ordering of methods is preserved under
backbone swaps; this script produces the numbers behind that claim.
"""
from _common import RunConfig, parse_dataset_arg, run_one, save_csv

DEFAULT_DATASETS = ["bitcoin-otc", "bitcoin-alpha", "uci-message"]
BACKBONES = ["GCNConv", "GATConv", "SAGEConv"]
SEEDS = [0, 1, 2]


def main() -> None:
    datasets = parse_dataset_arg(DEFAULT_DATASETS)
    rows = []
    for ds in datasets:
        for backbone in BACKBONES:
            for seed in SEEDS:
                cfg = RunConfig(
                    dataset     = ds,
                    gnn_type    = backbone,
                    eval_method = "live-update",
                    seed        = seed,
                )
                print(f"[backbone] dataset={ds} backbone={backbone} seed={seed}")
                rows.append(run_one(cfg))
    save_csv(rows, "ablation_gnn_backbone.csv")


if __name__ == "__main__":
    main()
