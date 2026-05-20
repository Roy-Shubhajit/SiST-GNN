"""NC ablation: choice of GNN backbone (GCN vs. SAGE vs. GATv2).

All other hyper-parameters are held at the ``main_nc.py`` defaults
(hidden_dim=128, num_layers=2, bucket_hours=6, pos_weight='none', seeds {0,1,2}).
Writes one row per (dataset, backbone, seed) to
``results/ablation_nc_gnn_backbone.csv``.
"""
from _nc_common import (
    DEFAULT_NC_DATASETS,
    RunConfig,
    parse_dataset_arg,
    run_one,
    save_csv,
)

BACKBONES = ["GCNConv", "SAGEConv", "GATv2Conv"]
SEEDS = [0, 1, 2]


def main() -> None:
    datasets = parse_dataset_arg(DEFAULT_NC_DATASETS)
    rows = []
    for ds in datasets:
        for backbone in BACKBONES:
            for seed in SEEDS:
                cfg = RunConfig(dataset=ds, gnn_type=backbone, seed=seed)
                print(f"[nc-backbone] dataset={ds} backbone={backbone} seed={seed}")
                rows.append(run_one(cfg))
                last = rows[-1]
                print(
                    f"   → val_auc={last['val_auc']:.4f}  test_auc={last['test_auc']:.4f}"
                )
    save_csv(rows, "ablation_nc_gnn_backbone.csv")


if __name__ == "__main__":
    main()
