"""NC ablation: sensitivity to the snapshot width (``bucket_hours``).

JODIE datasets are continuous-time interaction streams; SiST-GNN imposes a
discrete-time decomposition by bucketing into snapshots of fixed temporal
width. This ablation sweeps that width to probe the time-granularity
trade-off (fine buckets → many sparse snapshots; coarse buckets → few dense
ones).

Sweep: bucket_hours ∈ {1, 3, 6, 12, 24} on all NC datasets, 3 seeds each.
"""
from _nc_common import (
    DEFAULT_NC_DATASETS,
    RunConfig,
    parse_dataset_arg,
    run_one,
    save_csv,
)

BUCKET_HOURS = [1.0, 3.0, 6.0, 12.0, 24.0]
SEEDS = [0, 1, 2]


def main() -> None:
    datasets = parse_dataset_arg(DEFAULT_NC_DATASETS)
    rows = []
    for ds in datasets:
        for bh in BUCKET_HOURS:
            for seed in SEEDS:
                cfg = RunConfig(dataset=ds, bucket_hours=bh, seed=seed)
                print(f"[nc-bucket] dataset={ds} bucket_hours={bh} seed={seed}")
                rows.append(run_one(cfg))
                last = rows[-1]
                print(
                    f"   → val_auc={last['val_auc']:.4f}  test_auc={last['test_auc']:.4f}"
                )
    save_csv(rows, "ablation_nc_bucket_hours.csv")


if __name__ == "__main__":
    main()
