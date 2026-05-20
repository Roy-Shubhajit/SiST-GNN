"""NC ablation: sensitivity to the hidden dimension $d_h$.

Sweeps $d_h \\in \\{32, 64, 128, 256\\}$ on the JODIE node-classification
datasets and writes ``(d_h, val_auc, test_auc)`` triples for plotting.
$d_h=128$ is the default reported in the main NC table.
"""
from _nc_common import (
    DEFAULT_NC_DATASETS,
    RunConfig,
    parse_dataset_arg,
    run_one,
    save_csv,
)

HIDDEN_DIMS = [32, 64, 128, 256]
SEEDS = [0, 1, 2]


def main() -> None:
    datasets = parse_dataset_arg(DEFAULT_NC_DATASETS)
    rows = []
    for ds in datasets:
        for d_h in HIDDEN_DIMS:
            for seed in SEEDS:
                # The classifier MLP width follows hidden_dim, matching main_nc.py.
                cfg = RunConfig(
                    dataset=ds,
                    hidden_dim=d_h,
                    classifier_hidden=d_h,
                    seed=seed,
                )
                print(f"[nc-hidden_dim] dataset={ds} d_h={d_h} seed={seed}")
                rows.append(run_one(cfg))
                last = rows[-1]
                print(
                    f"   → val_auc={last['val_auc']:.4f}  test_auc={last['test_auc']:.4f}"
                )
    save_csv(rows, "ablation_nc_hidden_dim.csv")


if __name__ == "__main__":
    main()
