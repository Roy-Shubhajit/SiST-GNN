"""Ablation: sensitivity to the hidden dimension $d_h$.

Sweeps $d_h \\in \\{32, 64, 128, 256\\}$ on a representative dataset and writes
the (d_h, MRR) pairs to `results/ablation_hidden_dim.csv` for plotting.
$d_h=128$ is the default reported in the main tables.
"""
from _common import RunConfig, parse_dataset_arg, run_one, save_csv

DEFAULT_DATASETS = ["bitcoin-otc", "bitcoin-alpha", "uci-message"]
HIDDEN_DIMS = [32, 64, 128, 256]
SEEDS = [0, 1, 2]


def main() -> None:
    datasets = parse_dataset_arg(DEFAULT_DATASETS)
    rows = []
    for ds in datasets:
        for d_h in HIDDEN_DIMS:
            for seed in SEEDS:
                cfg = RunConfig(
                    dataset     = ds,
                    hidden_dim  = d_h,
                    eval_method = "live-update",
                    seed        = seed,
                )
                print(f"[hidden_dim] dataset={ds} d_h={d_h} seed={seed}")
                rows.append(run_one(cfg))
    save_csv(rows, "ablation_hidden_dim.csv")


if __name__ == "__main__":
    main()
