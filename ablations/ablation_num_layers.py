"""Ablation: sensitivity to the number of stacked SiST-GNN layers.

Sweeps $L \\in \\{1, 2, 3, 4\\}$ on a representative dataset. $L=2$ is the
default reported in the main tables. We expect $L=1$ to be markedly weaker
(no two-hop reasoning) and $L \\ge 3$ to over-smooth on the smaller benchmarks.
"""
from _common import RunConfig, parse_dataset_arg, run_one, save_csv

DEFAULT_DATASETS = ["bitcoin-otc", "bitcoin-alpha", "uci-message"]
NUM_LAYERS = [1, 2, 3, 4]
SEEDS = [0, 1, 2]


def main() -> None:
    datasets = parse_dataset_arg(DEFAULT_DATASETS)
    rows = []
    for ds in datasets:
        for L in NUM_LAYERS:
            for seed in SEEDS:
                cfg = RunConfig(
                    dataset     = ds,
                    num_layers  = L,
                    eval_method = "live-update",
                    seed        = seed,
                )
                print(f"[num_layers] dataset={ds} L={L} seed={seed}")
                rows.append(run_one(cfg))
    save_csv(rows, "ablation_num_layers.csv")


if __name__ == "__main__":
    main()
