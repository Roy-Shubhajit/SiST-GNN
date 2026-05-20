"""NC ablation: BCE positive-class weighting strategy.

Wikipedia/Reddit/MOOC NC labels are highly imbalanced (positives are rare
events: bans, dropouts). The default ``pos_weight='none'`` follows the
TGN / JODIE convention. This ablation compares:

    - 'none'     : plain BCE
    - 'sqrt'     : sqrt(neg/pos) — middle-ground reweighting
    - 'balanced' : neg/pos       — aggressive reweighting

Reported metric: ROC AUC (val + test).
"""
from _nc_common import (
    DEFAULT_NC_DATASETS,
    RunConfig,
    parse_dataset_arg,
    run_one,
    save_csv,
)

POS_WEIGHTS = ["none", "sqrt", "balanced"]
SEEDS = [0, 1, 2]


def main() -> None:
    datasets = parse_dataset_arg(DEFAULT_NC_DATASETS)
    rows = []
    for ds in datasets:
        for pw in POS_WEIGHTS:
            for seed in SEEDS:
                cfg = RunConfig(dataset=ds, pos_weight=pw, seed=seed)
                print(f"[nc-pos_weight] dataset={ds} pos_weight={pw} seed={seed}")
                rows.append(run_one(cfg))
                last = rows[-1]
                print(
                    f"   → val_auc={last['val_auc']:.4f}  test_auc={last['test_auc']:.4f}"
                )
    save_csv(rows, "ablation_nc_pos_weight.csv")


if __name__ == "__main__":
    main()
