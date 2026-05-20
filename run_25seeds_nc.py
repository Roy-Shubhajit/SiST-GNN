"""
25-seed sweep for SiST-GNN node classification (wikipedia / reddit / mooc).

Loads each JODIE dataset once and replays the train_nc_fixed_split pipeline
across 25 different seeds. Per-run JSONs land in results/nc_25seeds/, and
the aggregated mean+/-std summary lands in results/nc_25seeds/summary.json.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from statistics import mean, stdev

import torch

from data_processing_nc import load_nc_dataset
from helper import save_json, seed_everything
from temporal_gnn import TemporalGNNModel
from train_nc import NCStatefulWrapper, train_nc_fixed_split

DATASETS = ["wikipedia", "reddit", "mooc"]
DEFAULT_SEEDS = list(range(25))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default="datasets/nc")
    p.add_argument("--out-dir", type=str, default="results/nc_25seeds")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--gnn-type", type=str, default="SAGEConv")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--bucket-hours", type=float, default=3.0)
    p.add_argument("--pos-weight", type=str, default="balanced")
    p.add_argument("--num-epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--num-seeds", type=int, default=25)
    p.add_argument("--datasets", type=str, default=",".join(DATASETS))
    return p.parse_args()


def build_and_train(bundle, args, device):
    base = TemporalGNNModel(
        input_dim=args.hidden_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.hidden_dim,
        num_layers=args.num_layers,
        gnn_type=args.gnn_type,
    )
    model = NCStatefulWrapper(
        base_model=base,
        num_nodes=bundle.num_nodes,
        hidden_dim=args.hidden_dim,
        node_features=bundle.node_features,
        edge_feat_dim=bundle.edge_feat_dim,
        device=device,
        use_node_emb=True,
        use_edge_emb=True,
        classifier_hidden=args.hidden_dim,
    ).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    metrics = train_nc_fixed_split(
        snapshots=bundle.snapshots,
        model=model,
        optimizer=optim,
        train_end=bundle.train_end_snap,
        val_end=bundle.val_end_snap,
        num_epochs=args.num_epochs,
        patience=args.patience,
        pos_weight=args.pos_weight,
        verbose=False,
    )
    del model, base, optim
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return metrics


def main() -> None:
    args = parse_args()
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = args.device

    os.makedirs(args.out_dir, exist_ok=True)
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    seeds = list(range(args.num_seeds))

    print("=" * 72)
    print("  SiST-GNN  |  25-seed node classification sweep")
    print(f"  device={device}  gnn={args.gnn_type}  hidden={args.hidden_dim}  "
          f"bucket={args.bucket_hours}h  pos_weight={args.pos_weight}")
    print(f"  datasets={datasets}  seeds={seeds[0]}..{seeds[-1]}  ({len(seeds)} seeds)")
    print("=" * 72)

    summary = {
        "config": vars(args),
        "datasets": {},
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    for ds_name in datasets:
        print(f"\n[{ds_name}] loading bundle ...")
        bundle = load_nc_dataset(
            ds_name, data_root=args.data_root, bucket_hours=args.bucket_hours
        )
        print(f"[{ds_name}] num_nodes={bundle.num_nodes} "
              f"snaps={len(bundle.snapshots)} "
              f"train_end={bundle.train_end_snap} val_end={bundle.val_end_snap}")

        per_seed = []
        for seed in seeds:
            seed_everything(seed)
            t0 = time.time()
            try:
                metrics = build_and_train(bundle, args, device)
                metrics["seed"] = seed
                metrics["elapsed_s"] = time.time() - t0
                per_seed.append(metrics)
                print(f"  seed={seed:02d}  val_auc={metrics['val_auc']:.4f}  "
                      f"test_auc={metrics['test_auc']:.4f}  "
                      f"({metrics['elapsed_s']:.1f}s)")
            except Exception as e:  # noqa: BLE001
                print(f"  seed={seed:02d}  FAILED: {e}")
                traceback.print_exc()
                per_seed.append({"seed": seed, "error": str(e)})

            # Persist incrementally so a long run is recoverable.
            per_run_path = os.path.join(args.out_dir, f"{ds_name}_seed{seed:02d}.json")
            save_json(per_run_path, {
                "dataset": ds_name,
                "seed": seed,
                "config": vars(args),
                "metrics": per_seed[-1],
            })

        test_aucs = [m["test_auc"] for m in per_seed if "test_auc" in m]
        val_aucs = [m["val_auc"] for m in per_seed if "val_auc" in m]
        summary["datasets"][ds_name] = {
            "n_seeds": len(test_aucs),
            "test_auc_mean": mean(test_aucs) if test_aucs else None,
            "test_auc_std": stdev(test_aucs) if len(test_aucs) > 1 else 0.0,
            "val_auc_mean": mean(val_aucs) if val_aucs else None,
            "val_auc_std": stdev(val_aucs) if len(val_aucs) > 1 else 0.0,
            "per_seed": per_seed,
        }
        print(f"[{ds_name}]  TEST AUC = "
              f"{summary['datasets'][ds_name]['test_auc_mean']:.4f} "
              f"+/- {summary['datasets'][ds_name]['test_auc_std']:.4f}  "
              f"(n={len(test_aucs)})")

        # Persist the running summary after each dataset.
        save_json(os.path.join(args.out_dir, "summary.json"), summary)

        del bundle
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    summary["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_json(os.path.join(args.out_dir, "summary.json"), summary)
    print("\nDone. summary -> " + os.path.join(args.out_dir, "summary.json"))


if __name__ == "__main__":
    main()
