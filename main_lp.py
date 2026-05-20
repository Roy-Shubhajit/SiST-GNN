"""Entry point for SiST-GNN dynamic link-prediction experiments.

Loads a Roland-style dataset, builds the SiST-GNN backbone, and runs either the
fixed-split or live-update evaluation. Results are written to
``results/exp_<timestamp>_lp.json``.
"""

from __future__ import annotations

import argparse
import os

import torch

from data_processing_lp import load_lp_dataset
from helper import now_tag, save_json, seed_everything
from temporal_gnn import TemporalGNNModel, TemporalTransformerGNNModel
from train_lp import StatefulGNNWrapper, train_fixed_split, train_live_update

DEFAULT_DATASETS = [
    "bitcoin-alpha", "bitcoin-otc", "uci-message",
    "reddit-body", "reddit-title", "as-733",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SiST-GNN dynamic link-prediction experiments")
    p.add_argument("--dataset", type=str, default="", help="Specific dataset; empty = all defaults")
    p.add_argument("--data-root", type=str, default="datasets/roland")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--eval-method", choices=["fixed-split", "live-update"], default="fixed-split")

    p.add_argument("--model", choices=["lstm", "transformer"], default="lstm")
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--gnn-type", type=str, default="GCNConv")

    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--num-epochs", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--out-dir", type=str, default="results")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n" + "=" * 70)
    print(" " * 12 + "SiST-GNN  |  Dynamic Link Prediction Experiments")
    print("=" * 70)
    print(f"Eval Method  : {args.eval_method}")
    print(f"Backbone     : {args.model}    GNN: {args.gnn_type}    Layers: {args.num_layers}")
    print(f"Hidden dim   : {args.hidden_dim}    LR: {args.lr}    Epochs: {args.num_epochs}")
    print(f"Device       : {args.device}    Seed: {args.seed}")
    print("=" * 70 + "\n")

    datasets = [args.dataset] if args.dataset else DEFAULT_DATASETS

    payload = {
        "timestamp": now_tag(),
        "task": "link_prediction",
        "eval_method": args.eval_method,
        "config": vars(args),
        "results": {"lp": {}},
    }
    results = payload["results"]["lp"]

    for ds_name in datasets:
        print(f"\n--- Running {ds_name} ---")
        try:
            snapshots = load_lp_dataset(ds_name, data_root=args.data_root)
            num_nodes = snapshots[0].num_nodes

            if args.model == "lstm":
                base = TemporalGNNModel(
                    input_dim=args.hidden_dim,
                    hidden_dim=args.hidden_dim,
                    output_dim=args.hidden_dim,
                    num_layers=args.num_layers,
                    gnn_type=args.gnn_type,
                )
            else:
                base = TemporalTransformerGNNModel(
                    input_dim=args.hidden_dim,
                    hidden_dim=args.hidden_dim,
                    output_dim=args.hidden_dim,
                    num_layers=args.num_layers,
                    gnn_type=args.gnn_type,
                )

            model = StatefulGNNWrapper(
                base, num_nodes, args.device,
                use_node_emb=True, node_emb_dim=args.hidden_dim,
            ).to(args.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

            if args.eval_method == "fixed-split":
                mrr = train_fixed_split(snapshots, model, optimizer, num_epochs=args.num_epochs)
            else:
                mrr = train_live_update(snapshots, model, optimizer, epochs_per_snapshot=args.num_epochs)

            results[ds_name] = {"mrr": float(mrr)}
            print(f"✓ {ds_name} Complete (MRR={mrr:.6f})")
        except Exception as e:  # noqa: BLE001
            print(f"Failed on {ds_name}: {e}")
            results[ds_name] = {"error": str(e)}

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = f"{args.out_dir}/exp_{payload['timestamp']}_lp.json"
    save_json(out_path, payload)
    print("\n" + "=" * 70)
    print(f"Results saved to: {out_path}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
