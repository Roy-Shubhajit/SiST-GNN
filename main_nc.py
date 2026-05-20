"""
Entry point for dynamic node classification experiments.

Mirrors ``main_lp.py`` but targets Wikipedia / Reddit / MOOC via the benchtemp pipeline.
"""

from __future__ import annotations

import argparse
import os

import torch

from data_processing_nc import load_nc_dataset
from helper import now_tag, save_json, seed_everything
from temporal_gnn import TemporalGNNModel, TemporalTransformerGNNModel
from train_nc import NCStatefulWrapper, train_nc_fixed_split

DEFAULT_DATASETS = ["wikipedia", "reddit", "mooc"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SiST-GNN dynamic node classification experiments")
    p.add_argument("--dataset", type=str, default="", help="Single dataset; empty = all")
    p.add_argument("--data-root", type=str, default="datasets/nc")
    p.add_argument("--device", type=str, default="auto")

    p.add_argument("--model", choices=["lstm", "transformer"], default="lstm")
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--gnn-type", type=str, default="GCNConv")

    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--num-epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument(
        "--bucket-hours",
        type=float,
        default=6.0,
        help=(
            "Snapshot width in hours. JODIE datasets are continuous-time "
            "streams; the snapshot count is derived from the dataset's span."
        ),
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--no-node-emb",
        action="store_true",
        help="Disable learnable per-node embedding (use static node_features only).",
    )
    p.add_argument(
        "--no-edge-emb",
        action="store_true",
        help="Disable the edge encoder; the GNN runs unweighted and the head sees only src_emb.",
    )
    p.add_argument(
        "--pos-weight",
        type=str,
        default="none",
        help=(
            "BCE positive-class weight strategy: 'none' (plain BCE, default), "
            "'balanced' (neg/pos), 'sqrt' (sqrt(neg/pos)), or a numeric value."
        ),
    )

    p.add_argument("--out-dir", type=str, default="results")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n" + "=" * 70)
    print(" " * 9 + "🚀 SiST-GNN Dynamic Node Classification Experiments")
    print("=" * 70)
    print(f"Device       : {args.device}")
    print(f"Backbone     : {args.model}    GNN: {args.gnn_type}    Layers: {args.num_layers}")
    print(f"Hidden dim   : {args.hidden_dim}    Bucket: {args.bucket_hours}h")
    print("=" * 70 + "\n")

    datasets = [args.dataset] if args.dataset else DEFAULT_DATASETS

    payload = {
        "timestamp": now_tag(),
        "task": "node_classification",
        "config": vars(args),
        "results": {},
    }

    for ds_name in datasets:
        print(f"\n--- Running {ds_name} ---")
        try:
            bundle = load_nc_dataset(
                ds_name, data_root=args.data_root, bucket_hours=args.bucket_hours
            )

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

            model = NCStatefulWrapper(
                base_model=base,
                num_nodes=bundle.num_nodes,
                hidden_dim=args.hidden_dim,
                node_features=bundle.node_features,
                edge_feat_dim=bundle.edge_feat_dim,
                device=args.device,
                use_node_emb=not args.no_node_emb,
                use_edge_emb=not args.no_edge_emb,
                classifier_hidden=args.hidden_dim,
            ).to(args.device)

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
            )
            payload["results"][ds_name] = metrics
            print(f"✓ {ds_name} done — Test AUC={metrics['test_auc']:.4f}")
        except Exception as e:  # noqa: BLE001
            print(f"Failed on {ds_name}: {e}")
            payload["results"][ds_name] = {"error": str(e)}

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = f"{args.out_dir}/exp_{payload['timestamp']}_nc.json"
    save_json(out_path, payload)
    print("\n" + "=" * 70)
    print(f"📁 Results saved to: {out_path}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
