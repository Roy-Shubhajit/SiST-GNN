"""Entry point for SiST-GNN dynamic link-prediction experiments.

Loads a Roland-style dataset, builds the SiST-GNN backbone, and runs either the
fixed-split or live-update evaluation. Results are written to
``results/exp_<timestamp>_lp.json``.
"""

from __future__ import annotations

import argparse
import os

import torch

from data_processing_lp import ROLAND_SPEC, load_lp_dataset
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
    p.add_argument("--eval-method",
                   choices=["fixed-split", "live-update", "roland-fixed-split",
                            "roland-live-update"],
                   default="fixed-split",
                   help="'roland-fixed-split' reproduces ROLAND's "
                        "live_update_fixed_split mode: incremental training over "
                        "every snapshot, MRR reported only from the dataset's "
                        "start_compute_mrr index; 'roland-live-update' adds "
                        "their per-snapshot 80/10/10 edge-label split and the "
                        ">=10-edge snapshot filter")
    p.add_argument("--snapshot-mode", choices=["paper", "roland"], default="paper",
                   help="'roland' uses ROLAND's own snapshot construction "
                        "(fixed-width second bins + undirected augmentation "
                        "for bitcoin) instead of weekly calendar bins")

    p.add_argument("--model", choices=["lstm", "transformer"], default="lstm")
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--gnn-type", type=str, default="GCNConv")

    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--num-epochs", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-neg", type=int, default=1000,
                   help="Negative tails per source node at evaluation (ROLAND: 1000)")
    p.add_argument("--train-num-neg", type=int, default=1,
                   help="Negatives per positive during training (eval uses --num-neg)")
    p.add_argument("--dropout", type=float, default=0.0,
                   help="Dropout between layers (paper Sec 5.3 states 0.1)")
    p.add_argument("--weight-decay", type=float, default=0.0,
                   help="Adam weight decay (paper Sec 5.3 states 1e-5)")
    p.add_argument("--decoder", choices=["dot", "mlp"], default="dot",
                   help="'dot' = inner product (paper eq. 8); 'mlp' = MLP on "
                        "[z_u || z_v || z_u * z_v]")
    p.add_argument("--val-frac", type=float, default=0.0,
                   help="Fraction of training tasks held out for model selection "
                        "(0 = off, fixed-split only)")
    p.add_argument("--full-self-temporal", action="store_true",
                   help="Emit one (i+N, i) self-temporal edge per NODE per "
                        "Algorithm 1, instead of one per edge at the source")
    p.add_argument("--train-target", choices=["next", "current"], default="next",
                   help="Supervision only; evaluation is always the ROLAND task "
                        "(encode G_t, score E_{t+1}). 'next' = train on E_{t+1} "
                        "(objective matches evaluation); 'current' = train on E_t "
                        "(reconstruction) and test on E_{t+1} (forecasting)")
    p.add_argument("--node-features", choices=["learned", "constant"],
                   default="learned",
                   help="'learned' = free per-node embedding table (paper Sec 4.5); "
                        "'constant' = all-ones features, matching ROLAND's "
                        "include_node_features=false / AS_node_feature='one'")

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
            snapshots = load_lp_dataset(ds_name, data_root=args.data_root,
                                        snapshot_mode=args.snapshot_mode)
            num_nodes = snapshots[0].num_nodes

            # With constant features the encoder consumes the datasets' 1-dim
            # all-ones x directly; with learned features it consumes the
            # embedding table, which is hidden_dim wide.
            use_node_emb = args.node_features == "learned"
            input_dim = args.hidden_dim if use_node_emb else snapshots[0].x.size(1)

            if args.model == "lstm":
                base = TemporalGNNModel(
                    input_dim=input_dim,
                    hidden_dim=args.hidden_dim,
                    output_dim=args.hidden_dim,
                    num_layers=args.num_layers,
                    gnn_type=args.gnn_type,
                    dropout=args.dropout,
                    full_self_temporal=args.full_self_temporal,
                )
            else:
                base = TemporalTransformerGNNModel(
                    input_dim=input_dim,
                    hidden_dim=args.hidden_dim,
                    output_dim=args.hidden_dim,
                    num_layers=args.num_layers,
                    gnn_type=args.gnn_type,
                    dropout=args.dropout,
                )

            model = StatefulGNNWrapper(
                base, num_nodes, args.device,
                use_node_emb=use_node_emb, node_emb_dim=args.hidden_dim,
                decoder=args.decoder, out_dim=args.hidden_dim,
            ).to(args.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                         weight_decay=args.weight_decay)

            if args.eval_method == "fixed-split":
                res = train_fixed_split(snapshots, model, optimizer,
                                        num_epochs=args.num_epochs,
                                        num_neg=args.num_neg,
                                        train_target=args.train_target,
                                        train_num_neg=args.train_num_neg,
                                        val_frac=args.val_frac)
            else:
                # ROLAND's table-2 "fixed-split" is a live-update loop that only
                # starts reporting MRR at the dataset's start_compute_mrr index.
                smrr, edge_split, min_edges = 0, None, 0
                if args.eval_method == "roland-fixed-split":
                    spec = ROLAND_SPEC.get(ds_name)
                    if spec is None:
                        raise ValueError(
                            f"no ROLAND table-2 spec for {ds_name}; "
                            "roland-fixed-split is defined for "
                            f"{sorted(ROLAND_SPEC)}")
                    smrr = spec["start_compute_mrr"]
                    print(f"  reporting MRR from snapshot {smrr} of {len(snapshots)}")
                elif args.eval_method == "roland-live-update":
                    # table-3 configs: split_method 'default', split [.8,.1,.1],
                    # and loaders keep only snapshots with >= 10 edges.
                    edge_split, min_edges = (0.8, 0.1, 0.1), 10
                    print("  ROLAND live-update: per-snapshot 80/10/10 edge-label "
                          "split, MRR on the test share; snapshots < 10 edges dropped")
                res = train_live_update(snapshots, model, optimizer,
                                        epochs_per_snapshot=args.num_epochs,
                                        num_neg=args.num_neg,
                                        train_target=args.train_target,
                                        train_num_neg=args.train_num_neg,
                                        start_compute_mrr=smrr,
                                        edge_split=edge_split,
                                        min_edges=min_edges,
                                        seed=args.seed)

            results[ds_name] = res
            print(f"✓ {ds_name} Complete (MRR={res['mrr']:.6f})")
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
