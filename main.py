import argparse
import torch
import json
from dataclasses import asdict
from typing import Dict, List
import os

from helper import now_tag, save_json
from data_processing_roland import load_roland_dataset
from train_lp_roland import train_live_update, train_fixed_split, StatefulGNNWrapper
from temporal_gnn import TemporalGNNModel, TemporalTransformerGNNModel

DEFAULT_DATASETS = ["bitcoin-alpha", "bitcoin-otc", "uci-message", "reddit-body", "reddit-title", "as-733", "bsi-zk", "bsi-svt"]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Roland dynamic graph experiments for LP")
    parser.add_argument("--dataset", type=str, default="", help="Specific dataset to run. Runs all default if empty.")
    parser.add_argument("--data-root", type=str, default="datasets/roland")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--eval-method", choices=["fixed-split", "live-update"], default="fixed-split", help="Evaluation split method")
    
    parser.add_argument("--model", choices=["lstm", "transformer"], default="lstm")
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--gnn-type", type=str, default="GCNConv")
    
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-epochs", type=int, default=50)

    parser.add_argument("--out-dir", type=str, default="results")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    if torch.cuda.is_available() and args.device == "auto":
        args.device = "cuda"
    else:
        args.device = "cpu"
        
    print("\n" + "=" * 70)
    print(" " * 15 + "🚀 Roland Temporal GNN Experiment Framework")
    print("=" * 70)
    print(f"Eval Method: {args.eval_method.upper():15}")
    print(f"Device: {args.device:20}")
    print("=" * 70 + "\n")

    final_payload = {
        "timestamp": now_tag(),
        "eval_method": args.eval_method,
        "results": {},
    }

    datasets = [args.dataset] if args.dataset else DEFAULT_DATASETS
    print(f"\n📊 Link Prediction Tasks ({len(datasets)} dataset{'s' if len(datasets) != 1 else ''})")
    print("-" * 70)
    
    results = {}
    for ds_name in datasets:
        print(f"\n--- Running {ds_name} ---")
        try:
            snapshots = load_roland_dataset(ds_name, data_root=args.data_root)
            num_nodes = snapshots[0].num_nodes
            
            if args.model == "lstm":
                base_model = TemporalGNNModel(
                    input_dim=args.hidden_dim, 
                    hidden_dim=args.hidden_dim, 
                    output_dim=args.hidden_dim, 
                    num_layers=args.num_layers,
                    gnn_type=args.gnn_type
                )
            else:
                base_model = TemporalTransformerGNNModel(
                    input_dim=args.hidden_dim,
                    hidden_dim=args.hidden_dim,
                    output_dim=args.hidden_dim,
                    num_layers=args.num_layers,
                    gnn_type=args.gnn_type
                )
                
            model = StatefulGNNWrapper(base_model, num_nodes, args.device, use_node_emb=True, node_emb_dim=args.hidden_dim).to(args.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            
            if args.eval_method == "fixed-split":
                mrr = train_fixed_split(snapshots, model, optimizer, num_epochs=args.num_epochs)
            else:
                mrr = train_live_update(snapshots, model, optimizer, epochs_per_snapshot=args.num_epochs)
                
            results[ds_name] = {"mrr": float(mrr)}
            print(f"✓ {ds_name} Complete (MRR={mrr:.6f})")
        except Exception as e:
            print(f"Failed on {ds_name}: {e}")
            results[ds_name] = {"error": str(e)}

    final_payload["results"]["lp"] = results

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = f"{args.out_dir}/exp_{final_payload['timestamp']}_roland.json"
    save_json(out_path, final_payload)
    
    print("\n" + "=" * 70)
    print("🎉 All experiments completed successfully!")
    print(f"📁 Results saved to: {out_path}")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
