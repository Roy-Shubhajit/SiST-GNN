"""Run our model inside ROLAND's pipeline.

The data pipeline (roland_pipeline.py) and the loop below are ports of
snap-stanford/roland. The only substitution is the encoder: our
TemporalGNNModel replaces their GNNStack, and our margin-ranking objective
replaces their BCE (the paper's eq. 9).

Loop ported from graphgym/contrib/train/train_live_update.py:250-339 --
for each task (t, t+1):
  1. evaluate on the TEST copy, before any gradient step on t+1
  2. train on the TRAIN copy for a fixed budget of epochs
  3. roll the recurrent state forward over G_t
MRR is suppressed while t+1 < start_compute_mrr (their ``fast`` flag).
"""

from __future__ import annotations

import argparse
import os
from typing import List

import numpy as np
import torch

from evaluate_lp import get_negative_samples, per_source_mrr
from helper import now_tag, save_json, seed_everything
from roland_pipeline import SplitSnapshot, load_roland, to_data
from temporal_gnn import TemporalGNNModel
from train_lp import StatefulGNNWrapper, model_states_copy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True)
    p.add_argument("--mode",
                   choices=["live_update", "live_update_fixed_split",
                            "paper_fixed_split"],
                   default="live_update",
                   help="'live_update_fixed_split' = table-2 setting as their "
                        "code runs it; 'paper_fixed_split' = the same setting "
                        "as the paper describes it (Table 1 weekly/daily bins, "
                        "directed, all edges, last 10%% of snapshots)")
    p.add_argument("--data-root", default="datasets/roland")
    p.add_argument("--undirected", choices=["auto", "yes", "no"], default="auto",
                   help="'auto' = ROLAND's own default for the mode (bitcoin is "
                        "symmetrised for the fixed-split table only). Set "
                        "explicitly to use one convention across both settings.")
    p.add_argument("--min-edges", type=int, default=-1,
                   help="drop snapshots with fewer edges; -1 = ROLAND's default "
                        "(10 for live_update, 0 otherwise). Match this across "
                        "settings to keep the snapshot sequences identical.")
    p.add_argument("--device", default="auto")

    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--gnn-type", default="GCNConv")
    p.add_argument("--decoder", choices=["dot", "mlp"], default="dot")
    p.add_argument("--node-features", choices=["learned", "constant"],
                   default="constant",
                   help="ROLAND's configs set include_node_features=false, i.e. "
                        "constant all-ones features")
    p.add_argument("--edge-features", choices=["none", "raw", "learned"],
                   default="learned",
                   help="'none' ignores edge attributes; 'raw' feeds ROLAND's "
                        "raw attributes ([rating, scaled time] for bitcoin, "
                        "[scaled time] elsewhere) straight in as GCN edge "
                        "weights; 'learned' passes them through a two-layer MLP "
                        "first, the role of ROLAND's roland_general encoder")
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--weight-decay", type=float, default=0.0)

    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-epochs", type=int, default=100,
                   help="inner epochs per snapshot")
    p.add_argument("--train-num-neg", type=int, default=1)
    p.add_argument("--loss", choices=["margin", "bce"], default="margin",
                   help="'margin' = paper eq. 9 margin-ranking; 'bce' = "
                        "ROLAND's BCEWithLogitsLoss over positives+negatives")
    p.add_argument("--num-neg", type=int, default=1000,
                   help="negatives per source at evaluation (ROLAND: 1000)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="results/roland_port")
    return p.parse_args()


def _loss(model, z, target_ei, num_nodes, k, kind="margin"):
    """Training objective on the supervision edges.

    ``margin`` -- the paper's eq. 9: margin-ranking with unit margin, each
    positive contrasted against its own sampled negatives.

    ``bce`` -- ROLAND's objective. graphgym/loss.py:38-39 applies
    ``BCEWithLogitsLoss`` to the concatenated positive/negative scores, with
    ``edge_label`` as the 1/0 target; DeepSNAP supplies one negative per
    positive by default (``edge_negative_sampling_ratio: 1.0``).
    """
    pos = model.score_edges(z, target_ei[0], target_ei[1])
    neg_ei = get_negative_samples(target_ei, num_nodes, num_neg=k,
                                  forbidden=target_ei)
    neg = model.score_edges(z, neg_ei[0], neg_ei[1])

    if kind == "bce":
        logits = torch.cat([pos, neg])
        labels = torch.cat([torch.ones_like(pos), torch.zeros_like(neg)])
        return torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)

    if k > 1:
        pos = pos.repeat_interleave(k)
    return torch.nn.functional.margin_ranking_loss(
        pos, neg, torch.ones_like(pos), margin=1.0)


def run(args) -> dict:
    device = torch.device(args.device)
    (train_seq, _val_seq, test_seq), smrr = load_roland(
        args.dataset, args.mode, args.data_root, seed=args.seed,
        undirected={"auto": None, "yes": True, "no": False}[args.undirected],
        min_edges=None if args.min_edges < 0 else args.min_edges)
    T = len(train_seq)
    num_nodes = train_seq[0].num_nodes
    print(f"  T={T}  N={num_nodes}  start_compute_mrr={smrr}")

    edge_dim = (train_seq[0].edge_feature.size(1)
                if train_seq[0].edge_feature is not None else 0)
    use_emb = args.node_features == "learned"
    base = TemporalGNNModel(
        input_dim=args.hidden_dim if use_emb else 1,
        hidden_dim=args.hidden_dim, output_dim=args.hidden_dim,
        num_layers=args.num_layers, gnn_type=args.gnn_type,
        dropout=args.dropout)
    model = StatefulGNNWrapper(
        base, num_nodes, device, use_node_emb=use_emb,
        node_emb_dim=args.hidden_dim, decoder=args.decoder,
        out_dim=args.hidden_dim,
        edge_encoder_dim=edge_dim if args.edge_features == "learned" else 0,
    ).to(device)
    print(f"  edge features: {args.edge_features} (raw dim {edge_dim})")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)

    use_ef = args.edge_features != "none"

    mrrs: List[float] = []
    recalls: List[dict] = []
    per_snapshot: List[dict] = []

    for t in range(T - 1):
        # Encoder input is snapshot t of the relevant copy; the supervision /
        # evaluation targets come from snapshot t+1 of that same copy.
        mp_test = to_data(test_seq[t], device, use_ef)
        tgt_test = test_seq[t + 1].edge_label_index.to(device)
        mp_train = to_data(train_seq[t], device, use_ef)
        tgt_train = train_seq[t + 1].edge_label_index.to(device)

        # (1) evaluate first -- the model has seen nothing of t+1
        if (t + 1) >= smrr and tgt_test.size(1) > 0 and mp_test.edge_index.size(1) > 0:
            model.eval()
            with torch.no_grad():
                z = model(mp_test.x, mp_test.edge_index, mp_test.edge_attr,
                          update_state=False)
                mrr, rec = per_source_mrr(
                    (lambda s, d: model.score_edges(z, s, d)),
                    tgt_test, num_nodes, args.num_neg, forbidden=tgt_test)
            mrrs.append(mrr); recalls.append(rec)
            per_snapshot.append({"t": t + 1, "mrr": mrr})
            print(f"Snapshot {t+1:04d} | MRR {mrr:.4f}", flush=True)

        # (2) reveal the ground truth, fine-tune on the TRAIN copy
        if mp_train.edge_index.size(1) > 0:
            prev = model.clone_states()
            if tgt_train.size(1) > 0:
                model.train()
                for _ in range(args.num_epochs):
                    model.load_states(*model_states_copy(*prev))
                    opt.zero_grad()
                    z = model(mp_train.x, mp_train.edge_index, mp_train.edge_attr,
                              update_state=True)
                    _loss(model, z, tgt_train, num_nodes,
                          args.train_num_neg, args.loss).backward()
                    model.detach_states()
                    opt.step()
            # (3) roll the state forward over G_t with the updated weights
            model.load_states(*model_states_copy(*prev))
            with torch.no_grad():
                model.eval()
                model(mp_train.x, mp_train.edge_index, mp_train.edge_attr,
                      update_state=True)
                model.detach_states()

    out = {"mrr": float(np.mean(mrrs)) if mrrs else 0.0,
           "num_eval_snapshots": len(mrrs),
           "per_snapshot_mrr": per_snapshot}
    for k in (1, 3, 10):
        out[f"recall@{k}"] = float(np.mean([r[k] for r in recalls])) if recalls else 0.0
    return out


def main() -> None:
    args = parse_args()
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_everything(args.seed)

    print(f"\n{'='*66}\nROLAND pipeline + SiST-GNN encoder\n"
          f"  dataset={args.dataset}  mode={args.mode}  decoder={args.decoder}\n"
          f"  node_features={args.node_features}  loss={args.loss}  seed={args.seed}\n{'='*66}")

    res = run(args)
    payload = {"timestamp": now_tag(), "task": "link_prediction",
               "pipeline": "roland_port", "eval_method": args.mode,
               "config": vars(args), "results": {"lp": {args.dataset: res}}}
    os.makedirs(args.out_dir, exist_ok=True)
    path = f"{args.out_dir}/exp_{payload['timestamp']}_lp.json"
    save_json(path, payload)
    print(f"\n{args.dataset}  MRR = {res['mrr']:.6f}  "
          f"({res['num_eval_snapshots']} eval snapshots)\nsaved -> {path}")


if __name__ == "__main__":
    main()
