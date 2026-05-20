import torch
import numpy as np
from torch_geometric.data import Data
from typing import Dict

def get_negative_samples(edge_index: torch.Tensor, num_nodes: int, num_neg: int = 1000) -> torch.Tensor:
    """
    Generates negative samples for MRR evaluation.
    For each positive edge (u, v), generates num_neg random edges (u, v_neg).
    """
    src = edge_index[0]
    num_pos = src.size(0)
    
    # Broadcast src to match negative samples: [num_pos * num_neg]
    src_neg = src.repeat_interleave(num_neg)
    
    # Randomly sample destinations. A strict implementation would ensure no actual 
    # edges are sampled, but uniform random over a large node set is standard proxy.
    dst_neg = torch.randint(0, num_nodes, (num_pos * num_neg,), device=edge_index.device)
    
    return torch.stack([src_neg, dst_neg], dim=0)

@torch.no_grad()
def evaluate_mrr(model, snapshot: Data, num_neg: int = 1000) -> float:
    """
    Computes the Mean Reciprocal Rank (MRR) for a given snapshot.
    For each positive edge, 1000 negative edges are sampled.
    """
    model.eval()
    device = snapshot.edge_index.device
    num_nodes = snapshot.num_nodes
    
    src = snapshot.edge_index[0]
    dst = snapshot.edge_index[1]
    
    # Positive scores
    # Assuming the temporal model exposes a decode method or forward takes edge_index
    # Positives
    node_emb = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr, update_state=False)
    
    pos_scores = (node_emb[src] * node_emb[dst]).sum(dim=-1)
    
    # Negative scores
    neg_edge_index = get_negative_samples(snapshot.edge_index, num_nodes, num_neg)
    neg_src = neg_edge_index[0]
    neg_dst = neg_edge_index[1]
    
    neg_scores = (node_emb[neg_src] * node_emb[neg_dst]).sum(dim=-1)
    neg_scores = neg_scores.view(-1, num_neg)
    
    # Calculate MRR
    pos_scores = pos_scores.view(-1, 1)
    
    # Rank positive score among negative scores
    # Number of negative scores greater than or equal to positive score
    ranks = (neg_scores >= pos_scores).sum(dim=-1) + 1
    
    mrr = (1.0 / ranks.float()).mean().item()
    return mrr

def evaluate_fixed_split(model, snapshots: list, start_idx: int) -> float:
    """ Evaluates MRR on the final test snapshots (Fixed-Split). """
    mrrs = []
    device = model.device if hasattr(model, 'device') else next(model.parameters()).device
    for i in range(start_idx, len(snapshots)):
        snap = snapshots[i].to(device)
        if snap.edge_index.size(1) == 0:
            continue
        mrr = evaluate_mrr(model, snap)
        mrrs.append(mrr)
    return np.mean(mrrs) if mrrs else 0.0

