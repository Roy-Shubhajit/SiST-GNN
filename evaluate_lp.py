"""Rank-based link-prediction evaluation, faithful to ROLAND.

Mirrors ``graphgym/contrib/train/train_utils.py`` of snap-stanford/roland:

* ``gen_negative_edges``  - ``num_neg`` negative tails per *source node*, drawn
  with 1.5x redundancy and filtered against the true edge set
  (ROLAND's ``edge_index_difference``).
* ``per_source_mrr``      - ROLAND's ``fast_batch_mrr_and_recall``: for each
  source node take the **best** score among its positive edges, rank it against
  that source's own negatives, and average ``1/rank`` over source nodes.

The task itself is ROLAND's ``(today, tomorrow)`` pair (``get_task_batch``):
message passing runs on :math:`G_t` while the scored edges come from
:math:`G_{t+1}`, so a positive is never present in the encoder's input.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Data

# Sources processed per chunk when scoring negatives. Bounds peak memory at
# CHUNK * num_neg * hidden_dim floats.
_SRC_CHUNK = 256


def _flat_ids(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """ROLAND's (i, j) -> i * num_nodes + j flattening."""
    return edge_index[0].to(torch.int64) * num_nodes + edge_index[1].to(torch.int64)


def gen_negative_edges(
    edge_index: torch.Tensor,
    num_neg_per_node: int,
    num_nodes: int,
    forbidden: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample ``num_neg_per_node`` negative tails for every source node.

    Follows ROLAND: oversample by 1.5x, drop candidates that are real edges,
    keep the first ``num_neg_per_node`` survivors per source. ROLAND flattens
    the result and slices it with ``first_occ_idx``, which silently borrows
    from the next source when a source falls short; we keep the per-source
    matrix and top up deficient rows instead, which is what that code intends.

    Parameters
    ----------
    edge_index : (2, E) positive edges; sources define the rows.
    forbidden  : optional (2, E') edge set to filter against. Defaults to
                 ``edge_index`` itself.

    Returns
    -------
    src_lst : (S,) sorted unique source nodes.
    neg_dst : (S, num_neg_per_node) negative tails, row i belongs to src_lst[i].
    """
    device = edge_index.device
    src_lst = torch.unique(edge_index[0], sorted=True)
    S = src_lst.numel()

    forbid = edge_index if forbidden is None else forbidden
    forbid_ids = _flat_ids(forbid, num_nodes).to(device)

    n_draw = int(1.5 * num_neg_per_node)  # ROLAND's redundancy factor
    cand = torch.randint(0, num_nodes, (S, n_draw), device=device)

    valid = ~torch.isin(
        src_lst.view(-1, 1) * num_nodes + cand, forbid_ids
    )

    # Stable sort pushes valid candidates to the front of each row.
    order = torch.argsort((~valid).to(torch.int8), dim=1, stable=True)
    cand = torch.gather(cand, 1, order)
    valid = torch.gather(valid, 1, order)

    neg_dst = cand[:, :num_neg_per_node].contiguous()
    short = ~valid[:, :num_neg_per_node].all(dim=1)

    # Rare: a source with very high degree exhausted its redundancy. Resample
    # just those rows rather than borrowing another source's negatives.
    for _ in range(5):
        if not bool(short.any()):
            break
        idx = short.nonzero(as_tuple=True)[0]
        redraw = torch.randint(0, num_nodes, (idx.numel(), n_draw), device=device)
        ok = ~torch.isin(src_lst[idx].view(-1, 1) * num_nodes + redraw, forbid_ids)
        o = torch.argsort((~ok).to(torch.int8), dim=1, stable=True)
        redraw, ok = torch.gather(redraw, 1, o), torch.gather(ok, 1, o)
        neg_dst[idx] = redraw[:, :num_neg_per_node]
        short[idx] = ~ok[:, :num_neg_per_node].all(dim=1)

    return src_lst, neg_dst


def get_negative_samples(
    edge_index: torch.Tensor,
    num_nodes: int,
    num_neg: int = 1,
    forbidden: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Per-*edge* negatives for the training objective, filtered against the
    true edge set. Returns a (2, E * num_neg) edge_index."""
    device = edge_index.device
    src = edge_index[0]
    src_neg = src.repeat_interleave(num_neg)

    forbid = edge_index if forbidden is None else forbidden
    forbid_ids = _flat_ids(forbid, num_nodes).to(device)

    dst_neg = torch.randint(0, num_nodes, (src_neg.numel(),), device=device)
    for _ in range(5):
        bad = torch.isin(src_neg * num_nodes + dst_neg, forbid_ids)
        if not bool(bad.any()):
            break
        dst_neg[bad] = torch.randint(
            0, num_nodes, (int(bad.sum()),), device=device
        )

    return torch.stack([src_neg, dst_neg], dim=0)


def _score(node_emb: torch.Tensor, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """Inner-product decoder, eq. (8) of the paper."""
    return (node_emb[src] * node_emb[dst]).sum(dim=-1)


@torch.no_grad()
def per_source_mrr(
    scorer,
    pos_edge_index: torch.Tensor,
    num_nodes: int,
    num_neg: int = 1000,
    forbidden: Optional[torch.Tensor] = None,
) -> Tuple[float, Dict[int, float]]:
    """ROLAND's ``fast_batch_mrr_and_recall``.

    For each source node: rank its best-scoring positive against its own
    ``num_neg`` negatives, then average ``1/rank`` across source nodes.

    ``scorer`` is either a node-embedding tensor (scored by inner product) or a
    callable ``(src_idx, dst_idx) -> scores`` for a learned decoder.
    """
    if torch.is_tensor(scorer):
        node_emb = scorer
        device = node_emb.device
        score_fn = lambda s, d: _score(node_emb, s, d)  # noqa: E731
    else:
        device = pos_edge_index.device
        score_fn = scorer

    src_lst, neg_dst = gen_negative_edges(
        pos_edge_index, num_neg, num_nodes, forbidden=forbidden
    )

    pos_scores = score_fn(pos_edge_index[0], pos_edge_index[1])

    # Best positive score per source (ROLAND's default scatter_max reduction).
    best = torch.full((num_nodes,), float("-inf"), device=device)
    best.scatter_reduce_(0, pos_edge_index[0], pos_scores, reduce="amax",
                         include_self=True)
    best_by_src = best[src_lst]

    # Rank against that source's negatives, chunked to bound memory.
    ranks = torch.empty(src_lst.numel(), device=device)
    for s in range(0, src_lst.numel(), _SRC_CHUNK):
        e = min(s + _SRC_CHUNK, src_lst.numel())
        chunk_src = src_lst[s:e]                                  # (C,)
        chunk_neg = neg_dst[s:e]                                  # (C, num_neg)
        flat_src = chunk_src.view(-1, 1).expand_as(chunk_neg).reshape(-1)
        neg_scores = score_fn(flat_src, chunk_neg.reshape(-1)).view(chunk_neg.shape)
        ranks[s:e] = (neg_scores >= best_by_src[s:e].view(-1, 1)).sum(1) + 1

    mrr = float((1.0 / ranks).mean())
    recall_at = {k: float((ranks <= k).float().mean()) for k in (1, 3, 10)}
    return mrr, recall_at


@torch.no_grad()
def evaluate_task(
    model,
    mp_snap: Data,
    target_snap: Data,
    num_neg: int = 1000,
    update_state: bool = True,
    target_edge_index: Optional[torch.Tensor] = None,
) -> Optional[Tuple[float, Dict[int, float]]]:
    """ROLAND's task ``(t, t+1)``: encode ``mp_snap`` (:math:`G_t`), score the
    edges of ``target_snap`` (:math:`G_{t+1}`).

    ``update_state=True`` advances the recurrent state with :math:`G_t`, which
    is what ROLAND's ``update_node_states`` does after each task. The target
    edges are scored *before* :math:`G_{t+1}` is ever fed to the encoder.

    ``target_edge_index`` scores a subset of :math:`G_{t+1}`'s edges (ROLAND's
    test edge-label split). Negatives are still filtered against the snapshot's
    *full* edge set, so a held-out true edge is never sampled as a negative.
    """
    model.eval()
    node_emb = model(mp_snap.x, mp_snap.edge_index, mp_snap.edge_attr,
                     update_state=update_state)
    targets = (target_snap.edge_index if target_edge_index is None
               else target_edge_index)
    if targets.size(1) == 0:
        return None
    score_edges = getattr(model, "score_edges", None)
    scorer = node_emb if score_edges is None else (
        lambda s, d: score_edges(node_emb, s, d))
    return per_source_mrr(scorer, targets,
                          mp_snap.num_nodes, num_neg,
                          forbidden=target_snap.edge_index)


def aggregate(values) -> float:
    """Snapshot-by-snapshot mean, as reported in the paper."""
    return float(np.mean(values)) if len(values) else 0.0
