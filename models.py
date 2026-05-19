from typing import Optional

import numpy as np
import torch
import torch.nn as nn


class DynamicBaseline(nn.Module):
    """
    Lightweight dynamic baseline:
    - Node embeddings are trainable.
    - A recency feature (time since last interaction) is concatenated.
    - LP head scores (src, dst, edge_feat, recency).
    - NR head predicts node labels from (node_emb, recency).
    """

    def __init__(
        self,
        num_nodes: int,
        emb_dim: int,
        edge_feat_dim: int,
        nr_num_classes: int,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.emb_dim = emb_dim
        self.edge_feat_dim = edge_feat_dim
        self.nr_num_classes = nr_num_classes

        self.node_emb = nn.Embedding(num_nodes, emb_dim)
        nn.init.xavier_uniform_(self.node_emb.weight)

        lp_in = 2 * emb_dim + edge_feat_dim + 2
        self.lp_head = nn.Sequential(
            nn.Linear(lp_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        nr_in = emb_dim + 1
        self.nr_head = nn.Sequential(
            nn.Linear(nr_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, nr_num_classes),
        )

    @staticmethod
    def _safe_log_delta(delta: torch.Tensor) -> torch.Tensor:
        return torch.log1p(torch.clamp(delta, min=0.0)).unsqueeze(-1)

    def _recency_feature(
        self,
        node_ids: torch.Tensor,
        current_ts: torch.Tensor,
        last_seen_ts: torch.Tensor,
    ) -> torch.Tensor:
        prev = last_seen_ts[node_ids]
        delta = current_ts - prev
        return self._safe_log_delta(delta)

    def lp_logits(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        edge_feat: Optional[torch.Tensor],
        ts: torch.Tensor,
        last_seen_ts: torch.Tensor,
    ) -> torch.Tensor:
        src_emb = self.node_emb(src)
        dst_emb = self.node_emb(dst)

        src_rec = self._recency_feature(src, ts, last_seen_ts)
        dst_rec = self._recency_feature(dst, ts, last_seen_ts)

        if edge_feat is None:
            ef = torch.zeros((src.shape[0], self.edge_feat_dim), device=src.device)
        else:
            ef = edge_feat
            if ef.ndim == 1:
                ef = ef.unsqueeze(-1)

        x = torch.cat([src_emb, dst_emb, ef, src_rec, dst_rec], dim=-1)
        return self.lp_head(x).squeeze(-1)

    def nr_logits(
        self,
        node_ids: torch.Tensor,
        ts_scalar: float,
        last_seen_ts: torch.Tensor,
    ) -> torch.Tensor:
        emb = self.node_emb(node_ids)
        ts = torch.full((node_ids.shape[0],), float(ts_scalar), device=node_ids.device)
        rec = self._recency_feature(node_ids, ts, last_seen_ts)
        x = torch.cat([emb, rec], dim=-1)
        return self.nr_head(x)


def update_last_seen(
    last_seen_ts: torch.Tensor,
    src: np.ndarray,
    dst: np.ndarray,
    ts: np.ndarray,
) -> None:
    with torch.no_grad():
        src_t = torch.from_numpy(src).long().to(last_seen_ts.device)
        dst_t = torch.from_numpy(dst).long().to(last_seen_ts.device)
        ts_t = torch.from_numpy(ts).float().to(last_seen_ts.device)
        last_seen_ts[src_t] = ts_t
        last_seen_ts[dst_t] = ts_t
