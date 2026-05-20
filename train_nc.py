"""
Training & evaluation loops for dynamic node classification (Wikipedia/Reddit).

Each interaction carries a binary label that describes the *source node's*
state at the time of the interaction (e.g. did the user get banned soon after
this edit). The model predicts that label from the temporal embedding of the
source node, following the TGN / JODIE convention used by benchtemp.

Backbone: `TemporalGNNModel` / `TemporalTransformerGNNModel` (SiST-GNN).
Inputs to the backbone:
    - benchtemp's static ``node_features`` (projected to hidden_dim)
    - plus an optional learnable per-node embedding (default ON because the
      JODIE preprocessor stores all-zero node features).
Head: 2-layer MLP on the source-node embedding → binary logit (BCE loss).
Metric: ROC AUC.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch_geometric.data import Data


# ──────────────────────────────────────────────────────────────────────────────
# Model wrapper
# ──────────────────────────────────────────────────────────────────────────────


class NCStatefulWrapper(nn.Module):
    """
    SiST-GNN backbone + node-classification head.

    Inputs at every snapshot:
        X_t  =  Linear(node_features)  +  node_emb(node_ids)    [if use_node_emb]
        X_t  =  Linear(node_features)                            [otherwise]

    Output at every snapshot: temporal node embedding [N, hidden_dim].

    Per-interaction prediction at time t:
        logit = MLP( emb_t[source_node] )           # node classification
    """

    def __init__(
        self,
        base_model: nn.Module,
        num_nodes: int,
        hidden_dim: int,
        node_features: torch.Tensor,             # [num_nodes, node_feat_dim]
        edge_feat_dim: int,
        device: torch.device,
        use_node_emb: bool = True,
        use_edge_emb: bool = True,
        classifier_hidden: int = 80,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.edge_feat_dim = edge_feat_dim
        self.device = device
        self.use_node_emb = use_node_emb
        self.use_edge_emb = use_edge_emb

        node_feat_dim = int(node_features.shape[1])
        # Register the static benchtemp features so they move with .to(device).
        self.register_buffer("node_features", node_features.to(torch.float))
        self.node_feat_proj = nn.Linear(node_feat_dim, hidden_dim)

        if use_node_emb:
            self.node_emb = nn.Embedding(num_nodes, hidden_dim)
            nn.init.xavier_uniform_(self.node_emb.weight)

        # Edge encoder: learn a hidden_dim embedding of each raw edge feature
        # vector, then squash to a positive scalar weight via softplus. The
        # scalar weight is what we feed into the GNN: this keeps GCNConv's
        # symmetric normalisation well-defined (sqrt(positive)) while still
        # letting the model learn per-edge importances from the LIWC features.
        # The full edge embedding is concatenated to the classifier so the
        # head also sees the current interaction's context.
        if use_edge_emb:
            self.edge_encoder = nn.Sequential(
                nn.Linear(edge_feat_dim, hidden_dim),
                nn.ReLU(),
            )
            self.edge_weight_head = nn.Sequential(
                nn.Linear(hidden_dim, 1),
                nn.Softplus(),
            )

        # Node-classification head: takes the source-node temporal embedding
        # (TGN / JODIE convention) plus, when edge_emb is enabled, the current
        # interaction's edge embedding as context.
        head_in = hidden_dim + (hidden_dim if use_edge_emb else 0)
        self.classifier = nn.Sequential(
            nn.Linear(head_in, classifier_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, classifier_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, 1),
        )

        self.is_lstm = hasattr(self.base_model, "init_hidden")
        self.reset_states()

    # ── State management ─────────────────────────────────────────────
    def reset_states(self) -> None:
        if self.is_lstm:
            self.H_list, self.C_list = self.base_model.init_hidden(self.num_nodes, self.device)
            self.history_list = None
        else:
            self.H_list, self.C_list = None, None
            self.history_list = self.base_model.init_history(self.num_nodes, self.device)

    def detach_states(self) -> None:
        if self.H_list is not None:
            self.H_list = [h.detach() for h in self.H_list]
        if self.C_list is not None:
            self.C_list = [c.detach() for c in self.C_list]
        if self.history_list is not None:
            self.history_list = [h.detach() for h in self.history_list]

    # ── Forward ──────────────────────────────────────────────────────
    def encode_edges(self, edge_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (edge_emb [E, hidden_dim], edge_weight [E]).
        ``edge_weight`` is strictly positive (softplus) so GCNConv's
        symmetric normalisation remains numerically stable even though the raw
        LIWC features can be negative.
        """
        edge_emb = self.edge_encoder(edge_feat)
        edge_weight = self.edge_weight_head(edge_emb).squeeze(-1) + 1e-6
        return edge_emb, edge_weight

    def compute_node_emb(
        self,
        edge_index: torch.Tensor,
        edge_feat: torch.Tensor | None = None,
        update_state: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor | None]:
        """
        Returns ``(node_emb, edge_emb)``. ``edge_emb`` is ``None`` when edge
        embeddings are disabled or no edge features were supplied.
        """
        x = self.node_feat_proj(self.node_features)
        if self.use_node_emb:
            ids = torch.arange(self.num_nodes, device=self.device)
            x = x + self.node_emb(ids)

        edge_emb: torch.Tensor | None = None
        gnn_edge_attr: torch.Tensor | None = None
        if self.use_edge_emb and edge_feat is not None and edge_feat.numel() > 0:
            edge_emb, gnn_edge_attr = self.encode_edges(edge_feat)

        if self.is_lstm:
            out, new_H, new_C = self.base_model.forward_step(
                x, edge_index, self.H_list, self.C_list, edge_attr=gnn_edge_attr
            )
            if update_state:
                self.H_list, self.C_list = new_H, new_C
        else:
            out, new_history = self.base_model.forward_step(
                x, edge_index, self.history_list, edge_attr=gnn_edge_attr
            )
            if update_state:
                self.history_list = new_history
        return out, edge_emb

    def classify_nodes(
        self,
        node_emb: torch.Tensor,
        src: torch.Tensor,
        interaction_edge_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Predict the source-node's binary state at this interaction time.

        When edge embeddings are enabled, ``interaction_edge_emb`` (the embedded
        edge feature of the *current* interaction) is concatenated with the
        source-node embedding before the MLP head.
        """
        if self.use_edge_emb:
            if interaction_edge_emb is None:
                raise ValueError("use_edge_emb=True but no interaction_edge_emb supplied")
            feats = torch.cat([node_emb[src], interaction_edge_emb], dim=-1)
        else:
            feats = node_emb[src]
        return self.classifier(feats).squeeze(-1)


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────


def _to_device(snap: Data, device: torch.device) -> Data:
    snap = snap.to(device)
    snap.edge_label = snap.edge_label.to(device)
    snap.edge_split = snap.edge_split.to(device)
    snap.edge_feat = snap.edge_feat.to(device)
    return snap


def _bce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    pos_weight: float | None,
) -> torch.Tensor:
    if pos_weight is None or pos_weight == 1.0:
        return F.binary_cross_entropy_with_logits(logits, labels)
    pw = torch.tensor(pos_weight, device=logits.device, dtype=logits.dtype)
    return F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pw)


def _resolve_pos_weight(
    spec: str | float,
    snapshots: List[Data],
    train_end: int,
) -> float | None:
    """
    Convert a user-facing pos_weight spec into a concrete value (or None for
    plain BCE). Supported specs:
        - "none"     : plain BCE (TGN/JODIE convention) — recommended for AP.
        - "balanced" : neg/pos ratio (aggressive — boosts AUC, hurts AP).
        - "sqrt"     : sqrt(neg/pos) — middle ground.
        - float      : use the given numeric weight verbatim.
    """
    if isinstance(spec, str):
        spec = spec.lower()
        if spec == "none":
            return None
        if spec in {"balanced", "sqrt"}:
            pos = neg = 0
            for snap in snapshots[:train_end]:
                mask = snap.edge_split == 0
                if mask.any():
                    lab = snap.edge_label[mask]
                    pos += int(lab.sum().item())
                    neg += int((1 - lab).sum().item())
            if pos == 0:
                return None
            ratio = neg / pos
            return ratio if spec == "balanced" else float(ratio ** 0.5)
    return float(spec)


@torch.no_grad()
def _eval_pass(
    model: NCStatefulWrapper,
    snapshots: List[Data],
    target_split: int,
    device: torch.device,
) -> float:
    """
    Stream every snapshot in order (state must propagate from t=0) and collect
    predictions on edges whose split == target_split. Returns ROC AUC.
    """
    model.eval()
    model.reset_states()

    all_logits: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    for snap in snapshots:
        snap = _to_device(snap, device)
        if snap.edge_index.size(1) == 0:
            continue
        node_emb, edge_emb = model.compute_node_emb(
            snap.edge_index, snap.edge_feat, update_state=True
        )

        mask = snap.edge_split == target_split
        if mask.any():
            src = snap.edge_index[0][mask]
            ie_emb = edge_emb[mask] if edge_emb is not None else None
            logits = model.classify_nodes(node_emb, src, interaction_edge_emb=ie_emb)
            all_logits.append(logits.detach().cpu())
            all_labels.append(snap.edge_label[mask].detach().cpu())

    if not all_logits:
        return 0.0

    y_score = torch.cat(all_logits).numpy()
    y_true = torch.cat(all_labels).numpy()
    if len(np.unique(y_true)) < 2:
        return 0.0
    return float(roc_auc_score(y_true, y_score))


def train_nc_fixed_split(
    snapshots: List[Data],
    model: NCStatefulWrapper,
    optimizer: torch.optim.Optimizer,
    train_end: int,
    val_end: int,
    num_epochs: int = 50,
    patience: int = 5,
    pos_weight: str | float = "none",
    verbose: bool = True,
) -> dict:
    """
    Fixed-split training: replay snapshots 0..train_end every epoch and supervise
    on edges with split==0. Validate / test by streaming all snapshots and
    collecting predictions at the corresponding edges.

    Parameters
    ----------
    pos_weight : "none" (default, plain BCE), "balanced" (neg/pos),
                 "sqrt" (sqrt(neg/pos)), or a numeric value.
    """
    device = model.device
    pos_weight_val = _resolve_pos_weight(pos_weight, snapshots, train_end)
    if verbose:
        pw_str = "None (plain BCE)" if pos_weight_val is None else f"{pos_weight_val:.3f}"
        print(
            f"[NC] pos_weight={pw_str}, "
            f"train_end={train_end}, val_end={val_end}"
        )

    best_val_auc = -1.0
    best_state: dict | None = None
    rounds_without_improve = 0

    for epoch in range(num_epochs):
        model.train()
        model.reset_states()

        total_loss = 0.0
        total_edges = 0
        for snap in snapshots[:train_end]:
            snap = _to_device(snap, device)
            if snap.edge_index.size(1) == 0:
                continue

            optimizer.zero_grad()
            node_emb, edge_emb = model.compute_node_emb(
                snap.edge_index, snap.edge_feat, update_state=True
            )

            mask = snap.edge_split == 0
            if not mask.any():
                model.detach_states()
                continue

            src = snap.edge_index[0][mask]
            labels = snap.edge_label[mask]
            ie_emb = edge_emb[mask] if edge_emb is not None else None
            logits = model.classify_nodes(node_emb, src, interaction_edge_emb=ie_emb)

            loss = _bce_loss(logits, labels, pos_weight_val)
            loss.backward()
            model.detach_states()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            total_edges += labels.size(0)

        avg_loss = total_loss / max(total_edges, 1)
        val_auc = _eval_pass(model, snapshots, target_split=1, device=device)

        if verbose:
            print(
                f"Epoch {epoch + 1:03d}/{num_epochs} | "
                f"loss={avg_loss:.4f} | val AUC={val_auc:.4f}"
            )

        improved = val_auc > best_val_auc + 1e-4
        if improved:
            best_val_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            rounds_without_improve = 0
        else:
            rounds_without_improve += 1
            if rounds_without_improve >= patience:
                if verbose:
                    print(
                        f"[NC] Early stopping at epoch {epoch + 1} "
                        f"(best val AUC={best_val_auc:.4f})"
                    )
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_auc = _eval_pass(model, snapshots, target_split=2, device=device)
    if verbose:
        print(f"[NC] Test ROC AUC={test_auc:.4f}")

    return {
        "val_auc": best_val_auc,
        "test_auc": test_auc,
        "pos_weight": pos_weight_val if pos_weight_val is not None else "none",
    }
