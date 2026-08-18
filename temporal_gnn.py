"""
Temporal GNN Layer and Model
============================
Implements a novel temporal GNN layer that couples per-node LSTMCell updates
with a graph convolution over a temporally-augmented edge set.

This file contains two variants:
1) LSTM-based temporal update (`TemporalGNNLayer`, `TemporalGNNModel`)
2) Transformer-based temporal update
    (`TemporalTransformerGNNLayer`, `TemporalTransformerGNNModel`)

Forward pass (per timestep i):
    1. LSTMCell  : X_i, (H_{i-1}, C_{i-1})  →  X̃_i (≡ H_i), C_i
    2. Stack     : cat([proj(X_i), X̃_i], dim=0)  →  X_new  ∈ ℝ^{2N × hidden_dim}
    3. Edge aug  : for every (u, v) ∈ edge_index, add (u, v+N)
    4. GNN       : X_new, new_edge_index  →  X′_new  ∈ ℝ^{2N × hidden_dim}

Returns: X′_new[:N], H_i, C_i
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn


# ──────────────────────────────────────────────────────────────────────────────
# Layer
# ──────────────────────────────────────────────────────────────────────────────

class TemporalGNNLayer(nn.Module):
    """
    A single temporal GNN layer.

    Parameters
    ----------
    input_dim  : Dimensionality of incoming node features X_i.
    hidden_dim : LSTM hidden size; also the uniform feature width used by the GNN.
    gnn_type   : Which GNN convolution to use – one of ``'GCNConv'``, ``'GATConv'``,
                 ``'SAGEConv'`` (case-insensitive).
    **gnn_kwargs
        Extra keyword arguments forwarded verbatim to the chosen GNN class
        (e.g. ``heads=4, dropout=0.1`` for GAT).
    """

    def __init__(
        self,
        input_dim:  int,
        hidden_dim: int,
        gnn_type:   str = "GCNConv",
        full_self_temporal: bool = False,
        **gnn_kwargs,
    ) -> None:
        super().__init__()
        LayerClass = getattr(pyg_nn, gnn_type, None)
        if LayerClass is None:
            raise ValueError(f"Unsupported gnn_type '{gnn_type}'.")


        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.full_self_temporal = full_self_temporal

        # Project X_i from input_dim → hidden_dim so both halves of X_new
        # share the same feature width before stacking.
        if input_dim != hidden_dim:
            self.input_proj = nn.Linear(input_dim, hidden_dim, bias=False)

        # Per-node temporal update – operates independently on each node's
        # feature trajectory across time steps.
        self.lstm_cell = nn.LSTMCell(input_dim, hidden_dim)

        # Calculate actual GNN output dimension based on heads parameter
        # For multi-head attention layers, output_dim = hidden_dim * heads
        heads = gnn_kwargs.get('heads', 1)
        self.gnn_output_dim = hidden_dim * heads
        
        # GNN processes the 2N-node temporally augmented graph.
        self.gnn  = LayerClass(hidden_dim, hidden_dim, **gnn_kwargs)
        
        # Project GNN output back to hidden_dim if it changed due to heads
        if self.gnn_output_dim != hidden_dim:
            self.gnn_output_proj = nn.Linear(self.gnn_output_dim, hidden_dim, bias=False)
        else:
            self.gnn_output_proj = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_augmented_edges(
        self,
        edge_index: torch.Tensor,   # [2, E]
        num_nodes:  int,
    ) -> torch.Tensor:
        """
        Extend *edge_index* with cross-time and self-temporal edges.

        For every (u, v) ∈ edge_index we append (v + N, u) — connecting the
        temporal counterpart of v back to u.

        The self-temporal edges (i + N, i) inject node i's own recurrent state
        into its output. Two variants:

        ``full_self_temporal=False`` (legacy)
            One per *edge*, at the source: ``(u + N, u)`` for each (u, v).
            Shape [2, 3E]. Nodes that never appear as a source therefore get no
            self-temporal edge at all, and since both cross- and self-edges also
            target only source nodes, such a node's output carries **zero**
            dependence on H — it has no temporal memory. High-degree sources
            additionally get their self-edge repeated deg(u) times, which skews
            the GCN normalisation.

        ``full_self_temporal=True`` (Algorithm 1)
            One per *node*: ``{(N+i, i)}`` for i = 1..N. Shape [2, 2E + N].
            Every node sees its own history exactly once.
        """
        src, tgt = edge_index[0], edge_index[1]

        temporal_tgt = tgt + num_nodes                         # v + N → v
        cross_edges  = torch.stack([temporal_tgt, src], dim=0)  # [2, E]

        if self.full_self_temporal:
            nodes = torch.arange(num_nodes, device=edge_index.device)
            self_loop_edges = torch.stack([nodes + num_nodes, nodes], dim=0)  # [2, N]
        else:
            self_loop_edges = torch.stack([src + num_nodes, src], dim=0)      # [2, E]

        return torch.cat([edge_index, cross_edges, self_loop_edges], dim=1)

    def _build_augmented_edge_attr(
        self,
        edge_attr: Optional[torch.Tensor],
        num_nodes: int,
    ) -> Optional[torch.Tensor]:
        if edge_attr is None:
            return None
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(1)
        if not self.full_self_temporal:
            # 3E entries: original, cross-time, and per-edge self-temporal.
            return torch.cat([edge_attr, edge_attr, edge_attr], dim=0)
        # 2E + N entries; give the N self-temporal edges a neutral weight on the
        # same scale as the observed edge attributes.
        fill = edge_attr.mean(dim=0, keepdim=True).expand(num_nodes, -1)
        return torch.cat([edge_attr, edge_attr, fill], dim=0)

    def _apply_gnn(
        self,
        X_new: torch.Tensor,
        aug_edges: torch.Tensor,
        aug_edge_attr: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if aug_edge_attr is None:
            X_out = self.gnn(X_new, aug_edges)
        else:
            try:
                X_out = self.gnn(X_new, aug_edges, edge_attr=aug_edge_attr)
            except (TypeError, AssertionError):
                # If edge_attr fails (e.g., GATv2Conv not initialized with edge_dim),
                # try converting to edge_weight
                if aug_edge_attr.dim() == 2 and aug_edge_attr.size(1) == 1:
                    edge_weight = aug_edge_attr[:, 0]
                elif aug_edge_attr.dim() == 1:
                    edge_weight = aug_edge_attr
                else:
                    edge_weight = aug_edge_attr.mean(dim=1)

                try:
                    X_out = self.gnn(X_new, aug_edges, edge_weight=edge_weight)
                except TypeError:
                    # If edge_weight also fails, just use the GNN without edge info
                    X_out = self.gnn(X_new, aug_edges)

        # Project back to hidden_dim if needed (e.g., after multi-head attention)
        if self.gnn_output_proj is not None:
            X_out = self.gnn_output_proj(X_out)

        return X_out

    def _apply_gnn_folded(
        self,
        X_new: torch.Tensor,            # [2*B*N, hidden]  laid out [X_proj(B,N); H(B,N)]
        edge_index: torch.Tensor,       # [2, E]  SINGLE-graph edges (shared by the batch)
        edge_attr: Optional[torch.Tensor],  # [E] / [E,1] single-graph weights, or None
        num_graphs: int,
    ) -> torch.Tensor:
        """Shared-adjacency batched GCN: one SpMM instead of B block-diagonal copies.

        Exploits that every sample in the batch lives on the SAME graph, so the
        augmented adjacency is normalized once (2N x 2N) and the batch rides in
        the dense columns:  einsum('nm,mbc->nbc', A_hat, X)  executed as
        A_hat @ X.view(2N, B*d).  Mathematically identical to feeding the
        block-diagonal batch through ``self.gnn`` (a PyG ``GCNConv``, whose
        ``lin``/``bias`` parameters and ``gcn_norm`` are reused verbatim) — only
        the execution path changes. Requires ``gnn_type='GCNConv'``.
        """
        from torch_geometric.nn.conv.gcn_conv import gcn_norm

        B = num_graphs
        N = X_new.size(0) // (2 * B)
        n = 2 * N

        # Normalized, transposed augmented adjacency — cached (static per run).
        key = (n, edge_index.size(1))
        cache = getattr(self, "_folded_adj", None)
        if cache is None or cache[0] != key:
            aug_ei = self._build_augmented_edges(edge_index, N)   # [2, 3E] or [2, 2E+N]
            if edge_attr is None:
                aug_ew = None
            else:
                ew = edge_attr[:, 0] if edge_attr.dim() == 2 else edge_attr
                tail = ew if not self.full_self_temporal else ew.mean().expand(N)
                aug_ew = torch.cat([ew, ew, tail], dim=0)
            ei_n, ew_n = gcn_norm(
                aug_ei, aug_ew, num_nodes=n,
                improved=self.gnn.improved, add_self_loops=self.gnn.add_self_loops,
            )
            # PyG aggregates src->dst (out[dst] += w * x[src]) => store A[dst, src].
            # Explicitly opt in to sparse invariant checks (torch's documented way
            # to silence its warning) — free here, since this is built once and
            # cached, and it validates our indices are in-bounds.
            with torch.sparse.check_sparse_tensor_invariants():
                A = torch.sparse_coo_tensor(
                    ei_n.flip(0), ew_n, (n, n), device=X_new.device
                ).coalesce()
            self._folded_adj = (key, A)
        A = self._folded_adj[1]

        # [2*B*N, d] -> [2N, B, d]: per-sample blocks of the single augmented graph.
        d = X_new.size(1)
        X3 = X_new.view(2, B, N, d).permute(0, 2, 1, 3).reshape(n, B, d)
        Z = self.gnn.lin(X3)                                             # per-sample W
        d_out = Z.size(-1)
        Y = torch.sparse.mm(A, Z.reshape(n, B * d_out)).view(n, B, d_out)
        if self.gnn.bias is not None:
            Y = Y + self.gnn.bias
        # Back to the batched flat layout [2*B*N, d_out].
        return Y.view(2, N, B, d_out).permute(0, 2, 1, 3).reshape(2 * B * N, d_out)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        X_i:       torch.Tensor,        # [N, input_dim]
        edge_index: torch.Tensor,       # [2, E]
        H_prev:    torch.Tensor,        # [N, hidden_dim]
        C_prev:    torch.Tensor,        # [N, hidden_dim]
        edge_attr: Optional[torch.Tensor] = None,
        num_graphs: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        X_i        : Node features at the current time step.   [N, input_dim]
        edge_index : COO-format edge list.                     [2, E]
        H_prev     : LSTM hidden state carried from time i-1.  [N, hidden_dim]
        C_prev     : LSTM cell state carried from time i-1.    [N, hidden_dim]
        num_graphs : If set, ``X_i`` holds ``num_graphs`` independent samples on
                     the SAME graph (batch-major), and ``edge_index``/``edge_attr``
                     describe that single shared graph. The GCN then runs via the
                     folded shared-adjacency path (GCNConv only). Default ``None``
                     keeps the standard behavior (``edge_index`` is the full graph).

        Returns
        -------
        X_prime[:N] : GNN output for the original N nodes.     [N, hidden_dim]
        H_i     : Updated LSTM hidden state.                   [N, hidden_dim]
        C_i     : Updated LSTM cell state.                     [N, hidden_dim]
        """
        # ── 1. Temporal update ─────────────────────────────────────────
        H_i, C_i = self.lstm_cell(X_i, (H_prev, C_prev))    # [N, hidden_dim]

        # ── 2. Stack: project X_i then concatenate along the node axis ─
        if self.input_dim != self.hidden_dim:
            X_proj = self.input_proj(X_i)                   # [N, hidden_dim]
        else:
            X_proj = X_i
        X_new  = torch.cat([X_proj, H_i], dim=0)            # [2N, hidden_dim]

        if num_graphs is not None:
            if not isinstance(self.gnn, pyg_nn.GCNConv):
                raise ValueError("num_graphs (folded batching) requires gnn_type='GCNConv'.")
            X_prime = self._apply_gnn_folded(X_new, edge_index, edge_attr, num_graphs)
            return X_prime[: X_i.size(0)], H_i, C_i

        # ── 3. Edge augmentation ────────────────────────────────────────
        aug_edges = self._build_augmented_edges(edge_index, X_i.size(0))
        aug_edge_attr = self._build_augmented_edge_attr(edge_attr, X_i.size(0))

        # ── 4. Graph convolution on the augmented graph ─────────────────
        X_prime = self._apply_gnn(X_new, aug_edges, aug_edge_attr) # [2N, hidden_dim]

        return X_prime[: X_i.size(0)], H_i, C_i


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────

class TemporalGNNModel(nn.Module):
    """
    Stacked temporal GNN model for dynamic graph sequences.

    Processes a sequence of T graph snapshots by feeding each through
    ``num_layers`` stacked :class:`TemporalGNNLayer` blocks. LSTM hidden and
    cell states are carried across time steps, giving the model memory of the
    graph's history.

    Node-level predictions are produced at every time step from the features
    of the *original* N nodes (the lower half of each layer's 2N output).

    Parameters
    ----------
    input_dim  : Raw node-feature dimensionality.
    hidden_dim : Latent width used throughout (LSTM + GNN).
    output_dim : Dimensionality of the per-node prediction head.
    num_layers : Number of stacked TemporalGNNLayer blocks.  (default 2)
    gnn_type   : ``'GCNConv'``, ``'GATConv'``, or ``'SAGEConv'``.       (default 'GCNConv')
    dropout    : Dropout probability applied between layers. (default 0.0)
    **gnn_kwargs
        Forwarded to every TemporalGNNLayer's GNN constructor.
    """

    def __init__(
        self,
        input_dim:  int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int  = 2,
        gnn_type:   str  = "GCNConv",
        dropout:    float = 0.0,
        full_self_temporal: bool = False,
        **gnn_kwargs,
    ) -> None:
        super().__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # Build each layer; layers 2+ receive hidden_dim-wide input
        # (sliced from the previous layer's 2N output).
        self.layers = nn.ModuleList()
        for idx in range(num_layers):
            in_dim = input_dim if idx == 0 else hidden_dim
            self.layers.append(
                TemporalGNNLayer(in_dim, hidden_dim, gnn_type,
                                 full_self_temporal=full_self_temporal,
                                 **gnn_kwargs)
            )

        self.act     = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

        # Per-node linear readout applied to the final layer's output.
        # Final layer always outputs hidden_dim due to projection in TemporalGNNLayer
        self.readout = nn.Linear(hidden_dim, output_dim)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def init_hidden(
        self, N: int, device: torch.device
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Initialise LSTM hidden and cell states to **zero** for all layers.

        Returns
        -------
        H_list, C_list : Each a list of ``num_layers`` tensors of shape [N, hidden_dim].
        """
        zeros = lambda: [
            torch.zeros(N, self.hidden_dim, device=device)
            for _ in range(self.num_layers)
        ]
        return zeros(), zeros()

    # ------------------------------------------------------------------
    # Single time-step
    # ------------------------------------------------------------------

    def forward_step(
        self,
        X:         torch.Tensor,            # [N, feature_dim]
        edge_index: torch.Tensor,           # [2, E]
        H_list:    List[torch.Tensor],
        C_list:    List[torch.Tensor],
        edge_attr: Optional[torch.Tensor] = None,
        num_graphs: Optional[int] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        Push one snapshot through all layers and return the prediction.

        Parameters
        ----------
        X          : Node features for this snapshot.
        edge_index : Edges for this snapshot.
        H_list     : Per-layer LSTM hidden states from the previous step.
        C_list     : Per-layer LSTM cell states from the previous step.
        num_graphs : Optional folded-batch hint (see ``TemporalGNNLayer.forward``).

        Returns
        -------
        out    : Per-node predictions.                          [N, output_dim]
        H_list : Updated hidden states (one tensor per layer). [N, hidden_dim] each
        C_list : Updated cell   states (one tensor per layer). [N, hidden_dim] each
        """
        N = X.size(0)

        new_H: List[torch.Tensor] = []
        new_C: List[torch.Tensor] = []

        for idx, layer in enumerate(self.layers):
            X_prime, H_i, C_i = layer(
                X, edge_index, H_list[idx], C_list[idx], edge_attr=edge_attr,
                num_graphs=num_graphs,
            )
            new_H.append(H_i)
            new_C.append(C_i)

            # Slice back to original N nodes for the next layer's input.
            X = X_prime[:N]                        # [N, hidden_dim]
            X = self.act(X)
            if idx < self.num_layers - 1:
                X = self.dropout(X)

        out = self.readout(X)                       # [N, output_dim]
        return out, new_H, new_C

    # ------------------------------------------------------------------
    # Full sequence
    # ------------------------------------------------------------------

    def forward(
        self,
        X_seq:        List[torch.Tensor],           # T × [N, input_dim]
        edge_seq:     List[torch.Tensor],           # T × [2, E_t]
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Process a full sequence of T graph snapshots.

        The LSTM states are initialised to zero and propagated through all T
        steps, so the model accumulates temporal context as T grows.

        Parameters
        ----------
        X_seq        : List of T node-feature tensors, each [N, input_dim].
        edge_seq     : List of T edge_index tensors, each [2, E_t].
                   Edges can change at every step (dynamic topology).

        Returns
        -------
        outputs : List of T per-node prediction tensors, each [N, output_dim].
        """
        T      = len(X_seq)
        N      = X_seq[0].size(0)
        device = X_seq[0].device

        H_list, C_list = self.init_hidden(N, device)
        outputs: List[torch.Tensor] = []

        for t in range(T):
            out, H_list, C_list = self.forward_step(
                X_seq[t], edge_seq[t], H_list, C_list
            )
            outputs.append(out)

        return out, outputs   # list of T tensors, each [N, output_dim]


# ──────────────────────────────────────────────────────────────────────────────
# Transformer Variant
# ──────────────────────────────────────────────────────────────────────────────

class TemporalTransformerGNNLayer(nn.Module):
    """
    A single temporal GNN layer using TransformerEncoder for temporal updates.

    Parameters
    ----------
    input_dim  : Dimensionality of incoming node features X_i.
    hidden_dim : Latent width used by Transformer and GNN.
    gnn_type   : Which GNN convolution to use - one of ``'GCNConv'``, ``'GATConv'``,
                 ``'SAGEConv'``.
    transformer_heads : Number of attention heads.
    transformer_layers: Number of Transformer encoder layers.
    transformer_ff_dim: Feed-forward width inside Transformer. Defaults to 4 * hidden_dim.
    transformer_dropout : Dropout inside Transformer blocks.
    max_history : Maximum temporal context length to keep per node.
    **gnn_kwargs
        Extra keyword arguments forwarded to the chosen GNN class.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        gnn_type: str = "GCNConv",
        transformer_heads: int = 4,
        transformer_layers: int = 1,
        transformer_ff_dim: Optional[int] = None,
        transformer_dropout: float = 0.1,
        max_history: int = 32,
        **gnn_kwargs,
    ) -> None:
        super().__init__()

        LayerClass = getattr(pyg_nn, gnn_type, None)
        if LayerClass is None:
            raise ValueError(f"Unsupported gnn_type '{gnn_type}'.")
        if hidden_dim % transformer_heads != 0:
            raise ValueError(
                "hidden_dim must be divisible by transformer_heads "
                f"(got hidden_dim={hidden_dim}, heads={transformer_heads})."
            )
        if max_history < 1:
            raise ValueError("max_history must be >= 1.")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_history = max_history

        if input_dim != hidden_dim:
            self.input_proj = nn.Linear(input_dim, hidden_dim, bias=False)

        ff_dim = transformer_ff_dim or (4 * hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=transformer_heads,
            dim_feedforward=ff_dim,
            dropout=transformer_dropout,
            activation="gelu",
            batch_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=transformer_layers
        )
        
        # Calculate actual GNN output dimension based on heads parameter
        # For multi-head attention layers (GAT, GATv2), output_dim = hidden_dim * heads
        # Extract and remove 'heads' from gnn_kwargs to avoid passing it twice
        if gnn_type in ("GATConv", "GATv2Conv"):
            gnn_heads = gnn_kwargs.pop('heads', transformer_heads)
        else:
            gnn_heads = 1
            
        self.gnn_output_dim = hidden_dim * gnn_heads
        
        # GNN initialization - handle multi-head layers properly
        if gnn_type in ("GATConv", "GATv2Conv"):
            self.gnn = LayerClass(hidden_dim, hidden_dim, heads=gnn_heads, **gnn_kwargs)
        else:
            self.gnn = LayerClass(hidden_dim, hidden_dim, **gnn_kwargs)
        
        # Project GNN output back to hidden_dim if it changed due to heads
        if self.gnn_output_dim != hidden_dim:
            self.gnn_output_proj = nn.Linear(self.gnn_output_dim, hidden_dim, bias=False)
        else:
            self.gnn_output_proj = None

    @staticmethod
    def _build_augmented_edges(
        edge_index: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        src, tgt = edge_index[0], edge_index[1]
        cross_edges = torch.stack([src, tgt + num_nodes], dim=0)
        return torch.cat([edge_index, cross_edges], dim=1)

    @staticmethod
    def _build_augmented_edge_attr(
        edge_attr: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if edge_attr is None:
            return None
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(1)
        return torch.cat([edge_attr, edge_attr], dim=0)

    def _apply_gnn(
        self,
        X_new: torch.Tensor,
        aug_edges: torch.Tensor,
        aug_edge_attr: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if aug_edge_attr is None:
            X_out = self.gnn(X_new, aug_edges)
        else:
            try:
                X_out = self.gnn(X_new, aug_edges, edge_attr=aug_edge_attr)
            except (TypeError, AssertionError):
                # If edge_attr fails (e.g., GATv2Conv not initialized with edge_dim),
                # try converting to edge_weight
                if aug_edge_attr.dim() == 2 and aug_edge_attr.size(1) == 1:
                    edge_weight = aug_edge_attr[:, 0]
                elif aug_edge_attr.dim() == 1:
                    edge_weight = aug_edge_attr
                else:
                    edge_weight = aug_edge_attr.mean(dim=1)

                try:
                    X_out = self.gnn(X_new, aug_edges, edge_weight=edge_weight)
                except TypeError:
                    # If edge_weight also fails, just use the GNN without edge info
                    X_out = self.gnn(X_new, aug_edges)
        
        # Project back to hidden_dim if needed (e.g., after multi-head attention)
        if self.gnn_output_proj is not None:
            X_out = self.gnn_output_proj(X_out)
        
        return X_out

    @staticmethod
    def _sinusoidal_positional_encoding(
        seq_len: int,
        dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        position = torch.arange(seq_len, device=device, dtype=dtype).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, device=device, dtype=dtype)
            * (-torch.log(torch.tensor(10000.0, device=device, dtype=dtype)) / dim)
        )

        pe = torch.zeros(1, seq_len, dim, device=device, dtype=dtype)
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(
        self,
        X_i: torch.Tensor,               # [N, input_dim]
        edge_index: torch.Tensor,        # [2, E]
        history_prev: torch.Tensor,      # [N, S_prev, hidden_dim]
        edge_attr: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        X_prime[:N] : GNN output for original nodes. [N, hidden_dim]
        history_i   : Updated temporal history.      [N, S_i, hidden_dim]
        """
        if self.input_dim != self.hidden_dim:
            X_proj = self.input_proj(X_i)
        else:
            X_proj = X_i

        # Append this step's token and keep a fixed-size rolling window.
        history_i = torch.cat([history_prev, X_proj.unsqueeze(1)], dim=1)
        history_i = history_i[:, -self.max_history :, :]

        pos = self._sinusoidal_positional_encoding(
            seq_len=history_i.size(1),
            dim=self.hidden_dim,
            device=history_i.device,
            dtype=history_i.dtype,
        )

        enc = self.temporal_encoder(history_i + pos)
        H_i = enc[:, -1, :]

        X_new = torch.cat([X_proj, H_i], dim=0)
        aug_edges = self._build_augmented_edges(edge_index, X_i.size(0))
        aug_edge_attr = self._build_augmented_edge_attr(edge_attr)
        X_prime = self._apply_gnn(X_new, aug_edges, aug_edge_attr)

        return X_prime[: X_i.size(0)], history_i


class TemporalTransformerGNNModel(nn.Module):
    """
    Stacked temporal GNN model where temporal updates are done with Transformer.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        gnn_type: str = "GCNConv",
        dropout: float = 0.0,
        transformer_heads: int = 4,
        transformer_layers: int = 1,
        transformer_ff_dim: Optional[int] = None,
        transformer_dropout: float = 0.1,
        max_history: int = 32,
        **gnn_kwargs,
    ) -> None:
        super().__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.layers = nn.ModuleList()
        for idx in range(num_layers):
            in_dim = input_dim if idx == 0 else hidden_dim
            self.layers.append(
                TemporalTransformerGNNLayer(
                    input_dim=in_dim,
                    hidden_dim=hidden_dim,
                    gnn_type=gnn_type,
                    transformer_heads=transformer_heads,
                    transformer_layers=transformer_layers,
                    transformer_ff_dim=transformer_ff_dim,
                    transformer_dropout=transformer_dropout,
                    max_history=max_history,
                    **gnn_kwargs,
                )
            )

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        
        # Per-node linear readout applied to the final layer's output.
        # Final layer always outputs hidden_dim due to projection in TemporalTransformerGNNLayer
        self.readout = nn.Linear(hidden_dim, output_dim)

    def init_history(self, N: int, device: torch.device) -> List[torch.Tensor]:
        """
        Initialize empty temporal histories for each layer.
        """
        return [
            torch.zeros(N, 0, self.hidden_dim, device=device)
            for _ in range(self.num_layers)
        ]

    def forward_step(
        self,
        X: torch.Tensor,
        edge_index: torch.Tensor,
        history_list: List[torch.Tensor],
        edge_attr: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        N = X.size(0)
        new_history: List[torch.Tensor] = []

        for idx, layer in enumerate(self.layers):
            X_prime, hist_i = layer(X, edge_index, history_list[idx], edge_attr=edge_attr)
            new_history.append(hist_i)

            X = X_prime[:N]
            X = self.act(X)
            if idx < self.num_layers - 1:
                X = self.dropout(X)

        out = self.readout(X)
        return out, new_history

    def forward(
        self,
        X_seq: List[torch.Tensor],
        edge_seq: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        T = len(X_seq)
        N = X_seq[0].size(0)
        device = X_seq[0].device

        history_list = self.init_history(N, device)
        outputs: List[torch.Tensor] = []

        for t in range(T):
            out, history_list = self.forward_step(
                X_seq[t], edge_seq[t], history_list
            )
            outputs.append(out)

        return out, outputs


# ──────────────────────────────────────────────────────────────────────────────
# Quick sanity-check / usage example
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(0)

    # ── Hyper-parameters ─────────────────────────────────────────────
    N          = 10    # nodes
    T          = 5     # time steps
    input_dim  = 16
    hidden_dim = 32
    output_dim = 4
    E          = 20    # edges per snapshot

    device = torch.device("cpu")

    # ── Synthetic data ────────────────────────────────────────────────
    X_seq    = [torch.randn(N, input_dim) for _ in range(T)]
    edge_seq = [
        torch.randint(0, N, (2, E))   # random edges; replace with real data
        for _ in range(T)
    ]

    # ── Build LSTM-based model ────────────────────────────────────────
    model = TemporalGNNModel(
        input_dim  = input_dim,
        hidden_dim = hidden_dim,
        output_dim = output_dim,
        num_layers = 2,
        gnn_type   = "GCNConv",   # swap to 'gat' or 'sage' freely
        dropout    = 0.1,
    )
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")

    # ── Forward pass over the full sequence ───────────────────────────
    out, outputs = model(X_seq, edge_seq)

    print(f"\nOutputs: {len(outputs)} steps × {list(out.shape)}")
    print("Last-step output (first 3 nodes):")
    print(outputs[-1][:3].detach())

    # ── Single-step interface (e.g. for online / streaming use) ───────
    H, C = model.init_hidden(N, device)
    step_out, H, C = model.forward_step(X_seq[0], edge_seq[0], H, C)
    print(f"\nSingle-step output shape: {step_out.shape}")

    # ── Build Transformer-based model ─────────────────────────────────
    tf_model = TemporalTransformerGNNModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=2,
        gnn_type="GCNConv",
        dropout=0.1,
        transformer_heads=4,
        transformer_layers=1,
        max_history=8,
    )
    tf_out, tf_outputs = tf_model(X_seq, edge_seq)
    print(f"Transformer outputs: {len(tf_outputs)} steps × {list(tf_out.shape)}")