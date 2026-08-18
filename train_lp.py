import torch
import torch.nn as nn
from torch_geometric.data import Data
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from evaluate_lp import (
    aggregate,
    evaluate_task,
    get_negative_samples,
    per_source_mrr,
)

class StatefulGNNWrapper(nn.Module):
    def __init__(self, base_model, num_nodes: int, device: torch.device,
                 use_node_emb: bool = True, node_emb_dim: int = 128,
                 decoder: str = "dot", out_dim: Optional[int] = None,
                 edge_encoder_dim: int = 0):
        super().__init__()
        self.base_model = base_model
        self.num_nodes = num_nodes
        self.device = device

        self.use_node_emb = use_node_emb
        if self.use_node_emb:
            self.node_emb = nn.Embedding(num_nodes, node_emb_dim)
            nn.init.xavier_uniform_(self.node_emb.weight)

        # Link-prediction head. "dot" is the paper's inner-product decoder
        # (eq. 8); "mlp" scores [z_u || z_v || z_u * z_v], which is what the
        # ROLAND baselines use and is strictly more expressive.
        if decoder not in ("dot", "mlp"):
            raise ValueError(f"decoder must be 'dot' or 'mlp', got {decoder!r}")
        self.decoder = decoder
        if decoder == "mlp":
            d = out_dim or node_emb_dim
            self.edge_mlp = nn.Sequential(
                nn.Linear(3 * d, d), nn.ReLU(), nn.Linear(d, 1)
            )

        # Learnable edge encoder. Edges cannot have a lookup-table embedding
        # (each edge instance is unique, and edges are what we predict), so
        # this is a small MLP over the dataset's raw edge attributes -- the
        # same role as ROLAND's `edge_encoder_name: roland_general`, and the
        # same two-layer form the paper already uses for edge features in the
        # node-classification model. Softplus keeps the resulting GCN edge
        # weight positive so the normalisation stays well defined.
        self.edge_encoder = None
        if edge_encoder_dim > 0:
            self.edge_encoder = nn.Sequential(
                nn.Linear(edge_encoder_dim, node_emb_dim), nn.ReLU(),
                nn.Linear(node_emb_dim, 1), nn.Softplus(),
            )

        self.is_lstm = hasattr(self.base_model, 'init_hidden')
        self.reset_states()

    def score_edges(self, z: torch.Tensor, src: torch.Tensor,
                    dst: torch.Tensor) -> torch.Tensor:
        """Score node pairs (src, dst) from node representations ``z``."""
        if self.decoder == "dot":
            return (z[src] * z[dst]).sum(dim=-1)
        zu, zv = z[src], z[dst]
        return self.edge_mlp(torch.cat([zu, zv, zu * zv], dim=-1)).squeeze(-1)

    def reset_states(self):
        if self.is_lstm:
            self.H_list, self.C_list = self.base_model.init_hidden(self.num_nodes, self.device)
            self.history_list = None
        else:
            self.H_list, self.C_list = None, None
            self.history_list = self.base_model.init_history(self.num_nodes, self.device)

    def detach_states(self):
        if self.H_list is not None:
            self.H_list = [h.detach() for h in self.H_list]
        if self.C_list is not None:
            self.C_list = [c.detach() for c in self.C_list]
        if hasattr(self, 'history_list') and self.history_list is not None:
            self.history_list = [h.detach() for h in self.history_list]

    def clear_states(self):
        self.H_list, self.C_list, self.history_list = None, None, None

    def clone_states(self):
        if self.is_lstm:
            h = [h_i.detach().clone() for h_i in self.H_list] if self.H_list else None
            c = [c_i.detach().clone() for c_i in self.C_list] if self.C_list else None
            return h, c
        else:
            hist = [h_i.detach().clone() for h_i in self.history_list] if self.history_list else None
            return hist, None

    def load_states(self, state1, state2):
        if self.is_lstm:
            self.H_list = state1
            self.C_list = state2
        else:
            self.history_list = state1

    def forward(self, x, edge_index, edge_attr=None, update_state=True):
        if (self.is_lstm and self.H_list is None) or (not self.is_lstm and getattr(self, 'history_list', None) is None):
            self.reset_states()

        if self.use_node_emb:
            node_ids = torch.arange(self.num_nodes, device=self.device)
            x = self.node_emb(node_ids)

        if self.edge_encoder is not None and edge_attr is not None:
            edge_attr = self.edge_encoder(edge_attr)

        if self.is_lstm:
            out, new_H, new_C = self.base_model.forward_step(x, edge_index, self.H_list, self.C_list, edge_attr=edge_attr)
            if update_state:
                self.H_list = new_H
                self.C_list = new_C
        else:
            out, new_history = self.base_model.forward_step(x, edge_index, self.history_list, edge_attr=edge_attr)
            if update_state:
                self.history_list = new_history

        return out


# ---------------------------------------------------------------------------
# ROLAND task (t, t+1): message passing on G_t, supervision on the edges of
# G_{t+1}. See graphgym/contrib/train/train_live_update.py::get_task_batch.
# ---------------------------------------------------------------------------

def _task_loss(model, node_emb: torch.Tensor, target_ei: torch.Tensor,
               num_nodes: int, train_num_neg: int = 1) -> torch.Tensor:
    """Margin-ranking loss (paper eq. 9).

    ``train_num_neg`` negatives per positive; each positive is repeated to line
    up against its own negatives, so the ranking target stays pairwise. Using
    more than one negative narrows the gap to the 1000-way ranking used at
    evaluation time.
    """
    pos = model.score_edges(node_emb, target_ei[0], target_ei[1])
    neg_ei = get_negative_samples(target_ei, num_nodes, num_neg=train_num_neg,
                                  forbidden=target_ei)
    neg = model.score_edges(node_emb, neg_ei[0], neg_ei[1])
    if train_num_neg > 1:
        pos = pos.repeat_interleave(train_num_neg)
    return torch.nn.functional.margin_ranking_loss(
        pos, neg, torch.ones_like(pos), margin=1.0
    )


def _edge_label_split(num_edges: int, ratios: Tuple[float, float, float],
                      generator: torch.Generator,
                      device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """DeepSNAP's ``GraphDataset.split(transductive=True, shuffle=True)``.

    Randomly partitions a snapshot's edges into train/val/test *label* sets.
    Message passing still uses every edge (``edge_train_mode: all``); only the
    supervision and evaluation targets are split. Returns (train_idx, test_idx).
    """
    perm = torch.randperm(num_edges, generator=generator).to(device)
    n_tr = int(num_edges * ratios[0])
    n_va = int(num_edges * (ratios[0] + ratios[1]))
    return perm[:n_tr], perm[n_va:]


def _summarise(mrrs: List[float], recalls: List[Dict[int, float]],
               per_snapshot: List[Any]) -> Dict[str, Any]:
    out = {"mrr": aggregate(mrrs), "num_eval_snapshots": len(mrrs)}
    for k in (1, 3, 10):
        out[f"recall@{k}"] = aggregate([r[k] for r in recalls]) if recalls else 0.0
    out["per_snapshot_mrr"] = per_snapshot
    return out


def train_live_update(snapshots: List[Data], model: StatefulGNNWrapper,
                      optimizer: torch.optim.Optimizer,
                      epochs_per_snapshot: int = 1,
                      num_neg: int = 1000,
                      train_target: str = "next",
                      train_num_neg: int = 1,
                      start_compute_mrr: int = 0,
                      edge_split: Optional[Tuple[float, float, float]] = None,
                      min_edges: int = 0,
                      seed: int = 0) -> Dict[str, Any]:
    """ROLAND live-update.

    At snapshot t the model (a) predicts the edges of G_{t+1} from G_t using
    states that carry information up to t-1, (b) is then trained for a fixed
    budget of inner epochs, and (c) finally rolls its recurrent state forward
    over G_t. Step (a) always precedes (b), so the ground truth for G_{t+1} is
    never observed before it is scored.

    ``start_compute_mrr`` reproduces ROLAND's ``live_update_fixed_split`` mode
    (the one their table-2 / "fixed-split" configs actually use): training runs
    incrementally over *every* snapshot, but MRR is only accumulated for tasks
    whose target index is >= this value. 0 means report over all snapshots,
    which is the plain live-update setting.

    ``edge_split`` reproduces ROLAND's ``split_method: default`` for live-update
    (their table-3 configs): each snapshot's edges are randomly partitioned
    train/val/test, supervision uses the train share and MRR is reported on the
    disjoint test share. Message passing always uses every edge. ``None`` keeps
    the previous behaviour of using all edges for both.

    ``min_edges`` drops snapshots with fewer than this many edges, matching the
    ``num_edges >= 10`` filter ROLAND's loaders apply for the default split.

    ``train_target`` selects the supervision only; evaluation is the ROLAND
    task either way.

    * ``"next"``    - supervise on E_{t+1}: train and test objectives match.
    * ``"current"`` - supervise on E_t (reconstruction), test on E_{t+1}
      (forecasting). Note the encoder sees E_t while being supervised on it,
      so this objective is trained through an in-graph target; that is a
      deliberate design choice here, not an evaluation leak, because the
      reported MRR is always computed on the held-out E_{t+1}.
    """
    if train_target not in ("next", "current"):
        raise ValueError(f"train_target must be 'next' or 'current', got {train_target!r}")
    print("Starting Live-Update Training & Evaluation (ROLAND task t -> t+1)...")
    device = model.device
    mrrs: List[float] = []
    recalls: List[Dict[int, float]] = []
    per_snapshot: List[Any] = []

    if min_edges > 0:
        kept = [s for s in snapshots if s.edge_index.size(1) >= min_edges]
        print(f"  snapshot filter (>= {min_edges} edges): "
              f"{len(snapshots)} -> {len(kept)}")
        snapshots = kept
    gen = torch.Generator().manual_seed(seed)

    snap_t = None
    for t in range(len(snapshots) - 1):
        if snap_t is None:
            snap_t = snapshots[t].to(device)
        snap_next = snapshots[t + 1].to(device)

        # ROLAND's per-snapshot edge-label split: message passing still uses
        # every edge of G_t; only the targets drawn from G_{t+1} are split.
        if edge_split is not None and snap_next.edge_index.size(1) > 0:
            tr_idx, te_idx = _edge_label_split(
                snap_next.edge_index.size(1), edge_split, gen, device)
            target_train = snap_next.edge_index[:, tr_idx]
            target_eval = snap_next.edge_index[:, te_idx]
        else:
            target_train = target_eval = snap_next.edge_index

        has_target = target_eval.size(1) > 0

        # (1) Evaluate on (t, t+1) before revealing G_{t+1}. ROLAND skips the
        # MRR computation entirely before start_compute_mrr (their `fast` flag).
        if has_target and (t + 1) >= start_compute_mrr:
            res = evaluate_task(model, snap_t, snap_next, num_neg=num_neg,
                                update_state=False, target_edge_index=target_eval)
            if res is not None:
                mrr_t, rec_t = res
                mrrs.append(mrr_t)
                recalls.append(rec_t)
                per_snapshot.append({"t": t + 1, "mrr": mrr_t})
                print(f"Snapshot Time {t + 1:03d} | Test MRR: {mrr_t:.4f}")

        if snap_t.edge_index.size(1) > 0:
            prev_1, prev_2 = model.clone_states()

            # (2) Reveal the ground truth and fine-tune.
            train_ei = (snap_t.edge_index if train_target == "current"
                        else target_train)
            if train_ei.size(1) > 0:
                model.train()
                for _ in range(epochs_per_snapshot):
                    model.load_states(*model_states_copy(prev_1, prev_2))
                    optimizer.zero_grad()
                    node_emb = model(snap_t.x, snap_t.edge_index,
                                     snap_t.edge_attr, update_state=True)
                    loss = _task_loss(model, node_emb, train_ei,
                                      snap_t.num_nodes, train_num_neg)
                    loss.backward()
                    model.detach_states()
                    optimizer.step()

            # (3) Roll the state forward over G_t with the updated model.
            # Always runs, so an empty supervision target never stalls the state.
            model.load_states(*model_states_copy(prev_1, prev_2))
            with torch.no_grad():
                model.eval()
                model(snap_t.x, snap_t.edge_index, snap_t.edge_attr,
                      update_state=True)
                model.detach_states()

        snapshots[t] = None
        snap_t = snap_next
        if str(model.device).startswith('cuda'):
            torch.cuda.empty_cache()

    summary = _summarise(mrrs, recalls, per_snapshot)
    print(f"Live-Update Complete. Average MRR over time: {summary['mrr']:.4f}")
    return summary


def model_states_copy(state1, state2):
    """Detached copies so each inner epoch restarts from the same t-1 state."""
    s1 = [s.detach().clone() for s in state1] if state1 else None
    s2 = [s.detach().clone() for s in state2] if state2 else None
    return s1, s2


def train_fixed_split(snapshots: List[Data], model: StatefulGNNWrapper,
                      optimizer: torch.optim.Optimizer, num_epochs: int = 50,
                      num_neg: int = 1000,
                      train_target: str = "next",
                      train_num_neg: int = 1,
                      val_frac: float = 0.0,
                      val_num_neg: int = 200,
                      eval_every: int = 5) -> Dict[str, Any]:
    """ROLAND task on a chronological 90/10 snapshot split.

    Training covers tasks (t, t+1) whose targets lie in the first 90% of
    snapshots; evaluation covers the tasks whose targets are the held-out final
    10%. The recurrent state runs continuously from training into evaluation,
    and message passing at test time only ever uses snapshots strictly older
    than the edges being scored.

    ``train_target`` selects the supervision only ("next" = E_{t+1},
    "current" = E_t); evaluation is the ROLAND task either way. Under
    "current" the training targets are E_0..E_{train_end-2}, still disjoint
    from the evaluation targets E_{train_end}..E_{T-1}.

    ``val_frac`` > 0 enables model selection: the last ``val_frac`` of the
    *training* tasks become a validation window that is never back-propagated
    through. Validation MRR is measured every ``eval_every`` epochs and the
    best-scoring parameters are restored before test. The test window is
    untouched by selection.
    """
    if train_target not in ("next", "current"):
        raise ValueError(f"train_target must be 'next' or 'current', got {train_target!r}")
    print("Starting Fixed-Split Training & Evaluation (ROLAND task t -> t+1)...")
    num_snaps = len(snapshots)
    train_end = int(num_snaps * 0.9)
    device = model.device

    # Tasks t in [0, train_end-2] are trainable. Carve the tail for validation.
    last_train_task = train_end - 2
    n_val = int(round(val_frac * (last_train_task + 1))) if val_frac > 0 else 0
    val_start = last_train_task + 1 - n_val if n_val > 0 else last_train_task + 1
    if n_val > 0:
        print(f"  model selection: tasks {val_start}..{last_train_task} "
              f"held out for validation ({n_val} tasks)")

    best = {"mrr": -1.0, "epoch": -1, "state": None}

    def _run_epoch(train: bool):
        """One pass over the training range; returns (mean loss, val MRRs)."""
        model.reset_states()
        total_loss, valid, val_scores = 0.0, 0, []
        for t in range(train_end - 1):
            snap_t = snapshots[t].to(device)
            snap_next = snapshots[t + 1].to(device)

            if t >= val_start:
                # Validation task: score it, then roll the state forward.
                res = evaluate_task(model, snap_t, snap_next,
                                    num_neg=val_num_neg, update_state=True)
                model.detach_states()
                if res is not None:
                    val_scores.append(res[0])
                continue

            model.train() if train else model.eval()
            optimizer.zero_grad()
            node_emb = model(snap_t.x, snap_t.edge_index, snap_t.edge_attr,
                             update_state=True)
            train_ei = (snap_t.edge_index if train_target == "current"
                        else snap_next.edge_index)
            if train_ei.size(1) == 0:
                model.detach_states()
                continue
            loss = _task_loss(model, node_emb, train_ei, snap_t.num_nodes,
                              train_num_neg)
            if train:
                loss.backward()
                model.detach_states()
                optimizer.step()
            else:
                model.detach_states()
            total_loss += loss.item()
            valid += 1
        return (total_loss / valid if valid else 0.0), val_scores

    for epoch in range(num_epochs):
        mean_loss, val_scores = _run_epoch(train=True)

        if n_val > 0 and ((epoch + 1) % eval_every == 0 or epoch == num_epochs - 1):
            v = aggregate(val_scores)
            if v > best["mrr"]:
                best = {"mrr": v, "epoch": epoch + 1,
                        "state": {k: p.detach().clone()
                                  for k, p in model.state_dict().items()}}
            print(f"Epoch {epoch+1}/{num_epochs} | Loss: {mean_loss:.4f} "
                  f"| val MRR: {v:.4f} (best {best['mrr']:.4f} @ {best['epoch']})")
        elif (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} | Loss: {mean_loss:.4f}")

    if best["state"] is not None:
        print(f"  restoring best model (val MRR {best['mrr']:.4f}, "
              f"epoch {best['epoch']})")
        model.load_state_dict(best["state"])
        # Recurrent state depends on the parameters, so rebuild it under the
        # restored weights before touching the test window.
        with torch.no_grad():
            _run_epoch(train=False)

    # Evaluation: targets are snapshots [train_end, num_snaps - 1]. The state
    # now covers G_0 ... G_{train_end-2}.
    mrrs: List[float] = []
    recalls: List[Dict[int, float]] = []
    per_snapshot: List[Any] = []

    for t in range(train_end - 1, num_snaps - 1):
        snap_t = snapshots[t].to(device)
        snap_next = snapshots[t + 1].to(device)
        res = evaluate_task(model, snap_t, snap_next, num_neg=num_neg,
                            update_state=True)
        model.detach_states()
        if res is None:
            continue
        mrr_t, rec_t = res
        mrrs.append(mrr_t)
        recalls.append(rec_t)
        per_snapshot.append({"t": t + 1, "mrr": mrr_t})

    summary = _summarise(mrrs, recalls, per_snapshot)
    if best["state"] is not None:
        summary["selected_epoch"] = best["epoch"]
        summary["val_mrr"] = best["mrr"]
    print(f"Fixed-Split Evaluation | Final 10% MRR: {summary['mrr']:.4f}")
    return summary
