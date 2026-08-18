"""Direct port of ROLAND's data pipeline. Our model is the only thing swapped in.

Every function here transcribes a specific piece of snap-stanford/roland (and
the DeepSNAP calls it relies on) rather than re-deriving the behaviour from the
YAML configs. Source references are given per function so the port can be
audited line by line.

Pieces ported
-------------
1. ``make_graph_snapshot`` / ``split_by_seconds``
       graphgym/contrib/loader/roland_btc.py, roland_ucimsg.py, roland_as.py
2. reversed-edge augmentation for the bitcoin datasets
       roland_btc.py::load_single_dataset, applied when
       ``cfg.train.mode in ['baseline','baseline_v2','live_update_fixed_split']``
3. the ``num_edges >= 10`` snapshot filter
       roland_btc.py::load_btc_dataset (and the sibling loaders), applied when
       ``cfg.dataset.split_method != 'chronological_temporal'``
4. the per-snapshot train/val/test edge split
       graphgym/loader.py:241 -> deepsnap GraphDataset.split(transductive=True)
       -> deepsnap/graph.py::split_link_pred

What the two ROLAND settings actually are
-----------------------------------------
``live_update``            (their table 3 == our Table 4)
    split_method 'default', split [0.8, 0.1, 0.1] applied *per snapshot*.
    Three parallel sequences; MRR read off the test one, training off the
    train one. Snapshot frequency W (D for AS-733), directed.

``live_update_fixed_split`` (their table 2 == our Table 5)
    split_method 'chronological_temporal', and graphgym/loader.py:222 sets
    ``datasets = [dataset, dataset, dataset]`` -- i.e. NO edge split at all.
    Instead MRR is simply suppressed until ``cfg.train.start_compute_mrr``.
    Snapshot frequency is a raw second count, and bitcoin is made undirected.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

# --------------------------------------------------------------------------
# Per-dataset settings, read from run/replication_configs/{table2,table3_bottom}
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RolandSpec:
    file: str
    sep: str
    cols: Tuple[str, ...]
    live_freq: str          # snapshot_freq for the live_update configs
    fixed_freq: Optional[str] = None   # snapshot_freq for the table-2 configs
    start_compute_mrr: Optional[int] = None
    bitcoin_style: bool = False        # gets the reversed-edge augmentation


SPECS: Dict[str, RolandSpec] = {
    "bitcoin-alpha": RolandSpec(
        "soc-sign-bitcoinalpha.csv", ",", ("src", "dst", "rating", "time"),
        live_freq="W", fixed_freq="1200000s", start_compute_mrr=109,
        bitcoin_style=True),
    "bitcoin-otc": RolandSpec(
        "soc-sign-bitcoinotc.csv", ",", ("src", "dst", "rating", "time"),
        live_freq="W", fixed_freq="1200000s", start_compute_mrr=110,
        bitcoin_style=True),
    "uci-message": RolandSpec(
        "CollegeMsg.txt", " ", ("src", "dst", "time"),
        live_freq="W", fixed_freq="190080s", start_compute_mrr=72),
    "reddit-title": RolandSpec(
        "soc-redditHyperlinks-title.tsv", "\t", (), live_freq="W"),
    "reddit-body": RolandSpec(
        "soc-redditHyperlinks-body.tsv", "\t", (), live_freq="W"),
    "as-733": RolandSpec("", "", (), live_freq="D"),
}


# --------------------------------------------------------------------------
# 1. Raw edge list
# --------------------------------------------------------------------------

def _read_edges(name: str, data_root: str
                ) -> Tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    """Return (edge_index[2, E] with 0..N-1 ids, edge_time[E], num_nodes,
    edge_feature[E, d]).

    Node ids are assigned with ``OrdinalEncoder(categories=sorted unique)`` in
    ROLAND's loaders, i.e. by *sorted* raw id, so we sort here too.

    ``edge_feature`` mirrors the loaders' own choice, with the timestamp put
    through ``MinMaxScaler((0, 2))`` exactly as they do:
      bitcoin  -> [RATING, TimestampScaled]   (roland_btc.py:33)
      uci      -> [TimestampScaled]           (roland_ucimsg.py:42)
      reddit / as-733 -> [TimestampScaled]
    """
    spec = SPECS[name]
    root = os.path.join(data_root, name)

    rating = None
    if name == "as-733":
        # roland_as.py: one file per day, timestamp parsed from the filename.
        from datetime import datetime
        as_dir = os.path.join(root, "as-733")
        if not os.path.isdir(as_dir):
            as_dir = root
        files = sorted(f for f in os.listdir(as_dir) if f.endswith(".txt"))
        src, dst, ts = [], [], []
        for fn in files:
            t = int(datetime.strptime(fn.strip(".txt").strip("as"),
                                      "%Y%m%d").timestamp())
            a = pd.read_csv(os.path.join(as_dir, fn), sep="\t", comment="#",
                            names=["src", "dst"])
            src.append(a["src"].to_numpy()); dst.append(a["dst"].to_numpy())
            ts.append(np.full(len(a), t, dtype=np.int64))
        raw_src = np.concatenate(src); raw_dst = np.concatenate(dst)
        t = np.concatenate(ts)
    elif name.startswith("reddit"):
        # roland_reddit_hyperlink.py: tsv with named columns, ISO timestamps.
        df = pd.read_csv(os.path.join(root, spec.file), sep="\t")
        raw_src = df["SOURCE_SUBREDDIT"].to_numpy()
        raw_dst = df["TARGET_SUBREDDIT"].to_numpy()
        # roland_reddit_hyperlink.py:66-69 -- convert to epoch SECONDS via an
        # explicit Timedelta. Do NOT use .astype('int64'): pandas 2 parses
        # these strings as datetime64[us], so dividing by 1e9 silently yields
        # microseconds/1e9 and collapses the whole dataset into one bucket.
        t = ((pd.to_datetime(df["TIMESTAMP"]) - pd.Timestamp("1970-01-01"))
             // pd.Timedelta("1s")).to_numpy().astype(np.int64)
    else:
        df = pd.read_csv(os.path.join(root, spec.file), sep=spec.sep,
                         names=list(spec.cols))
        raw_src = df["src"].to_numpy(); raw_dst = df["dst"].to_numpy()
        # roland_btc.py rounds the (occasionally decimal) OTC timestamps.
        t = df["time"].to_numpy().astype(np.int64)
        if "rating" in spec.cols:
            rating = df["rating"].to_numpy()

    nodes = np.sort(pd.unique(np.concatenate([raw_src, raw_dst])))
    remap = {v: i for i, v in enumerate(nodes)}
    src = np.fromiter((remap[v] for v in raw_src), dtype=np.int64, count=len(raw_src))
    dst = np.fromiter((remap[v] for v in raw_dst), dtype=np.int64, count=len(raw_dst))

    # MinMaxScaler((0, 2)) on the raw timestamp, as in every ROLAND loader.
    lo, hi = t.min(), t.max()
    t_scaled = (t - lo) / (hi - lo) * 2.0 if hi > lo else np.zeros(len(t))
    if rating is not None:
        ef = np.stack([rating.astype(np.float64), t_scaled], axis=1)
    else:
        ef = t_scaled.reshape(-1, 1)
    return np.stack([src, dst]), t, len(nodes), ef.astype(np.float32)


# --------------------------------------------------------------------------
# 2. Snapshot construction  (roland_btc.py / roland_as.py)
# --------------------------------------------------------------------------

def _make_graph_snapshot(edge_index, t, freq: str) -> List[np.ndarray]:
    """Port of ``make_graph_snapshot``.

    Groups by (calendar year, sub-year field) where the sub-year field is
    ``%j`` for 'D', ``%W`` for 'W', ``%m`` for 'M'. NOTE this is week-of-year
    bucketing, not pandas' ``Grouper(freq='W')`` period bucketing.
    """
    fmt = {"D": "%j", "W": "%W", "M": "%m"}[freq.upper()]
    dt = pd.to_datetime(t, unit="s")
    key = pd.DataFrame({
        "year": dt.strftime("%Y").astype(int),
        "sub": dt.strftime(fmt).astype(int),
    })
    groups = key.groupby(["year", "sub"]).indices
    return [np.sort(groups[p]) for p in sorted(groups)]


def _split_by_seconds(t, freq_sec: int) -> List[np.ndarray]:
    """Port of ``split_by_seconds``: bucket on ``edge_time // freq_sec``."""
    crit = t // freq_sec
    return [np.where(crit == g)[0] for g in np.unique(np.sort(crit))]


# --------------------------------------------------------------------------
# 3. DeepSNAP's per-snapshot link-pred split  (deepsnap/graph.py)
# --------------------------------------------------------------------------

@dataclass
class SplitSnapshot:
    """One snapshot as seen by one of the three dataset copies."""
    edge_index: torch.Tensor        # message passing edges for this split
    edge_label_index: torch.Tensor  # supervision / evaluation targets
    num_nodes: int
    edge_feature: Optional[torch.Tensor] = None   # [E_mp, d] for edge_index


def _split_link_pred(edge_index: torch.Tensor, num_nodes: int,
                     ratio: Tuple[float, float, float],
                     rng: np.random.Generator,
                     edge_feature: Optional[torch.Tensor] = None,
                     ) -> List[SplitSnapshot]:
    """Port of ``deepsnap.graph.Graph.split_link_pred`` (tensor backend).

    graph.py:1484-1573. For a 3-way ratio:

        edges_train = perm[:n_tr]
        edges_val   = perm[n_tr:n_tr+n_va]
        edges_test  = perm[n_tr+n_va:]

        graph_train.edge_index = edges_train                  (80%)
        graph_val   = copy of graph_train                     (80%)
        graph_test.edge_index  = edges_train + edges_val      (90%)

        _create_label_link_pred(graph_train, edges_train)     labels = 80%
        _create_label_link_pred(graph_val,   edges_val)       labels = 10%
        _create_label_link_pred(graph_test,  edges_test)      labels = 10%

    So the *message passing* graph is restricted per split as well; only the
    training copy reuses its own share for both roles.
    """
    E = edge_index.size(1)
    perm = torch.from_numpy(rng.permutation(E))

    n_tr = int(ratio[0] * E)
    n_va = int(ratio[1] * E)
    n_te = E - n_tr - n_va
    # deepsnap's "secure split": every part must hold at least one edge.
    if n_tr == 0 or n_va == 0 or n_te == 0:
        n_tr = 1 + int(ratio[0] * (E - 3))
        n_va = 1 + int(ratio[1] * (E - 3))

    e_tr = perm[:n_tr]
    e_va = perm[n_tr:n_tr + n_va]
    e_te = perm[n_tr + n_va:]

    e_mp_test = torch.cat([e_tr, e_va])
    mp_train = edge_index[:, e_tr]
    mp_test = edge_index[:, e_mp_test]
    ef_train = None if edge_feature is None else edge_feature[e_tr]
    ef_test = None if edge_feature is None else edge_feature[e_mp_test]
    return [
        SplitSnapshot(mp_train, edge_index[:, e_tr], num_nodes, ef_train),  # train
        SplitSnapshot(mp_train, edge_index[:, e_va], num_nodes, ef_train),  # val
        SplitSnapshot(mp_test,  edge_index[:, e_te], num_nodes, ef_test),   # test
    ]


# --------------------------------------------------------------------------
# 4. Public entry point
# --------------------------------------------------------------------------

def load_roland(name: str, mode: str, data_root: str = "datasets/roland",
                seed: int = 0,
                ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1),
                undirected: Optional[bool] = None,
                min_edges: Optional[int] = None,
                ) -> Tuple[List[List[SplitSnapshot]], int]:
    """Build ROLAND's three dataset copies for ``mode``.

    Returns ``([train_seq, val_seq, test_seq], start_compute_mrr)`` where each
    sequence is a list of :class:`SplitSnapshot` over time.

    mode='live_update'
        table-3 setting: W/D bins, directed, per-snapshot 80/10/10 split,
        snapshots with < 10 edges dropped, MRR over every task.
    mode='live_update_fixed_split'
        table-2 setting AS THE CODE RUNS IT: second-count bins, bitcoin made
        undirected, NO edge split (all three copies are the same object, per
        loader.py:222), MRR suppressed before start_compute_mrr (last ~21%).

    mode='paper_fixed_split'
        the fixed-split setting AS THE PAPER DESCRIBES IT. Section 4.1:
        "Fixed-split evaluates models using all edges from the last 10% of
        snapshots." Snapshot frequency is taken from the paper's Table 1
        (weekly, daily for AS-733), which our live_update bins reproduce
        exactly. Directed, because Table 1's edge counts (35,592 / 24,186)
        are the raw undoubled figures. No edge split; MRR over the last 10%
        of snapshots.

        This differs from the table-2 configs in three ways -- bins, edge
        direction, and evaluation fraction -- so running both shows whether a
        conclusion depends on reading the paper or the code.

    ``undirected`` / ``min_edges`` override the mode's defaults. Passing the
    same values to ``paper_fixed_split`` and ``live_update`` makes the two
    settings share an identical snapshot sequence, so the only remaining
    difference is the evaluation protocol (all edges over the last 10% of
    snapshots, versus a per-snapshot 10% edge share over every snapshot).
    Note that forcing them equal is OUR choice: ROLAND symmetrises bitcoin for
    the fixed-split table only, and applies the >=10-edge filter only to the
    edge-level split.
    """
    if name not in SPECS:
        raise ValueError(f"unknown dataset {name!r}; have {sorted(SPECS)}")
    spec = SPECS[name]
    ei_np, t, num_nodes, ef_np = _read_edges(name, data_root)

    if mode not in ("live_update", "live_update_fixed_split", "paper_fixed_split"):
        raise ValueError("mode must be live_update, live_update_fixed_split "
                         "or paper_fixed_split")

    # ROLAND's own defaults, overridable so the two settings can be matched.
    default_undirected = (mode == "live_update_fixed_split" and spec.bitcoin_style)
    # The >=10-edge filter belongs to the default (edge-level) split only;
    # a chronological fixed split keeps every snapshot.
    default_min_edges = 10 if mode == "live_update" else 0
    undirected = default_undirected if undirected is None else undirected
    min_edges = default_min_edges if min_edges is None else min_edges

    if undirected:
        # roland_btc.py::load_single_dataset -- append the reversed edge set.
        ei_np = np.concatenate([ei_np, np.stack([ei_np[1], ei_np[0]])], axis=1)
        t = np.concatenate([t, t])
        ef_np = np.concatenate([ef_np, ef_np], axis=0)

    if mode == "live_update_fixed_split":
        if spec.fixed_freq is None:
            raise ValueError(f"{name} has no table-2 config")
        buckets = _split_by_seconds(t, int(spec.fixed_freq.rstrip("s")))
    else:
        # Table 1's frequency: weekly, daily for AS-733.
        buckets = _make_graph_snapshot(ei_np, t, spec.live_freq)

    ei = torch.from_numpy(ei_np).long()
    ef = torch.from_numpy(ef_np).float()
    snaps = [(ei[:, torch.from_numpy(b)], ef[torch.from_numpy(b)])
             for b in buckets]
    if min_edges:
        kept = [s for s in snaps if s[0].size(1) >= min_edges]
        print(f"  snapshot filter (>= {min_edges} edges): {len(snaps)} -> {len(kept)}")
        snaps = kept

    if mode in ("live_update_fixed_split", "paper_fixed_split"):
        # loader.py:222 -- datasets = [dataset, dataset, dataset], no split:
        # every edge is both a message-passing edge and a target.
        seq = [SplitSnapshot(s, s, num_nodes, f) for s, f in snaps]
        if mode == "paper_fixed_split":
            # Sec 4.1: "all edges from the last 10% of snapshots".
            return [seq, seq, seq], int(round(0.9 * len(seq)))
        return [seq, seq, seq], (spec.start_compute_mrr or 0)

    rng = np.random.default_rng(seed)
    per_split: List[List[SplitSnapshot]] = [[], [], []]
    for s, f in snaps:
        for i, g in enumerate(_split_link_pred(s, num_nodes, ratio, rng, f)):
            per_split[i].append(g)
    return per_split, 0


def to_data(g: SplitSnapshot, device, use_edge_feature: bool = True) -> Data:
    """Wrap a split snapshot as the PyG Data our model consumes."""
    x = torch.ones((g.num_nodes, 1), dtype=torch.float, device=device)
    ea = (g.edge_feature.to(device)
          if (use_edge_feature and g.edge_feature is not None) else None)
    return Data(x=x, edge_index=g.edge_index.to(device),
                edge_attr=ea, num_nodes=g.num_nodes)
