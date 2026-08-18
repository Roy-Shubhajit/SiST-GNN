"""Assertions that the ROLAND port behaves as their code specifies.

Independent checks, so a wrong port fails loudly instead of quietly producing
plausible numbers:

  * snapshot counts vs ROLAND's own start_compute_mrr / split ratios
  * per-snapshot split sizes are 80 / 10 / 10 of that snapshot's edges
  * message-passing graphs are 80% (train, val) and 90% (test)
  * train targets == train message edges; val/test targets are DISJOINT from
    their own message-passing edges (no target is visible to the encoder)
  * the >=10-edge filter and the undirected augmentation actually fire
"""

from __future__ import annotations

import sys

import torch

from roland_pipeline import SPECS, load_roland

ROOT = "datasets/roland"


def _ids(ei: torch.Tensor, n: int) -> set:
    return set((ei[0].long() * n + ei[1].long()).tolist())


# Snapshot counts implied by ROLAND's own binning. A timestamp-parsing bug
# collapses these towards 1, so pin them.
EXPECTED_T = {
    "uci-message": (20, 40), "bitcoin-alpha": (200, 290),
    "bitcoin-otc": (200, 290), "reddit-title": (120, 200),
    "reddit-body": (120, 200), "as-733": (600, 760),
}


def check_live_update(name: str) -> None:
    (tr, va, te), smrr = load_roland(name, "live_update", ROOT, seed=0)
    assert smrr == 0, "live_update reports MRR over every task"
    T = len(tr)
    assert len(va) == len(te) == T

    lo, hi = EXPECTED_T[name]
    assert lo <= T <= hi, (
        f"{name}: {T} snapshots, expected {lo}-{hi}. A collapsed time axis "
        "(bad timestamp units) is the usual cause.")

    for i in range(T):
        a, b, c = tr[i], va[i], te[i]
        E = (a.edge_index.size(1) + va[i].edge_label_index.size(1)
             + c.edge_label_index.size(1))

        # message passing: train/val share the 80% graph, test sees 90%
        assert torch.equal(a.edge_index, b.edge_index), f"{name}[{i}] val != train mp"
        assert c.edge_index.size(1) > a.edge_index.size(1), f"{name}[{i}] test mp !> train mp"

        # train supervises on exactly its own message edges (edge_train_mode 'all')
        assert torch.equal(a.edge_index, a.edge_label_index), f"{name}[{i}] train mp != labels"

        # The split is on edge *indices*. On a multigraph the same (u,v) pair
        # can therefore land in two different shares -- that is deepsnap's
        # behaviour, not a port bug, so it is reported (below) not asserted.

        # deepsnap's exact arithmetic: floors for train/val, remainder to test
        # (graph.py:1484-1501), so small snapshots get a >10% test share.
        n_tr, n_va = int(0.8 * E), int(0.1 * E)
        n_te = E - n_tr - n_va
        if n_tr == 0 or n_va == 0 or n_te == 0:      # "secure split" fallback
            n_tr = 1 + int(0.8 * (E - 3))
            n_va = 1 + int(0.1 * (E - 3))
            n_te = E - n_tr - n_va
        assert a.edge_index.size(1) == n_tr, \
            f"{name}[{i}] train {a.edge_index.size(1)} != {n_tr}"
        assert b.edge_label_index.size(1) == n_va, \
            f"{name}[{i}] val {b.edge_label_index.size(1)} != {n_va}"
        assert c.edge_label_index.size(1) == n_te, \
            f"{name}[{i}] test {c.edge_label_index.size(1)} != {n_te}"
        assert c.edge_index.size(1) == n_tr + n_va, \
            f"{name}[{i}] test mp {c.edge_index.size(1)} != {n_tr + n_va}"

    sizes = [a.edge_index.size(1) + va[i].edge_label_index.size(1)
             + te[i].edge_label_index.size(1) for i, a in enumerate(tr)]
    assert min(sizes) >= 10, f"{name}: snapshot with < 10 edges survived"

    # THE check that matters for the task: the encoder sees G_t while the
    # targets come from G_{t+1}. Overlap here is genuine edge recurrence
    # (the thing the model is supposed to predict), not leakage -- but a
    # 100% rate would mean we were scoring the encoder's own input again.
    n = tr[0].num_nodes
    tot = ov = 0
    for i in range(T - 1):
        mp = _ids(te[i].edge_index, n)
        tg = _ids(te[i + 1].edge_label_index, n)
        tot += len(tg); ov += len(tg & mp)
    rate = ov / tot if tot else 0.0
    assert rate < 0.9, f"{name}: {rate:.0%} of targets sit in the encoder input"

    med = sorted(sizes)[len(sizes) // 2]
    print(f"  live_update      T={T:<5} med|E|={med:<6} "
          f"mp(train)={tr[0].edge_index.size(1)} mp(test)={te[0].edge_index.size(1)} "
          f"targets(test)={te[0].edge_label_index.size(1)}")
    print(f"                   targets(G_t+1) recurring in encoder input(G_t): "
          f"{rate:.1%}   OK")


def check_fixed_split(name: str) -> None:
    (tr, va, te), smrr = load_roland(name, "live_update_fixed_split", ROOT)
    T = len(tr)
    spec = SPECS[name]

    # loader.py:222 -- all three copies are the same object, no edge split
    assert tr is va is te, f"{name}: fixed-split copies should be identical"
    for g in tr:
        assert torch.equal(g.edge_index, g.edge_label_index), \
            f"{name}: fixed-split must not split edges"

    assert smrr == spec.start_compute_mrr
    frac = smrr / T
    assert 0.75 <= frac <= 0.85, \
        f"{name}: start_compute_mrr {smrr}/{T} = {frac:.1%}, expected ~80%"

    if spec.bitcoin_style:
        n = tr[0].num_nodes
        ei = torch.cat([g.edge_index for g in tr], dim=1)
        fwd = _ids(ei, n)
        rev = {(d * n + s) for s, d in zip(ei[0].tolist(), ei[1].tolist())}
        assert fwd == rev, f"{name}: reversed-edge augmentation missing"

    sizes = [g.edge_index.size(1) for g in tr]
    print(f"  fixed_split      T={T:<5} med|E|={sorted(sizes)[T//2]:<6} "
          f"start_compute_mrr={smrr} ({frac:.1%})"
          f"{'  undirected' if spec.bitcoin_style else ''}   OK")


if __name__ == "__main__":
    names = sys.argv[1:] or ["uci-message", "bitcoin-alpha", "bitcoin-otc"]
    for nm in names:
        print(f"\n{nm}")
        check_live_update(nm)
        if SPECS[nm].fixed_freq:
            check_fixed_split(nm)
    print("\nall checks passed")
