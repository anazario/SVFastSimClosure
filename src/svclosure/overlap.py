"""Event-level category checks: which selections share events, and which events enter none.

Expected relations (from LLPCombine's category logic; ``overlaps`` block of the study YAML):

=====================  ==================================================  =====================
relation               pair                                                 expectation
=====================  ==================================================  =====================
within_channel         two kinematic bins of one channel group              disjoint (REQUIRED)
highlow_partner        a high-S_xy bin and its low-S_xy partner             disjoint (expected)
cross_channel          bins of different channel groups, same mode          disjoint (expected:
                                                                            one simultaneous fit)
cross_mode             compressed vs high-Delta-m bins                      overlap allowed
=====================  ==================================================  =====================

A within-channel overlap is a structural failure (the scan stops).  The others are reported
with their selections; ``fail_on_unexpected`` turns them into failures too.

Omissions: events in a channel's ``group_envelope`` (all cuts shared by its bins) that enter
none of its bins.  The static coverage analysis (bins.coverage_gaps) explains part of them as
kinematic ranges no bin covers by construction (e.g. R_ISR < 0.4 in the delayed-photon
channels); ``omission_unexplained`` is what remains and is reported as unexpected.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EXPECT = {"within_channel": "disjoint_required", "highlow_partner": "disjoint_expected",
          "cross_channel": "disjoint_expected", "cross_mode": "allowed"}


def relation(meta: dict, a: tuple[str, str], b: tuple[str, str]) -> str:
    ma, mb = meta[a], meta[b]
    if a[0] != b[0]:
        return "cross_mode"
    if ma["group"] == mb["group"]:
        return "within_channel"
    if ma["partner"] == b[1]:
        return "highlow_partner"
    return "cross_channel"


def classify(overlaps: pd.DataFrame, bin_meta: pd.DataFrame, variant: str = "noreweight") -> pd.DataFrame:
    """Bin-pair overlaps (n_ab > 0) with their relation, expectation and verdict."""
    if overlaps is None or not len(overlaps):
        return pd.DataFrame()
    meta = {(r["mode"], r["name"]): r for r in bin_meta.to_dict("records")}
    o = overlaps[(overlaps.kind_a == "bin") & (overlaps.kind_b == "bin") & (overlaps.variant == variant)].copy()
    if not len(o):
        return o
    o["relation"] = [relation(meta, (ma, a), (mb, b)) for ma, a, mb, b in zip(o.mode_a, o.sel_a, o.mode_b, o.sel_b)]
    o["expectation"] = o.relation.map(EXPECT)
    o["frac_of_smaller"] = o.n_ab / np.minimum(o.n_a, o.n_b)
    o["verdict"] = np.select([o.expectation == "disjoint_required", o.expectation == "disjoint_expected"],
                             ["structural_failure", "unexpected_overlap"], "allowed")
    o["group_a"] = [meta[(m, s)]["group"] for m, s in zip(o.mode_a, o.sel_a)]
    o["group_b"] = [meta[(m, s)]["group"] for m, s in zip(o.mode_b, o.sel_b)]
    return o.sort_values(["verdict", "point", "sim", "hlt", "n_ab"], ascending=[True, True, True, True, False])


def group_matrix(overlaps: pd.DataFrame, yields: pd.DataFrame, bin_meta: pd.DataFrame, point: str, sim: str,
                 hlt: str = "parity", sxy: str = "high") -> pd.DataFrame:
    """Event counts in the intersection of channel groups (union of each group's bins).

    Exact because bins within a group are disjoint (checked): every event is in at most one bin
    per group, so |G1 n G2| = sum over bin pairs of |b_i n b_j|.
    """
    bm = bin_meta[bin_meta.sxy == sxy]
    gkey = {(r["mode"], r["name"]): f"{r['mode']}:{r['group']}" for r in bm.to_dict("records")}
    groups = list(dict.fromkeys(gkey.values()))
    M = pd.DataFrame(0, index=groups, columns=groups, dtype=np.int64)
    y = yields[(yields.point == point) & (yields.sim == sim) & (yields.hlt == hlt) & (yields.kind == "bin")
               & (yields.variant == "noreweight")]
    for r in y.itertuples():
        g = gkey.get((r.mode, r.bin))
        if g:
            M.loc[g, g] += r.n_raw
    o = overlaps[(overlaps.point == point) & (overlaps.sim == sim) & (overlaps.hlt == hlt)
                 & (overlaps.variant == "noreweight") & (overlaps.kind_a == "bin") & (overlaps.kind_b == "bin")]
    for r in o.itertuples():
        ga, gb = gkey.get((r.mode_a, r.sel_a)), gkey.get((r.mode_b, r.sel_b))
        if ga and gb and ga != gb:
            M.loc[ga, gb] += r.n_ab
            M.loc[gb, ga] += r.n_ab
    return M


def bin_matrix(overlaps: pd.DataFrame, yields: pd.DataFrame, bin_meta: pd.DataFrame, point: str, sim: str,
               hlt: str = "parity", sxy: str = "high") -> pd.DataFrame:
    """Pairwise event-count overlap matrix of the bins of one S_xy class (diagonal = bin count)."""
    bm = bin_meta[bin_meta.sxy == sxy].sort_values(["mode", "order"])
    names = [f"{m}:{n}" for m, n in zip(bm["mode"], bm.name)]
    M = pd.DataFrame(0, index=names, columns=names, dtype=np.int64)
    y = yields[(yields.point == point) & (yields.sim == sim) & (yields.hlt == hlt) & (yields.kind == "bin")
               & (yields.variant == "noreweight")]
    for r in y.itertuples():
        k = f"{r.mode}:{r.bin}"
        if k in M.index:
            M.loc[k, k] = r.n_raw
    o = overlaps[(overlaps.point == point) & (overlaps.sim == sim) & (overlaps.hlt == hlt)
                 & (overlaps.variant == "noreweight")]
    for r in o.itertuples():
        a, b = f"{r.mode_a}:{r.sel_a}", f"{r.mode_b}:{r.sel_b}"
        if a in M.index and b in M.index:
            M.loc[a, b] = M.loc[b, a] = r.n_ab
    return M


def omissions(yields: pd.DataFrame, refs: list) -> pd.DataFrame:
    """Per channel envelope: events in no bin, the part explained by static kinematic gaps, the rest."""
    y = yields[yields.variant == "noreweight"]
    keyed = y.set_index(["mode", "bin", "point", "sim", "hlt"])
    rows = []
    for r in refs:
        if r.kind != "group_envelope":
            continue
        for (pt, sim, hlt), _ in y[(y["mode"] == r.mode) & (y.bin == r.name)].groupby(["point", "sim", "hlt"]):
            def get(name, col="n_raw"):
                try:
                    return keyed.loc[(r.mode, name, pt, sim, hlt), col]
                except KeyError:
                    return 0
            n_ref, n_none = get(r.name), get(r.name + "|none")
            n_unexp = get(r.name + "|none_unexplained") if r.gap_expression else n_none
            rows.append(dict(mode=r.mode, envelope=r.name, point=pt, sim=sim, hlt=hlt, n_envelope=int(n_ref),
                             n_in_no_bin=int(n_none), sumw_in_no_bin=float(get(r.name + "|none", "sumw")),
                             n_unexplained=int(n_unexp), static_gaps="; ".join(r.gaps) or "none",
                             frac_in_no_bin=(n_none / n_ref) if n_ref else np.nan,
                             verdict="unexpected_omission" if n_unexp else ("explained_by_binning" if n_none else "none"),
                             members=";".join(r.members)))
    return pd.DataFrame(rows)


def summarize(cls: pd.DataFrame, om: pd.DataFrame) -> dict:
    out = {"bin_pairs_overlapping": {}, "structural_failures": [], "unexpected_overlaps": [], "unexpected_omissions": []}
    if cls is not None and len(cls):
        for rel, g in cls.groupby("relation"):
            out["bin_pairs_overlapping"][rel] = dict(n_rows=len(g), n_pairs=g[["mode_a", "sel_a", "mode_b", "sel_b"]]
                                                     .drop_duplicates().shape[0], max_n_ab=int(g.n_ab.max()))
        for verdict, key in (("structural_failure", "structural_failures"), ("unexpected_overlap", "unexpected_overlaps")):
            g = cls[cls.verdict == verdict]
            out[key] = (g.groupby(["relation", "mode_a", "sel_a", "sel_b", "hlt"], as_index=False)
                        .agg(n_ab=("n_ab", "sum"), points=("point", "nunique")).to_dict("records"))
    if om is not None and len(om):
        g = om[om.verdict == "unexpected_omission"]
        out["unexpected_omissions"] = (g.groupby(["mode", "envelope", "hlt"], as_index=False)
                                       .agg(n=("n_unexplained", "sum"), points=("point", "nunique")).to_dict("records"))
        out["explained_omissions"] = (om[om.verdict == "explained_by_binning"].groupby(["mode", "envelope", "static_gaps"],
                                      as_index=False).agg(n=("n_in_no_bin", "sum")).to_dict("records"))
    return out
