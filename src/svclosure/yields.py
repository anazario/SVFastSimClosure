"""Bin selection + weighted-yield aggregation over chunks (uproot backend).

HLT treatment (``hlt`` key of every yield):

* ``parity``    - fit parity with LLPCombine: Trigger_* terms kept for FullSim,
                  replaced by ``true`` for FastSim (sMCType == 2).
* ``symmetric`` - diagnostic: Trigger_* replaced by ``true`` for *both*, so the
                  FullSim/FastSim comparison isolates reconstruction effects.

Weight variants: every task carries a list of targets, ``None`` (no lifetime factor:
the original study, bit for bit) and/or target lifetimes in cm.  For each chunk the
selection masks are evaluated **once**; the per-target weights only rescale them.  Per
(selection, HLT mode, variant) the task accumulates n, sum w, sum w^2, the cross
moments sum w_i w_j between variants (covariance across targets), the largest weights
(concentration diagnostics) and the yield carried by events with invalid lifetime
inputs.  Per HLT mode it also accumulates the event-level overlap of every pair of
selections (count, sum w, sum w^2), which is what the disjointness checks and the
correlated high/low-S_xy ratios need.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import numpy as np

from . import expr
from .rootio import IOStats, iterate_chunks
from .weights import (ANOM_NEGATIVE, ANOM_NONFINITE, DECAY_BRANCHES, LIFETIME_BRANCHES, WeightSpec,
                      lifetime_anomalies, weight_matrix)

HLT_MODES = ("parity", "symmetric")
ANOM_CLASSES = ("any", "ctau_nonfinite", "ctau_negative", "ctau_zero", "decay_flags_not_one")
INVALID_FOR_POLICY = ANOM_NONFINITE | ANOM_NEGATIVE  # what 'exclude'/'fail' act on; zero/flags are recorded only
MAX_EVENT_RECORDS = 2000


@dataclass
class CompiledBin:
    mode: str
    name: str
    nodes: dict[str, expr.Node] | None  # hlt mode -> AST (None for derived omission selections)
    kind: str = "bin"  # bin | group_envelope | sxy_inclusive | omission | gap | omission_unexplained
    ref: int | None = None  # omission: index of the reference selection
    members: tuple[int, ...] = ()  # omission: indices of the selections that should cover the reference
    veto: int | None = None  # omission_unexplained: index of the static coverage-gap selection


def compile_bins(bins: list[dict], is_fastsim: bool) -> list[CompiledBin]:
    """Bins and reference selections (``kind`` key, default bin); omissions derived from references."""
    out = []
    for b in bins:
        full = expr.parse(b["expression"])
        stripped = expr.parse(b["expression_no_hlt"])
        out.append(CompiledBin(b["mode"], b["name"], {"parity": stripped if is_fastsim else full, "symmetric": stripped},
                               b.get("kind", "bin")))
    index = {(c.mode, c.name): i for i, c in enumerate(out)}
    for i, b in enumerate(bins):
        if b.get("members"):
            mem = tuple(index[(b["mode"], m)] for m in b["members"])
            out.append(CompiledBin(b["mode"], b["name"] + "|none", None, "omission", i, mem))
            if b.get("gap_expression"):
                g = expr.parse(b["gap_expression"])
                out.append(CompiledBin(b["mode"], b["name"] + "|gap", {"parity": g, "symmetric": g}, "gap"))
                out.append(CompiledBin(b["mode"], b["name"] + "|none_unexplained", None, "omission_unexplained", i,
                                       mem, len(out) - 1))
    return out


def needed_branches(compiled: list[CompiledBin], wspecs: list[WeightSpec], lifetime: bool = True) -> list[str]:
    names: set[str] = set()
    for ws in wspecs:
        names |= set(ws.branches())
    if lifetime:
        names |= set(LIFETIME_BRANCHES) | set(DECAY_BRANCHES)
    for cb in compiled:
        for node in (cb.nodes or {}).values():
            names |= node.branches()
    return sorted(names)


@dataclass
class Acc:
    """Single-variant accumulator (kept for the LLPCombine-JSON comparison and old callers)."""

    n: int = 0
    sumw: float = 0.0
    sumw2: float = 0.0

    def fill(self, w: np.ndarray, mask: np.ndarray) -> None:
        sel = w[mask]
        self.n += int(mask.sum())
        self.sumw += float(sel.sum())
        self.sumw2 += float(np.square(sel).sum())

    def add(self, o: "Acc") -> None:
        self.n += o.n
        self.sumw += o.sumw
        self.sumw2 += o.sumw2


class MultiAcc:
    """Arrays over (selection, HLT mode, variant); see the module docstring."""

    def __init__(self, n_sel: int, n_var: int, topk: int):
        S, H, V, K = n_sel, len(HLT_MODES), n_var, topk
        self.n = np.zeros((S, H), np.int64)
        self.sumw = np.zeros((S, H, V))
        self.sumw2 = np.zeros((S, H, V))
        self.cross = np.zeros((S, H, V, V))
        self.top = np.zeros((S, H, V, K))  # largest |w| (unsorted)
        self.n_anom = np.zeros((S, H, len(ANOM_CLASSES)), np.int64)
        self.sumw_anom = np.zeros((S, H, V))
        self.ov_n = np.zeros((H, S, S), np.int64)
        self.ov_sumw = np.full((H, V, S, S), np.nan)  # weighted overlaps: within one analysis mode only
        self.ov_sumw2 = np.full((H, V, S, S), np.nan)

    @staticmethod
    def _topk(a: np.ndarray, b: np.ndarray, k: int) -> np.ndarray:
        c = np.concatenate([a, b])
        return c if len(c) <= k else np.partition(c, len(c) - k)[len(c) - k:]

    def add(self, o: "MultiAcc") -> None:
        for k in ("n", "sumw", "sumw2", "cross", "n_anom", "sumw_anom", "ov_n"):
            getattr(self, k).__iadd__(getattr(o, k))
        for k in ("ov_sumw", "ov_sumw2"):
            a, b = getattr(self, k), getattr(o, k)
            setattr(self, k, np.where(np.isnan(a), b, np.where(np.isnan(b), a, a + b)))
        K = self.top.shape[-1]
        S, H, V, _ = self.top.shape
        for s in range(S):
            for h in range(H):
                for v in range(V):
                    self.top[s, h, v] = self._topk(self.top[s, h, v], o.top[s, h, v], K)


@dataclass
class TaskResult:
    point: str
    sim: str
    path: str
    entry_start: int
    entry_stop: int
    variants: list = field(default_factory=list)
    acc: MultiAcc | None = None
    io: IOStats = field(default_factory=IOStats)
    seconds_total: float = 0.0
    seconds_select: float = 0.0
    out_of_range: dict[str, int] = field(default_factory=dict)
    n_zero_weight: int = 0
    sum_evtFillWgt: float = 0.0
    trigger_any_true: int = 0
    anomaly_totals: dict[str, int] = field(default_factory=dict)  # over all skimmed events of the range
    ctau_moments: dict[str, float] = field(default_factory=dict)  # n, sum Xa, sum Xb (finite, > 0)
    events: list[dict] = field(default_factory=list)  # anomalous events (capped)
    n_excluded: int = 0
    pid: int = 0
    error: str | None = None

    _compiled: list = field(default_factory=list, repr=False)


def _accumulate(acc: MultiAcc, masks: np.ndarray, h: int, W: dict, mode_idx: dict, anom: np.ndarray,
                compiled: list[CompiledBin]) -> None:
    K = acc.top.shape[-1]
    anysel = masks.any(axis=0)
    sub = np.flatnonzero(anysel)
    if not len(sub):
        return
    Ms = masks[:, sub]
    an = anom[sub]
    an_cls = np.stack([an != 0] + [(an & b) != 0 for b in (1, 2, 4, 8)])  # matches ANOM_CLASSES
    for mode, idx in mode_idx.items():
        Wm = np.ascontiguousarray(W[mode][:, sub])  # (V, k)
        V = Wm.shape[0]
        for s in idx:
            m = Ms[s]
            ns = int(m.sum())
            if not ns:
                continue
            sel = Wm[:, m]  # (V, ns), each row contiguous: same summation as the original w[mask].sum()
            acc.n[s, h] += ns
            for v in range(V):
                acc.sumw[s, h, v] += float(sel[v].sum())
                acc.sumw2[s, h, v] += float(np.square(sel[v]).sum())
                a = np.abs(sel[v])
                acc.top[s, h, v] = MultiAcc._topk(acc.top[s, h, v], a if ns <= K else np.partition(a, ns - K)[ns - K:], K)
            acc.cross[s, h] += sel @ sel.T
            ac = an_cls[:, m]
            acc.n_anom[s, h] += ac.sum(axis=1)
            if ac[0].any():
                acc.sumw_anom[s, h] += sel[:, ac[0]].sum(axis=1)
        Mm = Ms[idx].astype(np.float64)
        for v in range(V):
            blk = np.ix_(idx, idx)
            w1 = (Mm * Wm[v]) @ Mm.T
            w2 = (Mm * np.square(Wm[v])) @ Mm.T
            for arr, val in ((acc.ov_sumw, w1), (acc.ov_sumw2, w2)):
                cur = arr[h, v][blk]
                arr[h, v][blk] = np.where(np.isnan(cur), val, cur + val)
    Mf = Ms.astype(np.float64)
    acc.ov_n[h] += np.rint(Mf @ Mf.T).astype(np.int64)


def run_task(task: dict) -> TaskResult:
    """Worker entry point (top-level so it pickles under the macOS 'spawn' start method)."""
    for var in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    t0 = time.perf_counter()
    variants = list(task.get("variants") or [None])
    if variants[0] is not None:
        raise ValueError("variant 0 must be the no-reweight reference (None)")
    res = TaskResult(task["point"], task["sim"], task["path"], task["entry_start"], task["entry_stop"],
                     variants=variants, pid=os.getpid())
    try:
        wspecs = {m: WeightSpec(**w) for m, w in task["weights"].items()}  # analysis mode -> weights
        compiled = compile_bins(task["bins"], task["is_fastsim"])
        res._compiled = compiled
        use_ctau = task.get("lifetime_branches", True)
        policy = task.get("invalid_ctau_policy", "llpcombine")
        source = task.get("source_ctau_cm")
        branches = needed_branches(compiled, list(wspecs.values()), lifetime=use_ctau)
        trig = [b for b in task.get("trigger_branches", []) if b not in branches]
        acc = MultiAcc(len(compiled), len(variants), int(task.get("topk", 10)))
        mode_idx: dict[str, list[int]] = {}
        for s, cb in enumerate(compiled):
            mode_idx.setdefault(cb.mode, []).append(s)
        tot = {c: 0 for c in ANOM_CLASSES}
        mom = {"n": 0, "sum_xa": 0.0, "sum_xb": 0.0}
        entry = task["entry_start"]
        for arrays in iterate_chunks(
            task["path"], task["tree"], branches + trig, task["entry_start"], task["entry_stop"], task["step_size"], res.io,
            timeout=task.get("timeout", 60.0),
        ):
            ts = time.perf_counter()
            n = len(arrays)
            cache: dict[tuple, np.ndarray] = {}  # identical weight specs (the usual case) computed once
            W = {}
            for m, ws in wspecs.items():
                key = tuple(sorted(task["weights"][m].items()))
                if key not in cache:
                    cache[key] = weight_matrix(arrays, ws, source, variants)
                W[m] = cache[key]
            w0 = next(iter(W.values()))[0]
            res.n_zero_weight += int(np.count_nonzero(w0 == 0))
            res.sum_evtFillWgt += float(np.asarray(arrays["evtFillWgt"], dtype=np.float64).sum())
            tb = task.get("trigger_branches", [])
            if tb:
                anyt = np.zeros(n, dtype=bool)
                for b in tb:
                    anyt |= np.asarray(arrays[b]) != 0
                res.trigger_any_true += int(anyt.sum())
            anom = lifetime_anomalies(arrays) if use_ctau else np.zeros(n, np.int8)
            if use_ctau:
                xa = np.asarray(arrays["Xa_ctau"], dtype=np.float64)
                xb = np.asarray(arrays["Xb_ctau"], dtype=np.float64)
                good = np.isfinite(xa) & np.isfinite(xb) & (xa > 0) & (xb > 0)
                mom["n"] += int(good.sum())
                mom["sum_xa"] += float(xa[good].sum())
                mom["sum_xb"] += float(xb[good].sum())
            for i, c in enumerate(ANOM_CLASSES):
                tot[c] += int(np.count_nonzero(anom != 0 if i == 0 else (anom & (1 << (i - 1))) != 0))
            invalid = (anom & INVALID_FOR_POLICY) != 0
            ctx = expr.EvalContext(arrays=arrays, n_events=n)
            for h, hm in enumerate(HLT_MODES):
                masks = np.zeros((len(compiled), n), dtype=bool)
                for s, cb in enumerate(compiled):
                    if cb.nodes is not None:
                        masks[s] = expr.evaluate(cb.nodes[hm], ctx)
                for s, cb in enumerate(compiled):
                    if cb.kind in ("omission", "omission_unexplained"):
                        masks[s] = masks[cb.ref] & ~masks[list(cb.members)].any(axis=0)
                        if cb.veto is not None:
                            masks[s] &= ~masks[cb.veto]
                if h == 0 and use_ctau:  # record anomalous events with the bins they enter (before any policy)
                    for i in np.flatnonzero(anom != 0)[: max(0, MAX_EVENT_RECORDS - len(res.events))]:
                        res.events.append(dict(
                            entry=int(entry + i), code=int(anom[i]),
                            Xa_ctau=float(arrays["Xa_ctau"][i]), Xb_ctau=float(arrays["Xb_ctau"][i]),
                            flags="".join(str(int(arrays[b][i])) for b in DECAY_BRANCHES),
                            evtrwt=[float(W[compiled[0].mode][v, i]) for v in range(len(variants))],
                            selections_parity=[compiled[s].name for s in np.flatnonzero(masks[:, i])
                                               if compiled[s].kind == "bin"]))
                if invalid.any():
                    hit = masks[:, invalid].any(axis=0)
                    if policy == "fail" and hit.any():
                        raise RuntimeError(f"{int(hit.sum())} selected events with invalid Xa/Xb_ctau "
                                           "(invalid_ctau_policy=fail)")
                    if policy == "exclude":
                        if h == 0:
                            res.n_excluded += int(invalid.sum())
                        masks[:, invalid] = False
                _accumulate(acc, masks, h, W, mode_idx, anom, compiled)
            for k, v in ctx.out_of_range.items():
                res.out_of_range[k] = res.out_of_range.get(k, 0) + v
            entry += n
            res.seconds_select += time.perf_counter() - ts
        res.acc = acc
        res.anomaly_totals = tot
        res.ctau_moments = mom
    except Exception as exc:
        res.error = f"{type(exc).__name__}: {exc}"
    res.seconds_total = time.perf_counter() - t0
    return res


def make_tasks(point_key: str, sim: str, path: str, n_entries: int, task_entries: int, common: dict) -> list[dict]:
    tasks = []
    start = 0
    while start < max(n_entries, 1):
        stop = min(start + task_entries, n_entries)
        tasks.append(dict(common, point=point_key, sim=sim, path=path, entry_start=start, entry_stop=stop))
        start = stop
        if n_entries == 0:
            break
    return tasks


def merge(results: list[TaskResult]) -> dict:
    """(point, sim) -> merged accumulators and diagnostics."""
    merged: dict[tuple[str, str], dict] = {}
    for r in sorted(results, key=lambda r: (r.point, r.sim, r.entry_start)):  # deterministic summation order
        m = merged.setdefault(
            (r.point, r.sim),
            dict(acc=None, compiled=r._compiled, variants=r.variants, path=r.path, io=IOStats(), seconds_total=0.0,
                 seconds_select=0.0, out_of_range={}, n_zero_weight=0, sum_evtFillWgt=0.0, trigger_any_true=0,
                 errors=[], entries=0, anomaly_totals={c: 0 for c in ANOM_CLASSES},
                 ctau_moments={"n": 0, "sum_xa": 0.0, "sum_xb": 0.0}, events=[], n_excluded=0),
        )
        if r.acc is not None:
            if m["acc"] is None:
                m["acc"] = r.acc
                m["compiled"] = r._compiled
            else:
                m["acc"].add(r.acc)
        m["io"].add(r.io)
        m["seconds_total"] += r.seconds_total
        m["seconds_select"] += r.seconds_select
        for k, v in r.out_of_range.items():
            m["out_of_range"][k] = m["out_of_range"].get(k, 0) + v
        m["n_zero_weight"] += r.n_zero_weight
        m["sum_evtFillWgt"] += r.sum_evtFillWgt
        m["trigger_any_true"] += r.trigger_any_true
        m["entries"] += r.entry_stop - r.entry_start
        m["n_excluded"] += r.n_excluded
        for k, v in r.anomaly_totals.items():
            m["anomaly_totals"][k] += v
        for k, v in r.ctau_moments.items():
            m["ctau_moments"][k] += v
        m["events"] += [dict(e, point=r.point, sim=r.sim) for e in r.events]
        if r.error:
            m["errors"].append(f"[{r.entry_start},{r.entry_stop}) {r.error}")
    for m in merged.values():  # backward-compatible single-variant view of the analysis bins
        y = {}
        if m["acc"] is not None:
            for s, cb in enumerate(m["compiled"]):
                if cb.kind == "bin":
                    for h, hm in enumerate(HLT_MODES):
                        y[(cb.mode, cb.name, hm)] = Acc(int(m["acc"].n[s, h]), float(m["acc"].sumw[s, h, 0]),
                                                        float(m["acc"].sumw2[s, h, 0]))
        m["yields"] = y
    return merged
