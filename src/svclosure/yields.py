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
from .weights import ANOM_CLASSES as _ANOM
from .weights import DECAY_BRANCHES, LIFETIME_BRANCHES, POLICY_ACTIONS, WeightSpec, lifetime_anomalies, weight_matrix

HLT_MODES = ("parity", "symmetric")
ANOM_CLASSES = tuple(name for name, _ in _ANOM)  # column order of the n_anom_* counters
ANOM_MASKS = tuple(bits for _, bits in _ANOM)
MAX_EVENT_RECORDS = 2000  # per task: cap of the anomalous-event EXAMPLES (counts are always complete)


class NonFiniteWeightError(RuntimeError):
    """A selected event has a non-finite weight in some variant: yields cannot be computed reliably."""


class InvalidInputError(RuntimeError):
    """invalid_ctau_policy=fail and a selected event has invalid weight inputs."""


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
    events: list[dict] = field(default_factory=list)  # anomalous events (examples, capped per task)
    anomaly_in_bins: dict[str, int] = field(default_factory=dict)  # anomalous events entering >= 1 analysis bin
    failure_events: list[dict] = field(default_factory=list)  # events that made the task fail (capped)
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
    an_cls = np.stack([(an & b) != 0 for b in ANOM_MASKS])  # matches ANOM_CLASSES
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
        if policy not in POLICY_ACTIONS:
            raise ValueError(f"unknown invalid_ctau_policy {policy!r}")
        act = POLICY_ACTIONS[policy]
        source = task.get("source_ctau_cm")
        if any(v is not None for v in variants) and not use_ctau:
            raise InvalidInputError(f"lifetime targets requested but {LIFETIME_BRANCHES} are missing from {task['path']}")
        branches = needed_branches(compiled, list(wspecs.values()), lifetime=use_ctau)
        trig = [b for b in task.get("trigger_branches", []) if b not in branches]
        acc = MultiAcc(len(compiled), len(variants), int(task.get("topk", 10)))
        mode_idx: dict[str, list[int]] = {}
        for s, cb in enumerate(compiled):
            mode_idx.setdefault(cb.mode, []).append(s)
        tot = {c: 0 for c in ANOM_CLASSES}
        in_bins = {c: 0 for c in ANOM_CLASSES}
        bin_rows = np.array([s for s, cb in enumerate(compiled) if cb.kind == "bin"], dtype=int)
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
            for c, bits in zip(ANOM_CLASSES, ANOM_MASKS):
                tot[c] += int(np.count_nonzero(anom & bits))
            excl = (anom & act["exclude"]) != 0
            failm = (anom & act["fail"]) != 0
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
                    inb = masks[bin_rows].any(axis=0) if len(bin_rows) else np.zeros(n, bool)
                    for c, bits in zip(ANOM_CLASSES, ANOM_MASKS):
                        in_bins[c] += int(np.count_nonzero(inb & ((anom & bits) != 0)))
                    for i in np.flatnonzero(anom != 0)[: max(0, MAX_EVENT_RECORDS - len(res.events))]:
                        res.events.append(_event_record(arrays, i, entry, anom, W, compiled, masks, variants))
                if failm.any():
                    hit = failm & masks.any(axis=0)
                    if hit.any():
                        for i in np.flatnonzero(hit)[:MAX_EVENT_RECORDS]:
                            res.failure_events.append(dict(_event_record(arrays, i, entry, anom, W, compiled, masks,
                                                                         variants), reason="invalid_input", hlt=hm))
                        raise InvalidInputError(f"{int(hit.sum())} selected events with invalid weight inputs "
                                                f"(invalid_ctau_policy=fail), first entry {entry + int(np.flatnonzero(hit)[0])}")
                if excl.any():
                    if h == 0:
                        res.n_excluded += int(excl.sum())
                    masks[:, excl] = False
                bad = np.zeros(n, dtype=bool)
                for mode, idx in mode_idx.items():
                    bad |= masks[idx].any(axis=0) & ~np.isfinite(W[mode]).all(axis=0)
                if bad.any():
                    for i in np.flatnonzero(bad)[:MAX_EVENT_RECORDS]:
                        res.failure_events.append(dict(_event_record(arrays, i, entry, anom, W, compiled, masks, variants),
                                                       reason="nonfinite_weight", hlt=hm))
                    raise NonFiniteWeightError(
                        f"{int(bad.sum())} selected events with non-finite weights (hlt={hm}, policy={policy}), "
                        f"first entry {entry + int(np.flatnonzero(bad)[0])}; see failure_events")
                _accumulate(acc, masks, h, W, mode_idx, anom, compiled)
            for k, v in ctx.out_of_range.items():
                res.out_of_range[k] = res.out_of_range.get(k, 0) + v
            entry += n
            res.seconds_select += time.perf_counter() - ts
        res.acc = acc
        res.anomaly_totals = tot
        res.anomaly_in_bins = in_bins
        res.ctau_moments = mom
    except Exception as exc:
        res.error = f"{type(exc).__name__}: {exc}"
    res.seconds_total = time.perf_counter() - t0
    return res


def _event_record(arrays, i, entry, anom, W, compiled, masks, variants) -> dict:
    """Everything needed to understand one event's weight: inputs, the weights per variant, its selections."""
    def g(b):
        return float(arrays[b][i]) if b in arrays.fields else None

    mode0 = compiled[0].mode
    return dict(entry=int(entry + i), code=int(anom[i]), evtFillWgt=g("evtFillWgt"), Xa_ctau=g("Xa_ctau"),
                Xb_ctau=g("Xb_ctau"),
                flags="".join(str(int(arrays[b][i])) if b in arrays.fields else "?" for b in DECAY_BRANCHES),
                targets_cm=[None if v is None else float(v) for v in variants],
                evtrwt=[float(W[mode0][v, i]) for v in range(len(variants))],
                nonfinite_targets=[None if variants[v] is None else float(variants[v]) for v in range(len(variants))
                                   if not np.isfinite(W[mode0][v, i])],
                selections_parity=[compiled[s].name for s in np.flatnonzero(masks[:, i]) if compiled[s].kind == "bin"],
                selections_all=[f"{compiled[s].mode}:{compiled[s].name}" for s in np.flatnonzero(masks[:, i])])


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
                 ctau_moments={"n": 0, "sum_xa": 0.0, "sum_xb": 0.0}, events=[], n_excluded=0,
                 anomaly_in_bins={c: 0 for c in ANOM_CLASSES}, failure_events=[], n_tasks=0,
                 n_tasks_event_list_full=0),
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
        for k, v in r.anomaly_in_bins.items():
            m["anomaly_in_bins"][k] += v
        m["n_tasks"] += 1
        m["n_tasks_event_list_full"] += int(len(r.events) >= MAX_EVENT_RECORDS)
        m["failure_events"] += [dict(e, point=r.point, sim=r.sim, path=r.path) for e in r.failure_events]
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
