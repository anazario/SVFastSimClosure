"""Bin selection + weighted-yield aggregation over chunks (uproot backend).

HLT treatment (``hlt`` key of every yield):

* ``parity``    - fit parity with LLPCombine: Trigger_* terms kept for FullSim,
                  replaced by ``true`` for FastSim (sMCType == 2).
* ``symmetric`` - diagnostic: Trigger_* replaced by ``true`` for *both*, so the
                  FullSim/FastSim comparison isolates reconstruction effects.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import numpy as np

from . import expr
from .rootio import IOStats, iterate_chunks
from .weights import WeightSpec, event_weights

HLT_MODES = ("parity", "symmetric")


@dataclass
class CompiledBin:
    mode: str
    name: str
    nodes: dict[str, expr.Node]  # hlt mode -> AST


def compile_bins(bins: list[dict], is_fastsim: bool) -> list[CompiledBin]:
    out = []
    for b in bins:
        full = expr.parse(b["expression"])
        stripped = expr.parse(b["expression_no_hlt"])
        out.append(CompiledBin(b["mode"], b["name"], {"parity": stripped if is_fastsim else full, "symmetric": stripped}))
    return out


def needed_branches(compiled: list[CompiledBin], wspecs: list[WeightSpec]) -> list[str]:
    names: set[str] = set()
    for ws in wspecs:
        names |= set(ws.branches())
    for cb in compiled:
        for node in cb.nodes.values():
            names |= node.branches()
    return sorted(names)


@dataclass
class Acc:
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


@dataclass
class TaskResult:
    point: str
    sim: str
    path: str
    entry_start: int
    entry_stop: int
    yields: dict[tuple[str, str, str], Acc] = field(default_factory=dict)  # (mode, bin, hlt)
    io: IOStats = field(default_factory=IOStats)
    seconds_total: float = 0.0
    seconds_select: float = 0.0
    out_of_range: dict[str, int] = field(default_factory=dict)
    n_zero_weight: int = 0
    sum_evtFillWgt: float = 0.0
    trigger_any_true: int = 0
    pid: int = 0
    error: str | None = None


def run_task(task: dict) -> TaskResult:
    """Worker entry point (top-level so it pickles under the macOS 'spawn' start method)."""
    for var in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    t0 = time.perf_counter()
    res = TaskResult(task["point"], task["sim"], task["path"], task["entry_start"], task["entry_stop"], pid=os.getpid())
    try:
        wspecs = {m: WeightSpec(**w) for m, w in task["weights"].items()}  # analysis mode -> weights
        compiled = compile_bins(task["bins"], task["is_fastsim"])
        branches = needed_branches(compiled, list(wspecs.values()))
        trig = [b for b in task.get("trigger_branches", []) if b not in branches]
        for cb in compiled:
            for h in HLT_MODES:
                res.yields[(cb.mode, cb.name, h)] = Acc()
        for arrays in iterate_chunks(
            task["path"], task["tree"], branches + trig, task["entry_start"], task["entry_stop"], task["step_size"], res.io,
            timeout=task.get("timeout", 60.0),
        ):
            ts = time.perf_counter()
            n = len(arrays)
            wmode = {m: event_weights(arrays, ws) for m, ws in wspecs.items()}
            res.n_zero_weight += int(np.count_nonzero(next(iter(wmode.values())) == 0))
            res.sum_evtFillWgt += float(np.asarray(arrays["evtFillWgt"], dtype=np.float64).sum())
            tb = task.get("trigger_branches", [])
            if tb:
                anyt = np.zeros(n, dtype=bool)
                for b in tb:
                    anyt |= np.asarray(arrays[b]) != 0
                res.trigger_any_true += int(anyt.sum())
            ctx = expr.EvalContext(arrays=arrays, n_events=n)
            for cb in compiled:
                for h, node in cb.nodes.items():
                    res.yields[(cb.mode, cb.name, h)].fill(wmode[cb.mode], expr.evaluate(node, ctx))
            for k, v in ctx.out_of_range.items():
                res.out_of_range[k] = res.out_of_range.get(k, 0) + v
            res.seconds_select += time.perf_counter() - ts
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
    for r in results:
        m = merged.setdefault(
            (r.point, r.sim),
            dict(yields={}, io=IOStats(), seconds_total=0.0, seconds_select=0.0, out_of_range={}, n_zero_weight=0,
                 sum_evtFillWgt=0.0, trigger_any_true=0, errors=[], entries=0),
        )
        for k, acc in r.yields.items():
            m["yields"].setdefault(k, Acc()).add(acc)
        m["io"].add(r.io)
        m["seconds_total"] += r.seconds_total
        m["seconds_select"] += r.seconds_select
        for k, v in r.out_of_range.items():
            m["out_of_range"][k] = m["out_of_range"].get(k, 0) + v
        m["n_zero_weight"] += r.n_zero_weight
        m["sum_evtFillWgt"] += r.sum_evtFillWgt
        m["trigger_any_true"] += r.trigger_any_true
        m["entries"] += r.entry_stop - r.entry_start
        if r.error:
            m["errors"].append(f"[{r.entry_start},{r.entry_stop}) {r.error}")
    return merged
