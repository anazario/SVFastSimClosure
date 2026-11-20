"""Orchestration: study config -> manifests -> yields -> SF tables.  No plotting, no file I/O details."""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import bins as binsmod
from . import expr, lifetime, llpcombine, provenance
from .samples import SignalPoint, build_pairs, inspect_points
from .weights import WeightSpec, anomaly_names, weight_spec_from_config
from .yields import ANOM_CLASSES, HLT_MODES, MAX_EVENT_RECORDS, make_tasks, merge, run_task

PKG_SRC = Path(__file__).resolve().parents[1]


# ----------------------------------------------------------------------------- study


@dataclass
class Analysis:
    mode: str
    cfg: llpcombine.AnalysisConfig
    fit: llpcombine.FitConfig
    luminosity: float
    weights: WeightSpec


@dataclass
class Study:
    path: Path
    raw: dict
    llpcombine_dir: Path
    llpstandardplots_dir: Path
    analyses: list[Analysis]
    lumi_map: dict
    regimes: "lifetime.Regimes | None" = None

    @property
    def io(self) -> dict:
        return self.raw["io"]

    @property
    def samples(self) -> dict:
        return self.raw["samples"]


def load_study(path: str | Path) -> Study:
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    lc = Path(raw["llpcombine_dir"]).expanduser()
    lumi_map = llpcombine.load_lumi_map(lc)
    analyses = []
    for a in raw["analyses"]:
        cfg = llpcombine.load_analysis_config(lc / a["analysis_config"])
        fit = llpcombine.load_fit_config(lc / a["fit_config"])
        lumi = llpcombine.resolved_luminosity(cfg, lumi_map)
        ws = weight_spec_from_config(cfg.raw, lumi, raw["weights"]["decay_defaults"])
        analyses.append(Analysis(a["mode"], cfg, fit, lumi, ws))
    regimes = lifetime.regimes_from_config(raw["signal_regimes"]) if raw.get("signal_regimes") else None
    return Study(path, raw, lc, Path(raw["llpstandardplots_dir"]).expanduser(), analyses, lumi_map, regimes)


def resolve_all_bins(study: Study) -> tuple[list[binsmod.ResolvedBin], list[str]]:
    out, problems = [], []
    for a in study.analyses:
        rb = binsmod.resolve_bins(a.mode, a.cfg, a.fit)
        problems += [f"{a.mode}: {p}" for p in binsmod.check_structure(rb)]
        out += rb
    return out, problems


def discover_points(study: Study, inspect: bool) -> list[SignalPoint]:
    master = llpcombine.parse_sample_tool(study.llpcombine_dir)
    s = study.samples
    points = build_pairs(master, s["fullsim_key"], s["fastsim_key"], s.get("expected_prefix"))
    if inspect:
        inspect_points(points, s["tree"], s["config_tree"], study.io["metadata_threads"], study.io["timeout"])
    return points


def points_frame(points: list[SignalPoint], regimes: "lifetime.Regimes | None" = None) -> pd.DataFrame:
    rows = []
    for p in points:
        d = dict(key=p.key, mass_key=p.key.rsplit("_", 1)[0], mgl=p.mgl, mn2=p.mn2, mn1=p.mn1, dm=p.dm,
                 ctau_cm=p.ctau_cm, ctau_cm_num=float(p.ctau_cm.replace("p", ".")),
                 short_label=f"({p.mgl}, {p.mn2}, {p.mn1}), {p.ctau_cm} cm",
                 mass_label=f"({p.mgl}, {p.mn2}, {p.mn1})", usable=p.usable, paired=p.paired)
        if p.paired:
            sl = lifetime.source_lifetime(p)
            d.update(source_ctau_fullsim_cm=sl.fullsim_cm, source_ctau_fastsim_cm=sl.fastsim_cm,
                     source_ctau_ok=sl.ok, source_ctau_problem=sl.problem)
        if regimes is not None:
            reg, unamb, alts = regimes.classify(p.mgl, p.mn2, p.mn1)
            d.update(regime=reg, regime_unambiguous=unamb, regime_alternatives=json.dumps(alts))
        rows.append(d)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- run directory


def new_run_dir(results: Path, tag: str) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    d = results / f"{stamp}_{tag}"
    d.mkdir(parents=True, exist_ok=False)
    return d


def write_provenance(study: Study, run: Path, command: list[str], extra: dict | None = None) -> dict:
    watch = ["src/SampleTool.cpp", "src/BuildFitInput.cpp", "src/ConfigParser.cpp", "src/main.cpp",
             "include/BuildFitTools.h", "include/ConfigParser.h"]
    for a in study.raw["analyses"]:
        watch += [a["analysis_config"], a["fit_config"]]
    prov = {
        "created": provenance.now(),
        "command": command,
        "study_config": str(study.path),
        "repos": {
            "SVFastSimClosure": provenance.git_state(PKG_SRC.parent, provenance.tracked_sources(PKG_SRC.parent),
                                                     diff_hash=True),
            "LLPCombine": provenance.git_state(study.llpcombine_dir, watch),
            "LLPStandardPlots": provenance.git_state(
                study.llpstandardplots_dir, ["src/style.py", "src/fit_plotter.py", "src/unrolled.py"]
            ),
        },
        "environment": provenance.environment(),
        "analyses": [
            dict(mode=a.mode, name=a.cfg.name, luminosity_resolved=a.luminosity, luminosity_yaml=a.cfg.yaml_luminosity,
                 weights=a.weights.__dict__, lumi_cpp=a.weights.lumi_cpp())
            for a in study.analyses
        ],
        "notes": [
            "FastSim inputs are Fast1_SxyScaleNominal skims: nominal S_xy correction already applied in skimming; not reapplied.",
            "hlt=parity: Trigger_* -> true for sMCType==2 (FastSim) only, as BuildFitInput::FilterRegions.",
            "hlt=symmetric: Trigger_* -> true for FullSim and FastSim (diagnostic).",
        ],
    }
    if extra:
        prov.update(extra)
    (run / "provenance.json").write_text(json.dumps(prov, indent=1, default=str))
    shutil.copy(study.path, run / "study_config.yaml")
    return prov


def write_manifests(run: Path, bins: list[binsmod.ResolvedBin], problems: list[str], points: list[SignalPoint]) -> None:
    m = run / "manifest"
    m.mkdir(exist_ok=True)
    (m / "bins.json").write_text(json.dumps({"problems": problems, "bins": [b.to_dict() for b in bins]}, indent=1))
    (m / "samples.json").write_text(json.dumps([p.to_dict() for p in points], indent=1, default=str))
    (m / "bins_resolved.txt").write_text(format_bins(bins))
    points_frame(points).to_csv(m / "points.csv", index=False)
    bin_frame(bins).to_csv(m / "bins.csv", index=False)


def bin_frame(bins: list[binsmod.ResolvedBin]) -> pd.DataFrame:
    return pd.DataFrame([{k: v for k, v in b.to_dict().items() if k not in ("kin_intervals", "extra")} for b in bins])


def format_bins(bins: list[binsmod.ResolvedBin], show_stripped: bool = True) -> str:
    lines = []
    for b in bins:
        lines.append(f"[{b.mode}] {b.name}   S_xy={b.sxy}  role={b.role}  group={b.group}  partner={b.partner}")
        lines.append(f"    kinematics : {b.kin_label}")
        lines.append(f"    FullSim    : {b.expression}")
        if show_stripped:
            lines.append(f"    FastSim/HLT-dropped : {b.expression_no_hlt}")
        lines.append("")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- yields


def _bin_payload(bins: list[binsmod.ResolvedBin], refs: list[binsmod.ReferenceSelection] | None = None) -> list[dict]:
    out = [dict(mode=b.mode, name=b.name, expression=b.expression, expression_no_hlt=b.expression_no_hlt, kind="bin")
           for b in bins]
    for r in refs or []:
        out.append(dict(mode=r.mode, name=r.name, expression=r.expression, expression_no_hlt=r.expression_no_hlt,
                         kind=r.kind, members=r.members, gap_expression=r.gap_expression))
    return out


def trigger_branches(bins: list[binsmod.ResolvedBin]) -> list[str]:
    return sorted({br for b in bins for br in b.branches if br.startswith("Trigger_")})


@dataclass
class YieldRun:
    """Tidy outputs of one pass over the files (see yields.py for what is accumulated)."""

    yields: pd.DataFrame  # one row per (selection, point, sim, hlt, variant)
    cross: pd.DataFrame  # sum w_i w_j between variants (same point/sim/selection/hlt)
    overlaps: pd.DataFrame  # event-level overlaps of selection pairs (n_ab > 0 only)
    anomalies: pd.DataFrame  # EXAMPLES of events with anomalous inputs (<= MAX_EVENT_RECORDS per task)
    files: pd.DataFrame  # per (point, sim): lifetime-input summary (complete counts)
    perf: dict
    merged: dict
    failures: pd.DataFrame = None  # events that made a task fail (non-finite weight / invalid input under 'fail')


def run_yields(study: Study, points: list[SignalPoint], bins: list[binsmod.ResolvedBin], workers: int,
               step_size: int, task_entries: int, sims=("fullsim", "fastsim"), log=print,
               refs: list[binsmod.ReferenceSelection] | None = None, targets_of=None, policy: str = "llpcombine",
               topk: int = 10) -> YieldRun:
    """Process every (point, sim) file once.

    ``targets_of(point) -> (source_cm, [target_cm, ...])``; ``None`` (default) = no lifetime
    variants (the original study).  Variant 0 is always the no-reweight reference.
    """
    payload = _bin_payload(bins, refs)
    weights = {a.mode: a.weights.__dict__ for a in study.analyses}
    trig = trigger_branches(bins)
    tasks = []
    for p in points:
        source, tg = (None, [])
        if targets_of is not None:
            source, tg = targets_of(p)
        variants = [None] + list(tg)
        for sim in sims:
            info = getattr(p, f"{sim}_info")
            if not info or not info.get("ok"):
                continue
            avail = set(info["branches"])
            is_fast = info["config"].get("sMCType") == 2
            common = dict(bins=payload, weights=weights, is_fastsim=is_fast, tree=study.samples["tree"],
                          step_size=step_size, timeout=study.io["timeout"],
                          trigger_branches=[t for t in trig if t in avail], variants=variants, source_ctau_cm=source,
                          invalid_ctau_policy=policy, topk=topk,
                          lifetime_branches=all(b in avail for b in ("Xa_ctau", "Xb_ctau")))
            tasks += make_tasks(p.key, sim, getattr(p, sim), info["tree_entries"], task_entries, common)
    log(f"{len(tasks)} tasks over {len({(t['point'], t['sim']) for t in tasks})} files, {workers} worker processes")
    t0 = time.perf_counter()
    results = []
    if workers <= 1:
        for i, t in enumerate(tasks):
            results.append(run_task(t))
            _progress(log, i + 1, len(tasks), results[-1], t0)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(run_task, t) for t in tasks]
            for i, f in enumerate(as_completed(futs)):
                results.append(f.result())
                _progress(log, i + 1, len(tasks), results[-1], t0)
    wall = time.perf_counter() - t0
    merged = merge(results)
    rows, crows, orows, perf_files, frows, events = [], [], [], [], [], []
    for (pk, sim), m in merged.items():
        acc, comp, variants = m["acc"], m["compiled"], m["variants"]
        src = None
        if targets_of is not None:
            src = targets_of(next(p for p in points if p.key == pk))[0]
        vkeys = [lifetime.variant_key(t) for t in variants]
        tgt = [src if t is None else t for t in variants]
        if acc is not None:
            K = acc.top.shape[-1]
            for s, cb in enumerate(comp):
                for h, hm in enumerate(HLT_MODES):
                    for v in range(len(variants)):
                        top = np.sort(acc.top[s, h, v])[::-1]
                        s2 = acc.sumw2[s, h, v]
                        fr = (lambda k: float(np.square(top[:k]).sum() / s2) if s2 > 0 else np.nan)
                        rows.append(dict(
                            mode=cb.mode, bin=cb.name, kind=cb.kind, point=pk, sim=sim, hlt=hm, variant=vkeys[v],
                            target_ctau_cm=tgt[v], n_raw=int(acc.n[s, h]), sumw=float(acc.sumw[s, h, v]), sumw2=float(s2),
                            w_max=float(top[0]) if len(top) else 0.0,
                            frac_wmax_sumw=float(top[0] / acc.sumw[s, h, v]) if acc.sumw[s, h, v] > 0 else np.nan,
                            top1_sumw2_frac=fr(1), top3_sumw2_frac=fr(3), top5_sumw2_frac=fr(min(5, K)),
                            top_w=top.copy(),
                            **{f"n_anom_{c}": int(acc.n_anom[s, h, i]) for i, c in enumerate(ANOM_CLASSES)},
                            sumw_anom=float(acc.sumw_anom[s, h, v])))
                        for u in range(len(variants)):
                            if targets_of is not None:
                                crows.append(dict(mode=cb.mode, bin=cb.name, kind=cb.kind, point=pk, sim=sim, hlt=hm,
                                                  variant_i=vkeys[v], variant_j=vkeys[u], target_i_cm=tgt[v],
                                                  target_j_cm=tgt[u], sumw_ij=float(acc.cross[s, h, v, u])))
            S = len(comp)
            for h, hm in enumerate(HLT_MODES):
                nz = np.argwhere(np.triu(acc.ov_n[h], 1) > 0)
                for a, b in nz:
                    base = dict(point=pk, sim=sim, hlt=hm, mode_a=comp[a].mode, sel_a=comp[a].name, kind_a=comp[a].kind,
                                mode_b=comp[b].mode, sel_b=comp[b].name, kind_b=comp[b].kind, n_a=int(acc.ov_n[h, a, a]),
                                n_b=int(acc.ov_n[h, b, b]), n_ab=int(acc.ov_n[h, a, b]))
                    for v in range(len(variants)):
                        orows.append(dict(base, variant=vkeys[v], target_ctau_cm=tgt[v],
                                          sumw_ab=float(acc.ov_sumw[h, v, a, b]), sumw2_ab=float(acc.ov_sumw2[h, v, a, b])))
        mom = m["ctau_moments"]
        frows.append(dict(point=pk, sim=sim, path=m["path"], entries=m["entries"], source_ctau_cm=src,
                          n_ctau_valid=mom["n"], mean_Xa_ctau=mom["sum_xa"] / mom["n"] if mom["n"] else np.nan,
                          mean_Xb_ctau=mom["sum_xb"] / mom["n"] if mom["n"] else np.nan, n_excluded=m["n_excluded"],
                          **{f"n_{k}": v for k, v in m["anomaly_totals"].items()},
                          **{f"n_in_bins_{k}": v for k, v in m["anomaly_in_bins"].items()},
                          n_events_listed=len(m["events"]), n_tasks=m["n_tasks"],
                          event_list_cap_per_task=MAX_EVENT_RECORDS,
                          event_list_truncated=bool(m["n_tasks_event_list_full"])))
        for e in m["events"]:
            e = dict(e)
            ev = e.pop("evtrwt")
            e.pop("targets_cm", None)
            e.pop("nonfinite_targets", None)
            e.pop("selections_all", None)
            e["anomaly"] = anomaly_names(e["code"])
            e["selections_parity"] = ";".join(e["selections_parity"])
            for v, w in enumerate(ev):
                e[f"evtrwt_{vkeys[v]}"] = w
            e["path"] = m["path"]
            events.append(e)
        io = m["io"]
        perf_files.append(dict(point=pk, sim=sim, entries=m["entries"], bytes=io.bytes_requested, requests=io.requests,
                               read_s=io.seconds_read, select_s=m["seconds_select"], task_s=m["seconds_total"],
                               MBps_read=(io.bytes_requested / 1e6 / io.seconds_read) if io.seconds_read else None,
                               n_zero_weight=m["n_zero_weight"], sum_evtFillWgt=m["sum_evtFillWgt"],
                               trigger_any_true=m["trigger_any_true"], out_of_range=m["out_of_range"],
                               errors=m["errors"]))
    total_bytes = sum(f["bytes"] for f in perf_files)
    perf = dict(workers=workers, step_size=step_size, task_entries=task_entries, n_tasks=len(tasks), wall_s=wall,
                total_bytes=total_bytes, aggregate_MBps=total_bytes / 1e6 / wall if wall else None,
                entries=sum(f["entries"] for f in perf_files), files=perf_files)
    y = pd.DataFrame(rows)
    if len(y):
        y["source_ctau_cm"] = y.point.map({p.key: (targets_of(p)[0] if targets_of else _key_ctau(p)) for p in points})
        y.loc[y.variant == "noreweight", "target_ctau_cm"] = y.loc[y.variant == "noreweight", "source_ctau_cm"]
    fails = []
    for m in merged.values():
        for e in m["failure_events"]:
            e = dict(e)
            e["anomaly"] = anomaly_names(e["code"])
            e["source_ctau_cm"] = next((targets_of(p)[0] for p in points if p.key == e["point"]), None) if targets_of else None
            for k in ("targets_cm", "evtrwt", "nonfinite_targets", "selections_parity", "selections_all"):
                e[k] = json.dumps(e[k])
            fails.append(e)
    return YieldRun(y, pd.DataFrame(crows), pd.DataFrame(orows), pd.DataFrame(events), pd.DataFrame(frows), perf, merged,
                    pd.DataFrame(fails))


def _key_ctau(p: SignalPoint) -> float:
    return lifetime.ctau_cm_from_key_token(p.ctau_cm)


LEGACY_YIELD_COLS = ["mode", "bin", "point", "sim", "hlt", "n_raw", "sumw", "sumw2"]


def legacy_yields(y: pd.DataFrame) -> pd.DataFrame:
    """The original yields.csv rows (analysis bins, no lifetime factor), original column order first."""
    d = y[(y.kind == "bin") & (y.variant == "noreweight")]
    extra = [c for c in d.columns if c not in LEGACY_YIELD_COLS]
    return d[LEGACY_YIELD_COLS + extra].reset_index(drop=True)


def annotate(df: pd.DataFrame, points: list[SignalPoint], prov_cols: dict) -> pd.DataFrame:
    """Append hypothesis metadata, sample paths and code revision to any per-point table."""
    if df is None or not len(df) or "point" not in df.columns:
        return df
    meta = {p.key: dict(mass_key=p.key.rsplit("_", 1)[0], mgl=p.mgl, mn2=p.mn2, mn1=p.mn1, dm=p.dm,
                        fullsim_path=p.fullsim, fastsim_path=p.fastsim) for p in points}
    df = df.copy()
    for col in ("mass_key", "mgl", "mn2", "mn1", "dm"):
        if col not in df.columns:
            df[col] = df.point.map(lambda k: meta.get(k, {}).get(col))
    if "sim" in df.columns and "sample_path" not in df.columns:
        df["sample_path"] = [meta.get(k, {}).get(f"{s}_path") for k, s in zip(df.point, df.sim)]
    elif "sim" not in df.columns:
        for s in ("fullsim", "fastsim"):
            if f"{s}_path" not in df.columns:
                df[f"{s}_path"] = df.point.map(lambda k: meta.get(k, {}).get(f"{s}_path"))
    for k, v in prov_cols.items():
        df[k] = v
    return df


def _progress(log, i, n, r, t0):
    if i == n or i % max(1, n // 20) == 0 or r.error:
        el = time.perf_counter() - t0
        msg = f"  [{i}/{n}] {el:7.1f}s  last: {r.point} {r.sim} [{r.entry_start},{r.entry_stop}) {r.seconds_total:.1f}s"
        log(msg + (f"  ERROR {r.error}" if r.error else ""))


# ----------------------------------------------------------------------------- validation helpers


def branch_check(info: dict, bins: list[binsmod.ResolvedBin], weights: list[WeightSpec], is_fastsim: bool) -> dict:
    avail = set(info["branches"])
    need: set[str] = set()
    for b in bins:
        need |= set(b.branches_no_hlt if is_fastsim else b.branches)
    for w in weights:
        need |= set(w.branches())
    missing_parity = sorted(need - avail)
    trig = trigger_branches(bins)
    return dict(n_needed=len(need), missing=missing_parity,
                trigger_branches_present=[t for t in trig if t in avail],
                trigger_branches_absent=[t for t in trig if t not in avail])


def run_rdf_backend(study: Study, path: str, bins: list[binsmod.ResolvedBin], workdir: Path, threads: int = 0,
                    timeout_s: int = 3600, lifetime_spec: dict | None = None, dump_weights: bool = False) -> dict:
    """Run the RDataFrame replica of LLPCombine in a separate Python process."""
    weights = {a.mode: a.weights for a in study.analyses}
    if len({(w.lumi_cpp(), w.sample_z, w.sample_g, w.target_z, w.target_g) for w in weights.values()}) != 1:
        raise RuntimeError("RDF backend assumes the same weight spec for all modes")
    w = next(iter(weights.values()))
    spec = dict(path=path, tree=study.samples["tree"], llpcombine_include=str(study.llpcombine_dir / "include"),
                threads=threads, bins=_bin_payload(bins),
                weights=dict(lumi_cpp=w.lumi_cpp(), sample_z=w.sample_z, sample_g=w.sample_g, target_z=w.target_z,
                             target_g=w.target_g), lifetime=lifetime_spec, dump_weights=dump_weights)
    if dump_weights and threads:
        raise ValueError("dump_weights needs threads=0 (event order)")
    workdir.mkdir(parents=True, exist_ok=True)
    tag = Path(path).name.replace(".root", "")
    sp, op = workdir / f"rdf_spec_{tag}.json", workdir / f"rdf_out_{tag}.json"
    sp.write_text(json.dumps(spec, indent=1))
    env = dict(os.environ, PYTHONPATH=str(PKG_SRC) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    t0 = time.perf_counter()
    proc = subprocess.run([sys.executable, "-m", "svclosure.rdf_backend", str(sp), str(op)], env=env,
                          capture_output=True, text=True, timeout=timeout_s)
    if proc.returncode != 0:
        raise RuntimeError(f"RDF backend failed ({proc.returncode}):\n{proc.stderr[-4000:]}")
    out = json.loads(op.read_text())
    out["wall_s"] = time.perf_counter() - t0
    return out


def compare_backends(uproot_rows: pd.DataFrame, rdf: dict, rel_tol: float = 1e-9) -> pd.DataFrame:
    """Counts must be identical; sums agree to ``rel_tol`` (summation order differs: RDF sums in its own order)."""
    r = pd.DataFrame(rdf["yields"]).rename(columns={"n_raw": "rdf_n", "sumw": "rdf_sumw", "sumw2": "rdf_sumw2"})
    u = uproot_rows.rename(columns={"n_raw": "up_n", "sumw": "up_sumw", "sumw2": "up_sumw2"})
    on = ["mode", "bin", "hlt"] + (["variant"] if "variant" in u.columns else [])
    if "variant" not in u.columns:
        r = r[r.variant == "noreweight"]
    t = u.merge(r, on=on, how="outer")
    t["n_match"] = t.up_n == t.rdf_n

    def close(a, b):
        return np.isclose(a, b, rtol=rel_tol, atol=1e-300)

    t["sumw_match"] = close(t.up_sumw, t.rdf_sumw)
    t["sumw2_match"] = close(t.up_sumw2, t.rdf_sumw2)
    with np.errstate(divide="ignore", invalid="ignore"):
        t["sumw_reldiff"] = (t.up_sumw - t.rdf_sumw) / t.rdf_sumw.where(t.rdf_sumw != 0)
        t["sumw2_reldiff"] = (t.up_sumw2 - t.rdf_sumw2) / t.rdf_sumw2.where(t.rdf_sumw2 != 0)
    t["ok"] = t.n_match & t.sumw_match & t.sumw2_match
    return t


def compare_weight_dump(npz_path: str, path: str, tree: str, spec_weights: WeightSpec, source_cm: float,
                        targets: list[float], timeout: float) -> pd.DataFrame:
    """Element-wise: Python lifetime_weight / evtrwt vs the C++ values Taken from RDataFrame."""
    from .rootio import open_file
    from .weights import lifetime_weight, weight_matrix

    d = np.load(npz_path)
    with open_file(path, timeout) as f:
        arr = f[tree].arrays(["evtFillWgt", "Xa_ctau", "Xb_ctau", "Evt_isZZ", "Evt_isGZ", "Evt_isGG"], library="np")
    if not np.array_equal(d["entry"], np.arange(len(arr["evtFillWgt"]))):
        raise RuntimeError("RDF dump is not in entry order")
    W = weight_matrix(arr, spec_weights, source_cm, [None] + list(targets))
    rows = []

    def ulp32(a, b):
        return np.abs(a.astype(np.float32).view(np.int32).astype(np.int64) - b.astype(np.float32).view(np.int32).astype(np.int64))

    for i, t in enumerate(targets):
        vk = lifetime.variant_key(t)
        for col, ours in (("xawt", lifetime_weight(arr["Xa_ctau"], source_cm, t)),
                          ("xbwt", lifetime_weight(arr["Xb_ctau"], source_cm, t))):
            ref = d[f"{col}_{vk}"]
            u = ulp32(ours, ref)
            rows.append(dict(target_ctau_cm=t, column=col, n=len(ref), n_bitwise_equal=int(np.sum(ours == ref)),
                             max_ulp_float32=int(u.max()),
                             max_rel_diff=float(np.max(np.where(ref != 0, np.abs(ours / np.where(ref != 0, ref, 1) - 1),
                                                                np.abs(ours)))), n_zero_underflow=int(np.sum(ref == 0))))
        ref = d[f"evtrwt_{vk}"]
        ours = W[i + 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.where(ref != 0, np.abs(ours / ref - 1), np.abs(ours))
        rows.append(dict(target_ctau_cm=t, column="evtrwt", n=len(ref), n_bitwise_equal=int(np.sum(ours == ref)),
                         max_ulp_float32=None, max_rel_diff=float(rel.max())))
    ref = d["evtrwt_noreweight"]
    rows.append(dict(target_ctau_cm=None, column="evtrwt_noreweight", n=len(ref), n_bitwise_equal=int(np.sum(W[0] == ref)),
                     max_ulp_float32=None, max_rel_diff=float(np.max(np.abs(W[0] - ref) / np.where(ref != 0, ref, 1)))))
    return pd.DataFrame(rows)


def compare_llpcombine_json(study: Study, json_path: Path, analysis_config: Path, fullsim_path: str,
                            point_key: str, workers: int, log=print) -> pd.DataFrame:
    """Numerically compare to an existing LLPCombine JSON made with ``analysis_config`` (inclusive / no split).

    The engine runs that config's SV bins on one FullSim file at unit luminosity; the
    JSON prediction for a merged signal process is then  n*K,  S*sum(L_k),
    sqrt(Q*sum(L_k^2)) over the K year-suffixed signal keys that contain the file
    (Process::Add sums yields and adds errors in quadrature).
    """
    cfg = llpcombine.load_analysis_config(analysis_config)
    master = llpcombine.parse_sample_tool(study.llpcombine_dir)
    data_years = {llpcombine.data_year_token(d, study.lumi_map) for d in cfg.data} - {""}
    lumis = []
    for key in cfg.signals:
        if fullsim_path not in master.get(key, []):
            continue
        yy = llpcombine.signal_year_token(key, study.lumi_map)
        if yy and yy in cfg.sample_lumis:
            lumis.append(cfg.sample_lumis[yy])
        elif yy:
            lumis.append(study.lumi_map[yy][1])
        else:
            lumis.append(sum(study.lumi_map[y][1] for y in data_years))
    if not lumis:
        raise RuntimeError(f"{fullsim_path} is not in any signal key of {analysis_config}")
    jd = json.loads(Path(json_path).read_text())
    payload = []
    for name in cfg.bins:
        if name not in jd:
            continue
        full = cfg.combined_cuts(name)
        if binsmod.required_sv_flags(expr.parse(full)):
            payload.append(dict(mode="cmp", name=name, expression=full, expression_no_hlt=expr.strip_triggers(full)))
    info_task = dict(bins=payload, weights={"cmp": WeightSpec(luminosity=1.0).__dict__}, is_fastsim=False,
                     tree=study.samples["tree"], step_size=study.io["step_size"], timeout=study.io["timeout"],
                     trigger_branches=[])
    from .rootio import inspect_file

    info = inspect_file(fullsim_path, study.samples["tree"], study.samples["config_tree"], study.io["timeout"])
    tasks = make_tasks(point_key, "fullsim", fullsim_path, info.tree_entries, study.io["task_entries"], info_task)
    with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
        res = list(pool.map(run_task, tasks))
    merged = merge(res)[(point_key, "fullsim")]
    K, L1, L2 = len(lumis), sum(lumis), math.sqrt(sum(l * l for l in lumis))
    rows = []
    for p in payload:
        acc = merged["yields"][("cmp", p["name"], "parity")]
        ref = jd[p["name"]].get(point_key)
        rows.append(dict(bin=p["name"], pred_n=acc.n * K, pred_w=acc.sumw * L1, pred_err=math.sqrt(acc.sumw2) * L2,
                         json_n=None if ref is None else ref[0], json_w=None if ref is None else ref[1],
                         json_err=None if ref is None else ref[2]))
    t = pd.DataFrame(rows)
    t["n_match"] = t.pred_n == t.json_n
    t["w_reldiff"] = (t.pred_w - t.json_w) / t.json_w.where(t.json_w != 0)
    t["err_reldiff"] = (t.pred_err - t.json_err) / t.json_err.where(t.json_err != 0)
    t.attrs.update(K=K, lumis=lumis)
    log(f"JSON comparison: {K} signal keys contain the file, lumis={lumis}")
    return t
