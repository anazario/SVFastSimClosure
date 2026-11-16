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
from . import expr, llpcombine, provenance
from .samples import SignalPoint, build_pairs, inspect_points
from .weights import WeightSpec, weight_spec_from_config
from .yields import HLT_MODES, make_tasks, merge, run_task

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
    return Study(path, raw, lc, Path(raw["llpstandardplots_dir"]).expanduser(), analyses, lumi_map)


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


def points_frame(points: list[SignalPoint]) -> pd.DataFrame:
    rows = []
    for p in points:
        rows.append(dict(key=p.key, mgl=p.mgl, mn2=p.mn2, mn1=p.mn1, dm=p.dm, ctau_cm=p.ctau_cm,
                         ctau_cm_num=float(p.ctau_cm.replace("p", ".")),
                         short_label=f"({p.mgl}, {p.mn2}, {p.mn1}), {p.ctau_cm} cm",
                         usable=p.usable, paired=p.paired))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- run directory


def new_run_dir(results: Path, tag: str) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    d = results / f"{stamp}_{tag}"
    d.mkdir(parents=True, exist_ok=False)
    return d


def write_provenance(study: Study, run: Path, command: list[str], extra: dict | None = None) -> None:
    watch = ["src/SampleTool.cpp", "src/BuildFitInput.cpp", "src/ConfigParser.cpp", "src/main.cpp",
             "include/BuildFitTools.h", "include/ConfigParser.h"]
    for a in study.raw["analyses"]:
        watch += [a["analysis_config"], a["fit_config"]]
    prov = {
        "created": provenance.now(),
        "command": command,
        "study_config": str(study.path),
        "repos": {
            "SVFastSimClosure": provenance.git_state(PKG_SRC.parent),
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


def _bin_payload(bins: list[binsmod.ResolvedBin]) -> list[dict]:
    return [dict(mode=b.mode, name=b.name, expression=b.expression, expression_no_hlt=b.expression_no_hlt) for b in bins]


def trigger_branches(bins: list[binsmod.ResolvedBin]) -> list[str]:
    return sorted({br for b in bins for br in b.branches if br.startswith("Trigger_")})


def run_yields(study: Study, points: list[SignalPoint], bins: list[binsmod.ResolvedBin], workers: int,
               step_size: int, task_entries: int, sims=("fullsim", "fastsim"), log=print) -> tuple[pd.DataFrame, dict]:
    payload = _bin_payload(bins)
    weights = {a.mode: a.weights.__dict__ for a in study.analyses}
    trig = trigger_branches(bins)
    tasks = []
    for p in points:
        for sim in sims:
            info = getattr(p, f"{sim}_info")
            if not info or not info.get("ok"):
                continue
            avail = set(info["branches"])
            is_fast = info["config"].get("sMCType") == 2
            common = dict(bins=payload, weights=weights, is_fastsim=is_fast, tree=study.samples["tree"],
                          step_size=step_size, timeout=study.io["timeout"],
                          trigger_branches=[t for t in trig if t in avail])
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
    rows, perf_files = [], []
    for (pk, sim), m in merged.items():
        for (mode, bn, hlt), acc in m["yields"].items():
            rows.append(dict(mode=mode, bin=bn, point=pk, sim=sim, hlt=hlt, n_raw=acc.n, sumw=acc.sumw, sumw2=acc.sumw2))
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
    return pd.DataFrame(rows), perf


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
                    timeout_s: int = 3600) -> dict:
    """Run the RDataFrame replica of LLPCombine in a separate Python process."""
    weights = {a.mode: a.weights for a in study.analyses}
    if len({(w.lumi_cpp(), w.sample_z, w.sample_g, w.target_z, w.target_g) for w in weights.values()}) != 1:
        raise RuntimeError("RDF backend assumes the same weight spec for all modes")
    w = next(iter(weights.values()))
    spec = dict(path=path, tree=study.samples["tree"], llpcombine_include=str(study.llpcombine_dir / "include"),
                threads=threads, bins=_bin_payload(bins),
                weights=dict(lumi_cpp=w.lumi_cpp(), sample_z=w.sample_z, sample_g=w.sample_g, target_z=w.target_z,
                             target_g=w.target_g))
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
    r = pd.DataFrame(rdf["yields"]).rename(columns={"n_raw": "rdf_n", "sumw": "rdf_sumw", "sumw2": "rdf_sumw2"})
    u = uproot_rows.rename(columns={"n_raw": "up_n", "sumw": "up_sumw", "sumw2": "up_sumw2"})
    t = u.merge(r, on=["mode", "bin", "hlt"], how="outer")
    t["n_match"] = t.up_n == t.rdf_n

    def close(a, b):
        return np.isclose(a, b, rtol=rel_tol, atol=1e-300)

    t["sumw_match"] = close(t.up_sumw, t.rdf_sumw)
    t["sumw2_match"] = close(t.up_sumw2, t.rdf_sumw2)
    t["ok"] = t.n_match & t.sumw_match & t.sumw2_match
    return t


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
