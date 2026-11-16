"""Command-line interface: ``svclosure <command> [options]``.

Commands
  manifest    build the resolved-bin and sample manifests (optionally open every file)
  show-bins   print the resolved LLPCombine selection of every SV bin
  validate    first-stage validation on one matched FullSim/FastSim point
  run         process all matched points -> yields, SF tables, figures
  plot        (re)make figures from an existing run directory
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

from . import pipeline as P
from . import stats

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "study.yaml"
DEFAULT_RESULTS = Path(__file__).resolve().parents[2] / "results"


def _log(msg: str) -> None:
    print(msg, flush=True)


def _select_points(points, keys: str | None, require_usable: bool = True):
    sel = [p for p in points if (not require_usable or p.usable)]
    if keys and keys != "all":
        want = set(keys.split(","))
        sel = [p for p in sel if p.key in want]
        missing = want - {p.key for p in sel}
        if missing:
            raise SystemExit(f"unknown/unusable point(s): {sorted(missing)}")
    return sel


def _report_points(points) -> None:
    n_ok = sum(p.usable for p in points)
    _log(f"signal points: {len(points)} discovered, {sum(p.paired for p in points)} paired, {n_ok} usable")
    for p in points:
        bad = [k for k in p.CRITICAL if p.checks.get(k) is not True and k in p.checks]
        flag = "OK " if p.usable else "BAD"
        extra = f"  failed={bad}" if bad else ""
        warn = f"  warnings={p.warnings}" if p.warnings else ""
        _log(f"  {flag} {p.key:28s}{extra}{warn}")
        for which in ("fullsim", "fastsim"):
            info = getattr(p, f"{which}_info")
            if info and not info["ok"]:
                _log(f"      {which} unreadable: {info['path']}\n        {info['error']}")


# ----------------------------------------------------------------------------- commands


def cmd_manifest(a) -> int:
    study = P.load_study(a.config)
    bins, problems = P.resolve_all_bins(study)
    points = P.discover_points(study, inspect=not a.no_inspect)
    run = P.new_run_dir(Path(a.results), "manifest")
    P.write_provenance(study, run, sys.argv)
    P.write_manifests(run, bins, problems, points)
    _log(f"bins: {len(bins)} SV bins ({sum(b.sxy == 'high' for b in bins)} high-S_xy, "
         f"{sum(b.role == 'anchor' for b in bins)} anchors); problems: {problems or 'none'}")
    _report_points(points)
    _log(f"-> {run}")
    return 0


def cmd_show_bins(a) -> int:
    study = P.load_study(a.config)
    bins, problems = P.resolve_all_bins(study)
    if a.mode:
        bins = [b for b in bins if b.mode == a.mode]
    if a.sxy:
        bins = [b for b in bins if b.sxy == a.sxy]
    if a.bin:
        bins = [b for b in bins if b.name in a.bin.split(",")]
    print(P.format_bins(bins))
    for an in study.analyses:
        print(f"[{an.mode}] luminosity (resolved as LLPCombine) = {an.luminosity} fb^-1 -> '{an.weights.lumi_cpp()}'; "
              f"weights: evtFillWgt*L*decayWeight(Z {an.weights.sample_z}->{an.weights.target_z}, "
              f"G {an.weights.sample_g}->{an.weights.target_g})")
    if problems:
        print("PROBLEMS:", *problems, sep="\n  ")
    return 0


def _yields_and_sf(study, points, bins, run, workers, step, task_entries):
    yields, perf = P.run_yields(study, points, bins, workers, step, task_entries, log=_log)
    yields.to_csv(run / "yields.csv", index=False)
    (run / "perf.json").write_text(json.dumps(perf, indent=1, default=str))
    bmeta = P.bin_frame(bins)
    sf = stats.sf_table(yields, bmeta)
    keys = [p.key for p in points]
    sfp = stats.sf_table(stats.pooled(yields, keys), bmeta)
    sf.to_csv(run / "sf.csv", index=False)
    sfp.to_csv(run / "sf_pooled.csv", index=False)
    return yields, perf, sf, sfp


def _plots(study, run: Path, sf: pd.DataFrame, sfp: pd.DataFrame, pts: pd.DataFrame, per_point: bool = True):
    from . import plotting

    rl = study.raw.get("plot", {}).get("rlabel", "(13.6 TeV)")
    fig = run / "figures"
    made = []
    for hlt in ("parity", "symmetric"):
        suffix = "" if hlt == "parity" else "_hltSymmetric"
        pooled_title = f"Pooled over {int(pts.usable.sum())} matched signal hypothes{'is' if int(pts.usable.sum()) == 1 else 'es'}"
        made += plotting.sf_plot(sfp, fig, f"sf_highSxy_pooled{suffix}", pooled_title, rl, hlt)
        made += plotting.sf_plot(sfp, fig, f"sf_highSxy_pooled{suffix}_noBinNames", pooled_title, rl, hlt,
                                 bin_names=False)
        if sf.point.nunique() > 1:
            made += plotting.heatmap(sf, pts, fig, f"sf_heatmap_highSxy{suffix}", rl, hlt, "high")
            made += plotting.heatmap(sf, pts, fig, f"sf_heatmap_lowSxy{suffix}", rl, hlt, "low")
            made += plotting.sf_vs_dm(sf, pts, fig, f"sf_vs_dm{suffix}", rl, hlt)
    if per_point:
        lab = dict(zip(pts.key, pts.short_label))
        for pk in sorted(sf.point.unique()):
            made += plotting.sf_plot(sf[sf.point == pk], fig / "per_point", f"sf_highSxy_{pk}",
                                     f"gogoGZ {lab.get(pk, pk)}", rl, "parity")
    return made


def cmd_validate(a) -> int:
    study = P.load_study(a.config)
    bins, problems = P.resolve_all_bins(study)
    run = P.new_run_dir(Path(a.results), f"validate_{a.point}")
    t0 = time.perf_counter()
    _log("=" * 100 + "\nResolved selections (LLPCombine ConfigParser::GetCombinedCuts semantics)\n" + "=" * 100)
    txt = P.format_bins(bins)
    print(txt)
    _log(f"bin structure problems: {problems or 'none'}")

    master_points = P.discover_points(study, inspect=False)
    sel = [p for p in master_points if p.key == a.point]
    if not sel:
        raise SystemExit(f"point {a.point} not in SampleTool pairs; known: {[p.key for p in master_points]}")
    from .samples import inspect_points

    inspect_points(sel, study.samples["tree"], study.samples["config_tree"], 2, study.io["timeout"])
    p = sel[0]
    P.write_provenance(study, run, sys.argv)
    P.write_manifests(run, bins, problems, sel)
    _report_points(sel)
    if not p.usable:
        _log("point not usable; stopping")
        return 2

    report: dict = {"point": p.key, "files": {}, "checks": {}}
    wspecs = [an.weights for an in study.analyses]
    for sim in ("fullsim", "fastsim"):
        info = getattr(p, f"{sim}_info")
        is_fast = info["config"].get("sMCType") == 2
        bc = P.branch_check(info, bins, wspecs, is_fast)
        report["files"][sim] = dict(path=info["path"], config_entry0=info["config"], config_entries=info["config_entries"],
                                    config_sums=info["config_sums"], tree_entries=info["tree_entries"],
                                    is_fastsim_by_sMCType=is_fast, branch_check=bc)
        _log(f"\n[{sim}] {info['path']}\n  config entry 0: {info['config']}  (config entries: {info['config_entries']})"
             f"\n  kuSkimTree entries: {info['tree_entries']}  sMCType==2 -> FastSim trigger stripping: {is_fast}"
             f"\n  branches needed: {bc['n_needed']}  missing: {bc['missing'] or 'none'}"
             f"\n  Trigger_* branches present: {bc['trigger_branches_present']}  absent: {bc['trigger_branches_absent']}")
        if bc["missing"]:
            _log("  !! missing branches: cannot evaluate those bins; stopping")
            (run / "validation.json").write_text(json.dumps(report, indent=1, default=str))
            return 3

    _log("\n--- uproot/awkward backend ---")
    yields, perf, sf, sfp = _yields_and_sf(study, sel, bins, run, a.workers, study.io["step_size"],
                                           study.io["task_entries"])
    for f in perf["files"]:
        _log(f"  {f['sim']}: {f['entries']} entries, {f['bytes']/1e6:.1f} MB requested, read {f['read_s']:.1f}s "
             f"({(f['MBps_read'] or 0):.1f} MB/s), select {f['select_s']:.1f}s, zero-weight events {f['n_zero_weight']}, "
             f"events with any Trigger_* true: {f['trigger_any_true']}, out-of-range index: {f['out_of_range'] or 'none'}"
             + (f"\n  ERRORS {f['errors']}" if f["errors"] else ""))
        finfo = getattr(p, f"{f['sim']}_info")
        info, sums = finfo["config"], finfo["config_sums"]
        if info.get("nTotEvts"):
            xs = info["sCrossSection"]
            ntot0, ntot_sum = info["nTotEvts"], sums.get("nTotEvts", float("nan"))
            per_evt0 = xs * 1000.0 / ntot0
            per_evt_sum = xs * 1000.0 / ntot_sum if ntot_sum else float("nan")
            mean_w = f["sum_evtFillWgt"] / f["entries"] if f["entries"] else float("nan")
            _log(f"    normalization: sCrossSection={xs:.6g} pb; config entries={finfo['config_entries']}; "
                 f"nTotEvts entry0={ntot0}, sum over entries={ntot_sum:.0f}"
                 f"\n      mean evtFillWgt in skim = {mean_w:.6g};  xsec*1000/nTot(entry0) = {per_evt0:.6g} "
                 f"(ratio {mean_w/per_evt0:.4f});  xsec*1000/nTot(sum) = {per_evt_sum:.6g} (ratio {mean_w/per_evt_sum:.4f})"
                 f"\n      expected signal events (sum w) at L: {f['sum_evtFillWgt']*study.analyses[0].luminosity:.4g}")
            report["files"][f["sim"]]["normalization"] = dict(
                sCrossSection=xs, nTotEvts_entry0=ntot0, nTotEvts_sum=ntot_sum, mean_evtFillWgt=mean_w,
                ratio_to_entry0=mean_w / per_evt0, ratio_to_sum=mean_w / per_evt_sum, sum_evtFillWgt=f["sum_evtFillWgt"],
                zero_weight_events=f["n_zero_weight"], trigger_any_true=f["trigger_any_true"],
                config_branch_entries=finfo.get("extra_config", {}).get("branch_entries"),
                config_n_distinct=finfo.get("extra_config", {}).get("n_distinct"))
    if any(f["errors"] for f in perf["files"]):
        _log("!! read errors; stopping")
        return 4

    if not a.no_rdf:
        _log("\n--- RDataFrame replica of LLPCombine (separate process) ---")
        cmp_all = []
        for sim in ("fullsim", "fastsim"):
            rdf = P.run_rdf_backend(study, getattr(p, sim), bins, run / "rdf", threads=a.rdf_threads)
            up = yields[(yields.sim == sim)][["mode", "bin", "hlt", "n_raw", "sumw", "sumw2"]]
            c = P.compare_backends(up, rdf)
            c["sim"] = sim
            cmp_all.append(c)
            strip_ok = all(y["cpp_stripped"] == next(b.expression_no_hlt for b in bins if b.name == y["bin"] and b.mode == y["mode"])
                           for y in rdf["yields"])
            _log(f"  {sim}: RDF sMCType={rdf['sMCType']} is_fastsim={rdf['is_fastsim']}  {rdf['wall_s']:.1f}s  "
                 f"C++ std::regex trigger stripping == Python: {strip_ok}  "
                 f"bins/modes agreeing: {int(c.ok.sum())}/{len(c)}")
            report["checks"][f"rdf_parity_{sim}"] = bool(c.ok.all())
            report["checks"][f"trigger_strip_cpp_equal_{sim}"] = strip_ok
            for r in c[~c.ok].itertuples():
                _log(f"    MISMATCH {r.mode}/{r.bin}/{r.hlt}: uproot n={r.up_n} w={r.up_sumw!r}  rdf n={r.rdf_n} w={r.rdf_sumw!r}")
        pd.concat(cmp_all).to_csv(run / "parity_uproot_vs_rdf.csv", index=False)

    if a.compare_json:
        _log("\n--- comparison with existing LLPCombine JSON ---")
        t = P.compare_llpcombine_json(study, Path(a.compare_json), Path(a.compare_config), p.fullsim, p.key, a.workers, log=_log)
        t.to_csv(run / "compare_llpcombine_json.csv", index=False)
        for r in t.itertuples():
            _log(f"  {r.bin:42s} n {r.pred_n:6d} vs {r.json_n}  w {r.pred_w:.6g} vs {r.json_w}  "
                 f"(rel {r.w_reldiff:+.2e})  err rel {r.err_reldiff:+.2e}")
        good = t.n_match & (t.w_reldiff.abs() < 1e-6) & (t.err_reldiff.abs() < 1e-6)
        report["checks"]["llpcombine_json_match"] = bool(good.all())
        report["llpcombine_json"] = dict(json=a.compare_json, config=a.compare_config, n_bins=len(t), n_match=int(good.sum()),
                                         K=t.attrs["K"], lumis=t.attrs["lumis"])

    _log("\n--- SF (hlt=parity), high-S_xy bins ---")
    show = sf[(sf.sxy == "high")].sort_values(["mode", "order", "hlt"])
    for r in show.itertuples():
        _log(f"  {r.mode:13s} {r.bin:40s} {r.role:6s} {r.hlt:9s} full n={r.full_n_raw:5d} Y={r.full_sumw:.4g}±{r.full_err:.2g}"
             f"  fast n={r.fast_n_raw:5d} Y={r.fast_sumw:.4g}±{r.fast_err:.2g}  SF={r.sf:.3f}±{r.sf_err:.3f} {r.status}")
    pts = P.points_frame(sel)
    made = _plots(study, run, sf, sfp, pts, per_point=False)
    report["figures"] = [str(m) for m in made]
    report["seconds"] = time.perf_counter() - t0
    (run / "validation.json").write_text(json.dumps(report, indent=1, default=str))
    _log(f"\nchecks: {report['checks']}\n-> {run}")
    return 0 if all(report["checks"].values()) else 5


def cmd_run(a) -> int:
    study = P.load_study(a.config)
    bins, problems = P.resolve_all_bins(study)
    if problems:
        _log(f"bin structure problems: {problems}")
    points = P.discover_points(study, inspect=True)
    _report_points(points)
    sel = _select_points(points, a.points)
    run = P.new_run_dir(Path(a.results), a.tag)
    P.write_provenance(study, run, sys.argv)
    P.write_manifests(run, bins, problems, points)
    workers = a.workers or study.io["workers"]
    yields, perf, sf, sfp = _yields_and_sf(study, sel, bins, run, workers, a.step_size or study.io["step_size"],
                                           a.task_entries or study.io["task_entries"])
    _log(f"wall {perf['wall_s']:.1f}s, {perf['total_bytes']/1e9:.2f} GB requested, "
         f"aggregate {(perf['aggregate_MBps'] or 0):.1f} MB/s with {workers} workers")
    bad = [f for f in perf["files"] if f["errors"]]
    for f in bad:
        _log(f"  ERRORS {f['point']} {f['sim']}: {f['errors']}")
    pts = P.points_frame(points)
    pts = pts[pts.key.isin([p.key for p in sel])]
    _plots(study, run, sf, sfp, pts)
    _log(f"-> {run}")
    return 1 if bad else 0


def cmd_plot(a) -> int:
    study = P.load_study(a.config)
    run = Path(a.run)
    sf, sfp = pd.read_csv(run / "sf.csv"), pd.read_csv(run / "sf_pooled.csv")
    pts = pd.read_csv(run / "manifest" / "points.csv")
    pts = pts[pts.key.isin(sf.point.unique())]
    for m in _plots(study, run, sf, sfp, pts, per_point=not a.no_per_point):
        _log(str(m))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="svclosure", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG), help="study YAML (default: configs/study.yaml)")
    ap.add_argument("--results", default=str(DEFAULT_RESULTS), help="results base directory")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("manifest", help="build bin + sample manifests")
    s.add_argument("--no-inspect", action="store_true", help="do not open the files (no XRootD access)")
    s.set_defaults(func=cmd_manifest)

    s = sub.add_parser("show-bins", help="print resolved selections")
    s.add_argument("--mode", choices=["compressed", "noncompressed"])
    s.add_argument("--sxy", choices=["high", "low"])
    s.add_argument("--bin", help="comma-separated bin names")
    s.set_defaults(func=cmd_show_bins)

    s = sub.add_parser("validate", help="first-stage validation on one matched point")
    s.add_argument("--point", required=True, help="LLPCombine signal key, e.g. gogoGZ_2300_1300_1000_10")
    s.add_argument("--workers", type=int, default=2)
    s.add_argument("--no-rdf", action="store_true", help="skip the RDataFrame parity check")
    s.add_argument("--rdf-threads", type=int, default=0, help="ROOT implicit MT threads in the RDF process (0 = off)")
    s.add_argument("--compare-json", help="existing LLPCombine output JSON")
    s.add_argument("--compare-config", help="analysis config that produced --compare-json")
    s.set_defaults(func=cmd_validate)

    s = sub.add_parser("run", help="process all matched points")
    s.add_argument("--points", default="all", help="'all' or comma-separated keys")
    s.add_argument("--workers", type=int)
    s.add_argument("--step-size", type=int)
    s.add_argument("--task-entries", type=int)
    s.add_argument("--tag", default="scan")
    s.set_defaults(func=cmd_run)

    s = sub.add_parser("plot", help="remake figures for a run directory")
    s.add_argument("--run", required=True)
    s.add_argument("--no-per-point", action="store_true")
    s.set_defaults(func=cmd_plot)

    a = ap.parse_args(argv)
    if getattr(a, "compare_json", None) and not getattr(a, "compare_config", None):
        ap.error("--compare-json needs --compare-config")
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
