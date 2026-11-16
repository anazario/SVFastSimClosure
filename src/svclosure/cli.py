"""Command-line interface: ``svclosure <command> [options]``.

Commands
  manifest       build the resolved-bin and sample manifests (optionally open every file)
  show-bins      print the resolved LLPCombine selection of every SV bin
  validate       validation on one matched FullSim/FastSim point: RDataFrame parity (direct and
                 lifetime-reweighted, element-wise weights), optional LLPCombine-JSON comparison
  run            process all matched points (generated lifetimes only) -> tables, figures
  lifetime-scan  same files, every configured target lifetime in one pass -> lifetime tables, figures
  plot           (re)make figures from an existing run directory

Examples
  svclosure run --workers 3
  svclosure lifetime-scan --workers 3                              # targets from configs/study.yaml
  svclosure lifetime-scan --targets "10:5,10;50:30,50"             # fewer targets
  svclosure lifetime-scan --targets "10:2,5,10,15,20;50:10,20,30,40,50,70,100" --tag lifetime_wide
  svclosure validate --point gogoGZ_2300_1300_1000_10              # + lifetime parity for 10 cm targets
  svclosure plot --run results/<run-dir>
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


def _process(study, points, all_points, bins, run, workers, step, task_entries, prov, scan=None):
    """One pass over the files -> every table (written to ``run``) -> (tables, YieldRun)."""
    from . import bins as B
    from . import lifetime as L
    from . import outputs as O
    from .provenance import short_rev

    avail = set.intersection(*(set(p.fullsim_info["branches"]) & set(p.fastsim_info["branches"]) for p in points))
    refs = B.reference_selections(bins, avail)
    q = scan.quality if scan else L.from_config(study.raw.get("lifetime_scan")).quality
    targets_of = None
    if scan is not None:
        def targets_of(p):
            src = L.source_lifetime(p).cm
            return src, scan.targets_for(src)
    yr = P.run_yields(study, points, bins, workers, step, task_entries, log=_log, refs=refs, targets_of=targets_of,
                      policy=scan.invalid_ctau_policy if scan else "llpcombine", topk=q.topk)
    (run / "perf.json").write_text(json.dumps(yr.perf, indent=1, default=str))
    pts = P.points_frame(all_points, study.regimes)
    pts.to_csv(run / "manifest" / "points.csv", index=False)
    pts = pts[pts.key.isin([p.key for p in points])]
    rev = short_rev(prov["repos"]["SVFastSimClosure"])
    tables = O.build(study, yr, bins, refs, points, pts, q, rev, scan=scan)
    cfg = dict(quality=q.__dict__, lifetime_scan=scan.to_dict() if scan else None,
               signal_regimes=study.raw.get("signal_regimes"), overlaps=study.raw.get("overlaps"),
               reference_selections=[r.to_dict() for r in refs], points=[p.key for p in points],
               hlt_modes=list(P.HLT_MODES))
    O.write(run, tables, {"run_config": cfg})
    O.overlap_matrices(run, yr, bins, (study.raw.get("overlaps") or {}).get("benchmark_points", []))
    return tables, yr, pts


def _overlap_verdict(study, tables) -> int:
    summ = tables["overlap_summary"]
    for k in ("structural_failures", "unexpected_overlaps", "unexpected_omissions"):
        for r in summ.get(k, []):
            _log(f"  {k[:-1].upper()}: {r}")
    if summ["structural_failures"]:
        _log("!! within-channel bins share events: structural failure")
        return 3
    if (study.raw.get("overlaps") or {}).get("fail_on_unexpected") and (summ["unexpected_overlaps"] or
                                                                      summ["unexpected_omissions"]):
        _log("!! unexpected overlaps/omissions and overlaps.fail_on_unexpected is set")
        return 3
    return 0


def _plots(study, run: Path, tables: dict, pts: pd.DataFrame, per_point: bool = True, lifetime_figs: bool = False):
    from . import plotting
    from . import lifetime_plots

    pc = study.raw.get("plot", {})
    rl = pc.get("rlabel", "(13.6 TeV)")
    fig = run / "figures"
    made = plotting.study_figures(tables, pts, study.regimes, fig, rl, per_point=per_point,
                                  all_hypotheses=pc.get("include_all_hypotheses_diagnostic", True),
                                  full_names=pc.get("full_bin_name_figures", False))
    if lifetime_figs and "lifetime_sf" in tables:
        from . import lifetime as L

        scan = L.from_config(study.raw.get("lifetime_scan"))
        made += lifetime_plots.all_figures(tables, pts, study.regimes, scan, fig / "lifetime", rl)
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
    prov = P.write_provenance(study, run, sys.argv)
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
    from . import lifetime as L

    scan = None
    sl = L.source_lifetime(p)
    report["source_lifetime"] = sl.__dict__
    _log(f"  generated lifetime: FullSim {sl.fullsim_cm} cm, FastSim {sl.fastsim_cm} cm, LLPCombine key {sl.key_cm} cm"
         f" -> {'OK' if sl.ok else 'PROBLEM ' + str(sl.problem)}")
    if not a.no_lifetime and sl.ok:
        scan = L.from_config(study.raw.get("lifetime_scan"), a.targets)
        if scan.targets_for(sl.cm) is None:
            _log(f"  no lifetime targets configured for {sl.cm} cm: lifetime parity skipped")
            scan = None
    tables, yr, pts = _process(study, sel, sel, bins, run, a.workers, study.io["step_size"], study.io["task_entries"],
                               prov, scan)
    yields, perf, sf = yr.yields, yr.perf, tables["sf"]
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
        cmp_all, wcmp = [], []
        tg = scan.targets_for(sl.cm) if scan else []
        lspec = dict(source_cm=sl.cm, targets=tg, variant_keys=[L.variant_key(t) for t in tg]) if tg else None
        for sim in ("fullsim", "fastsim"):
            rdf = P.run_rdf_backend(study, getattr(p, sim), bins, run / "rdf", threads=a.rdf_threads,
                                    lifetime_spec=lspec, dump_weights=bool(lspec) and not a.rdf_threads)
            up = yields[(yields.sim == sim) & (yields.kind == "bin")][["mode", "bin", "hlt", "variant", "n_raw", "sumw", "sumw2"]]
            c = P.compare_backends(up, rdf)
            c["sim"] = sim
            cmp_all.append(c)
            if rdf.get("weights_dump"):
                w = P.compare_weight_dump(rdf["weights_dump"], getattr(p, sim), study.samples["tree"],
                                          study.analyses[0].weights, sl.cm, tg, study.io["timeout"])
                w["sim"] = sim
                wcmp.append(w)
                for r in w.itertuples():
                    _log(f"    {sim} weights {r.column:18s} target {r.target_ctau_cm}: {r.n_bitwise_equal}/{r.n} bitwise equal,"
                         f" max ulp(float32) {r.max_ulp_float32}, max rel diff {r.max_rel_diff:.2e}")
                report["checks"][f"lifetime_weights_bitwise_{sim}"] = bool((w.n_bitwise_equal == w.n).all())
            for vk, g in c.groupby("variant"):
                _log(f"    {sim} {vk:12s}: {int(g.ok.sum())}/{len(g)} agree; counts identical {int(g.n_match.sum())}/{len(g)};"
                     f" max |rel diff| sumw {g.sumw_reldiff.abs().max():.1e}, sumw2 {g.sumw2_reldiff.abs().max():.1e}")
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
        if wcmp:
            pd.concat(wcmp).to_csv(run / "parity_lifetime_weights_eventwise.csv", index=False)

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
    report["checks"]["overlaps_ok"] = _overlap_verdict(study, tables) == 0
    made = _plots(study, run, tables, pts, per_point=False, lifetime_figs=False)
    report["figures"] = [str(m) for m in made]
    report["seconds"] = time.perf_counter() - t0
    (run / "validation.json").write_text(json.dumps(report, indent=1, default=str))
    _log(f"\nchecks: {report['checks']}\n-> {run}")
    return 0 if all(report["checks"].values()) else 5


def _scan_common(a, scan=None, tag=None) -> int:
    study = P.load_study(a.config)
    bins, problems = P.resolve_all_bins(study)
    if problems:  # structural bin failures stop the production scan
        for pr in problems:
            _log(f"STRUCTURE PROBLEM: {pr}")
        raise SystemExit("bin structure problems: refusing to run (see above)")
    points = P.discover_points(study, inspect=True)
    _report_points(points)
    sel = _select_points(points, a.points)
    if scan is not None:
        from . import lifetime as L

        keep = []
        for p in sel:
            sl = L.source_lifetime(p)
            if not sl.ok:
                _log(f"  SKIP {p.key}: generated lifetime check failed: {sl.problem}")
            elif scan.targets_for(sl.cm) is None:
                _log(f"  SKIP {p.key}: no targets configured for generated lifetime {sl.cm:g} cm")
            else:
                keep.append(p)
        sel = keep
        if not sel:
            raise SystemExit("no points with configured source lifetimes")
        _log("lifetime scan: " + "; ".join(f"{s:g} cm -> {scan.targets_for(s)}" for s in scan.targets)
             + f"  (policy for invalid Xa/Xb_ctau: {scan.invalid_ctau_policy})")
    run = P.new_run_dir(Path(a.results), tag or a.tag)
    prov = P.write_provenance(study, run, sys.argv)
    P.write_manifests(run, bins, problems, points)
    workers = a.workers or study.io["workers"]
    tables, yr, pts = _process(study, sel, points, bins, run, workers, a.step_size or study.io["step_size"],
                               a.task_entries or study.io["task_entries"], prov, scan)
    perf = yr.perf
    _log(f"wall {perf['wall_s']:.1f}s, {perf['total_bytes']/1e9:.2f} GB requested, "
         f"aggregate {(perf['aggregate_MBps'] or 0):.1f} MB/s with {workers} workers")
    bad = [f for f in perf["files"] if f["errors"]]
    for f in bad:
        _log(f"  ERRORS {f['point']} {f['sim']}: {f['errors']}")
    an = yr.files.filter(like="n_ctau").sum().to_dict() if len(yr.files) else {}
    _log(f"lifetime-input anomalies (all skimmed events): {an}; events listed in lifetime_anomalies.csv: {len(yr.anomalies)}")
    if scan is not None:
        chk = tables["lifetime_checks"]["direct_target_equals_noreweight"]
        _log(f"direct target == no-reweight: {chk['bitwise_equal']}/{chk['rows']} rows bitwise identical")
        q = tables["lifetime_quality"]
        for r in q[q.level == "channel"].itertuples():
            _log(f"  {r.source_ctau_cm:g} -> {r.target_ctau_cm:g} cm [{r.mode}]: {r.n_reliable}/{r.n_direct_ok} channel SFs "
                 f"reliable, median N_eff ratio {r.median_neff_ratio:.2f} -> {r.verdict}")
    rc = _overlap_verdict(study, tables)
    if not a.no_plots:
        _plots(study, run, tables, pts, per_point=True, lifetime_figs=scan is not None)
    _log(f"-> {run}")
    return 1 if bad else rc


def cmd_run(a) -> int:
    return _scan_common(a)


def cmd_lifetime_scan(a) -> int:
    from . import lifetime as L

    study = P.load_study(a.config)
    scan = L.from_config(study.raw.get("lifetime_scan"), a.targets, min_neff=a.min_neff,
                         min_neff_reweighted=a.min_neff_reweighted, max_top1_sumw2_frac=a.max_top1)
    if a.invalid_ctau_policy:
        scan.invalid_ctau_policy = a.invalid_ctau_policy
    if a.no_crosscheck:
        scan.crosscheck_targets = {}
    problems = scan.validate()
    if problems:
        raise SystemExit("invalid lifetime scan: " + "; ".join(problems))
    return _scan_common(a, scan, a.tag)


def _read_tables(run: Path) -> dict:
    t = {}
    for p in run.glob("*.csv"):
        t[p.stem] = pd.read_csv(p)
    for p in run.glob("*.json"):
        if p.stem in ("overlap_summary", "lifetime_checks"):
            t[p.stem] = json.loads(p.read_text())
    return t


def cmd_plot(a) -> int:
    study = P.load_study(a.config)
    run = Path(a.run)
    tables = _read_tables(run)
    if "sf_channel" not in tables:
        raise SystemExit(f"{run} predates the current output layout; rerun 'svclosure run' (see README: migration)")
    pts = pd.read_csv(run / "manifest" / "points.csv")
    pts = pts[pts.key.isin(tables["sf"].point.unique())]
    for m in _plots(study, run, tables, pts, per_point=not a.no_per_point, lifetime_figs="lifetime_sf" in tables):
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
    s.add_argument("--no-lifetime", action="store_true", help="skip the lifetime-reweighting parity checks")
    s.add_argument("--targets", help="lifetime targets for the parity check (default: lifetime_scan.targets)")
    s.set_defaults(func=cmd_validate)

    def common(s, tag):
        s.add_argument("--points", default="all", help="'all' or comma-separated keys")
        s.add_argument("--workers", type=int)
        s.add_argument("--step-size", type=int)
        s.add_argument("--task-entries", type=int)
        s.add_argument("--tag", default=tag, help=f"run-directory suffix (default {tag})")
        s.add_argument("--no-plots", action="store_true")

    s = sub.add_parser("run", help="process all matched points at their generated lifetimes")
    common(s, "scan")
    s.set_defaults(func=cmd_run)

    s = sub.add_parser("lifetime-scan", help="event-level lifetime reweighting to every configured target",
                       description="Every file is read once; masks are evaluated once per chunk and all target "
                                   "lifetimes are accumulated together.  Lifetimes in cm.")
    common(s, "lifetime")
    s.add_argument("--targets", help='source:targets map, e.g. "10:1,5,10,15;50:20,30,40,50,70" (overrides the YAML)')
    s.add_argument("--min-neff", type=float, help="status low_stats below this N_eff (default from YAML)")
    s.add_argument("--min-neff-reweighted", type=float, help="N_eff needed to draw a reweighted point as trustworthy")
    s.add_argument("--max-top1", type=float, help="max single-event fraction of sum w^2 before weight_concentrated")
    s.add_argument("--invalid-ctau-policy", choices=["llpcombine", "exclude", "fail"])
    s.add_argument("--no-crosscheck", action="store_true", help="drop lifetime_scan.crosscheck_targets (50->10, 10->50)")
    s.set_defaults(func=cmd_lifetime_scan)

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
