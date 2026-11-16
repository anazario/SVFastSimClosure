"""From one pass over the files (pipeline.YieldRun) to every table of a run directory.

Legacy files keep their names and original columns (new columns are appended):
  yields.csv, sf.csv, sf_pooled.csv, sf_high_vs_low_pooled.csv
New files are documented in the OUTPUTS.md written into each run directory.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import overlap as ov
from . import stats
from .pipeline import LEGACY_YIELD_COLS, annotate, bin_frame, legacy_yields

OUTPUTS_MD = """# Output files

All yields are weighted with LLPCombine's `evtrwt` (evtFillWgt x L x decayWeight [x xawt x xbwt]).
Lifetimes are c*tau in cm.  `point` is always the LLPCombine signal key of the *generated* sample
(its last token is the generated lifetime); `source_ctau_cm` repeats it explicitly and
`target_ctau_cm` is the lifetime the events are weighted to (equal to the source for direct MC).
`variant` = `noreweight` (no lifetime factor; the original study, bit for bit) or `tgt<T>cm`.
`hlt` = `parity` (fit: Trigger_* dropped for FastSim only) or `symmetric` (dropped for both).
Every per-point table carries the mass hypothesis (`mass_key`, masses, `dm`), the sample path(s)
and the svclosure revision (`svclosure_commit`, `svclosure_dirty`, `svclosure_diff_sha256`);
the full provenance is in `provenance.json`, the complete configuration in `run_config.json`.

## Unreweighted study (legacy names; original columns first, new ones appended)
* `yields.csv` - per (mode, bin, point, sim, hlt): n_raw, sumw, sumw2 (+ concentration, anomalies, metadata)
* `sf.csv` - per (mode, bin, point, hlt): FullSim and FastSim yields, SF, sf_err, status, sf_class
* `sf_pooled.csv` - **ratio of sums over all hypotheses of both generated lifetimes** (`pool = all_hypotheses`).
  A diagnostic only: it mixes regimes and lifetimes.  Not a fitted common SF.
* `sf_high_vs_low_pooled.csv` - high/low-S_xy double ratio for that same pool (columns of the earlier ad-hoc
  file; `ratio_err` now includes the high/low covariance, `ratio_err_uncorrelated` is the old definition)

## New
* `sf_pooled_regime.csv` - ratio of sums per analysis mode over **its own** signal regime
  (Compressed mode: compressed hypotheses; High-Delta-m mode: high-Delta-m hypotheses), separately per
  generated lifetime (`pool = <regime>_ct<cm>`)
* `sf_channel.csv`, `sf_channel_pooled_regime.csv` - channel-integrated SFs: sum of the disjoint bins of a
  channel group (`<group>` = all bins, `<group>:SR` = SR bins only)
* `highlow.csv`, `highlow_pooled_regime.csv` - per high/low pair: both SFs, DR = SF_high/SF_low with the
  covariance from events in both selections (`*_n_both`, `*_cov_hl`), `disjoint`, and `*_exhaust_frac`
  (fraction of the S_xy-inclusive reference population covered by high OR low)
* `overlap_bin_pairs.csv` - every bin pair sharing events, with relation, expectation and verdict
* `overlap_omissions.csv` - per channel envelope: events in no bin, explained by static kinematic gaps or not
* `overlap_summary.json`; `overlap_matrices/` - bin x bin and channel x channel event-count matrices for the
  benchmark points
* `reference_yields.csv` - the reference selections (envelopes, S_xy-inclusive, omissions)
* `sample_lifetime_inputs.csv` - per file: Xa/Xb_ctau validity counts and means
* `lifetime_anomalies.csv` - every event with invalid lifetime inputs or != 1 decay flag, with its bins/weights

## Lifetime scan (svclosure lifetime-scan)
* `lifetime_yields.csv` - per (selection, point, sim, hlt, target): n_raw, sumw, sumw2, w_max,
  frac_wmax_sumw (largest single weight / total), top{1,3,5}_sumw2_frac, anomalies
* `lifetime_sf.csv`, `lifetime_sf_channel.csv` - per-hypothesis SFs per target; `status`/`sf_class` use the
  quality criteria in `run_config.json`
* `lifetime_sf_pooled_regime.csv`, `lifetime_sf_channel_pooled_regime.csv` - regime ratio-of-sums pools per
  (generated lifetime, target)
* `lifetime_cov_yields.csv` - sum_events w_i w_j between targets i, j of the same generated sample
  (per point, sim, selection, hlt).  This is Cov(Y_i, Y_j) of the weighted yields; diagonal = sumw2.
* `lifetime_cov_logsf.csv` - Cov(ln SF_i, ln SF_j) = Cov_F/(F_i F_j) + Cov_S/(S_i S_j) and the correlation,
  first-order; `gaussian_ok` is false when either point is not status ok.  Targets from the SAME generated
  sample are strongly correlated: never treat them as independent measurements.
* `lifetime_highlow.csv` - high/low double ratios per target
* `lifetime_source_crosscheck.csv`, `lifetime_source_crosscheck_channel.csv` - direct MC at T vs the other
  generated sample reweighted to T (`role = cross_check` for 50->10 cm, `stress_test` for 10->50 cm), z-scores
* `lifetime_quality.csv` - per (generated lifetime, target): reliability fractions, N_eff ratios, verdict
* `lifetime_checks.json` - direct-target == no-reweight identity, anomaly totals, source-lifetime checks
"""


def _sf_points(sf: pd.DataFrame) -> pd.DataFrame:
    return sf[sf.point != "pooled"]


def build(study, yr, bins, refs, points, pts_frame: pd.DataFrame, quality, prov_cols: dict, scan=None) -> dict:
    """All tables of one run (dict name -> DataFrame / dict)."""
    T: dict = {}
    bmeta = bin_frame(bins)
    cmeta = stats.channel_meta(bmeta)
    y = annotate(yr.yields, points, prov_cols)
    usable = list(pts_frame[pts_frame.usable].key) if "usable" in pts_frame else list(pts_frame.key)

    # ---- unreweighted (variant noreweight) ------------------------------------------------------------
    y0 = y[y.variant == "noreweight"]
    yb0 = y0[y0.kind == "bin"]
    T["yields"] = legacy_yields(y)
    T["reference_yields"] = y0[y0.kind != "bin"].reset_index(drop=True)
    sf = stats.sf_table(yb0.drop(columns=["variant", "target_ctau_cm", "source_ctau_cm"]), bmeta, quality)
    T["sf"] = annotate(sf, points, prov_cols)
    allpool = stats.pooled(yb0[LEGACY_YIELD_COLS + ["w_max"]], usable, "all_hypotheses")
    T["sf_pooled"] = stats.sf_table(allpool, bmeta, quality).assign(estimator="ratio_of_sums")
    reg = stats.regime_pools(yb0[LEGACY_YIELD_COLS + ["w_max", "source_ctau_cm"]], pts_frame[pts_frame.usable], study.regimes)
    T["sf_pooled_regime"] = stats.sf_table(reg, bmeta, quality).assign(estimator="ratio_of_sums")
    ch = stats.channel_yields(yb0.drop(columns=["variant", "target_ctau_cm"]), bmeta)
    T["sf_channel"] = annotate(stats.sf_table(ch.drop(columns=["source_ctau_cm"]), cmeta, quality), points, prov_cols)
    regc = stats.regime_pools(ch, pts_frame[pts_frame.usable], study.regimes)
    T["sf_channel_pooled_regime"] = stats.sf_table(regc, cmeta, quality).assign(estimator="ratio_of_sums")

    o0 = yr.overlaps[yr.overlaps.variant == "noreweight"] if len(yr.overlaps) else yr.overlaps
    refy = y0[y0.kind == "sxy_inclusive"]
    hl = stats.highlow(yb0[LEGACY_YIELD_COLS], o0, bmeta, quality, references=refy[LEGACY_YIELD_COLS])
    T["highlow"] = annotate(hl, points, prov_cols)
    hlp = []
    pool_sets = [("all_hypotheses", usable)]
    for regime, mode in study.regimes.mode_of.items():
        pm = pts_frame[pts_frame.usable & pts_frame.regime_unambiguous & (pts_frame.regime == regime)]
        for src, g in pm.groupby("ctau_cm_num"):
            pool_sets.append((f"{regime}_ct{src:g}", list(g.key)))
    for label, keys in pool_sets:
        py = stats.pooled(yb0[LEGACY_YIELD_COLS], keys, label).drop(columns=["estimator", "n_hypotheses"])
        po = stats.pool_overlaps(o0, keys, label)
        pr = stats.pooled(refy[LEGACY_YIELD_COLS], keys, label).drop(columns=["estimator", "n_hypotheses"])
        if label != "all_hypotheses":
            mode = study.regimes.mode_of[label.split("_ct")[0]]
            py, pr = py[py["mode"] == mode], pr[pr["mode"] == mode]
        h = stats.highlow(py, po, bmeta, quality, references=pr)
        h["pool"], h["n_hypotheses"] = label, len(keys)
        hlp.append(h)
    hlp = pd.concat(hlp, ignore_index=True)
    T["highlow_pooled_regime"] = hlp[hlp.pool != "all_hypotheses"].reset_index(drop=True)
    T["sf_high_vs_low_pooled"] = stats.highlow_legacy(hlp[hlp.pool == "all_hypotheses"], T["sf"])

    # ---- overlaps / omissions / anomalies -------------------------------------------------------------
    cls = ov.classify(yr.overlaps, bmeta)
    om = ov.omissions(y, refs)
    T["overlap_bin_pairs"] = cls
    T["overlap_omissions"] = om
    T["overlap_summary"] = ov.summarize(cls, om)
    T["lifetime_anomalies"] = yr.anomalies
    T["sample_lifetime_inputs"] = yr.files

    # ---- lifetime scan --------------------------------------------------------------------------------
    if scan is not None:
        T.update(_lifetime_tables(y, yr, bmeta, cmeta, points, pts_frame, quality, prov_cols, scan, study))
    return T


def _lifetime_tables(y, yr, bmeta, cmeta, points, pts_frame, quality, prov_cols, scan, study) -> dict:
    T = {}
    yl = y[y.variant != "noreweight"]
    ylb = yl[yl.kind == "bin"]
    T["lifetime_yields"] = yl.reset_index(drop=True)
    sfl = stats.sf_table(ylb[LEGACY_YIELD_COLS + ["variant", "source_ctau_cm", "target_ctau_cm", "w_max",
                                                  "top1_sumw2_frac", "top3_sumw2_frac", "top5_sumw2_frac"]], bmeta, quality)
    T["lifetime_sf"] = annotate(sfl, points, prov_cols)
    chl = stats.channel_yields(ylb, bmeta)
    sfc = stats.sf_table(chl, cmeta, quality)
    T["lifetime_sf_channel"] = annotate(sfc, points, prov_cols)
    usable = pts_frame[pts_frame.usable]
    reg = stats.regime_pools(ylb[LEGACY_YIELD_COLS + ["variant", "source_ctau_cm", "target_ctau_cm", "w_max"]], usable,
                             study.regimes)
    T["lifetime_sf_pooled_regime"] = stats.sf_table(reg, bmeta, quality).assign(estimator="ratio_of_sums")
    regc = stats.regime_pools(chl, usable, study.regimes)
    T["lifetime_sf_channel_pooled_regime"] = stats.sf_table(regc, cmeta, quality).assign(estimator="ratio_of_sums")

    cross = yr.cross[(yr.cross.variant_i != "noreweight") & (yr.cross.variant_j != "noreweight")]
    ccross = stats.channel_cross(cross, bmeta)
    cov = pd.concat([cross[cross.kind == "bin"], ccross], ignore_index=True)
    T["lifetime_cov_yields"] = annotate(cov, points, prov_cols)
    T["lifetime_cov_logsf"] = pd.concat([stats.logsf_covariance(cross[cross.kind == "bin"], sfl),
                                         stats.logsf_covariance(ccross, sfc)], ignore_index=True)
    ol = yr.overlaps[yr.overlaps.variant != "noreweight"] if len(yr.overlaps) else yr.overlaps
    refy = yl[yl.kind == "sxy_inclusive"]
    keep = LEGACY_YIELD_COLS + ["variant", "source_ctau_cm", "target_ctau_cm"]
    T["lifetime_highlow"] = annotate(stats.highlow(ylb[keep], ol, bmeta, quality, references=refy[keep]), points, prov_cols)
    T["lifetime_source_crosscheck"] = stats.source_crosscheck(sfl, quality)
    T["lifetime_source_crosscheck_channel"] = stats.source_crosscheck(sfc, quality)
    T["lifetime_quality"] = quality_table(sfl, sfc, quality, pts_frame, study.regimes)
    for k in [k for k in T if k.startswith("lifetime_") and isinstance(T[k], pd.DataFrame)]:
        t = T[k]
        if len(t) and "target_ctau_cm" in t.columns and "source_ctau_cm" in t.columns:
            t["target_role"] = [scan.role(s_, t_) for s_, t_ in zip(t.source_ctau_cm, t.target_ctau_cm)]
    T["lifetime_checks"] = lifetime_checks(y, yr, pts_frame)
    return T


def quality_table(sfl: pd.DataFrame, sfc: pd.DataFrame, q, pts_frame: pd.DataFrame | None = None,
                  regimes=None) -> pd.DataFrame:
    """Per (generated lifetime, target, mode): how many per-hypothesis SFs remain reliable.

    Only regime-matched hypotheses and high-S_xy selections enter (Compressed mode: compressed points,
    High-Delta-m mode: high-Delta-m points).  Denominator: cells whose direct (target = source) SF is
    status ok with N_eff >= min_neff_reweighted in both samples, so the fraction measures what the
    reweighting itself costs, not the sparseness of the generated sample.  N_eff ratio: min(N_eff Full, N_eff Fast) at
    the target over the same at the source, median over those cells.
    """
    rows = []
    allowed = None
    if pts_frame is not None and regimes is not None:
        pm = pts_frame[pts_frame.regime_unambiguous]
        allowed = {(regimes.mode_of[r], k) for r, k in zip(pm.regime, pm.key)}
    for level, t, sel in (("bin", sfl, sfl.sxy == "high"), ("channel", sfc, (sfc.role == "channel") & (sfc.sxy == "high"))):
        d = t[sel & (t.hlt == "parity")].copy()
        if allowed is not None:
            d = d[[(m, p) in allowed for m, p in zip(d["mode"], d.point)]]
        d["neff_min"] = np.minimum(d.full_n_eff, d.fast_n_eff)
        direct = d[np.isclose(d.target_ctau_cm, d.source_ctau_cm)].set_index(["mode", "bin", "point"])
        for (src, tgt, mode), g in d.groupby(["source_ctau_cm", "target_ctau_cm", "mode"]):
            g = g.set_index(["mode", "bin", "point"])
            ref = direct.reindex(g.index)
            # cells the direct MC itself measures to the reweighted trust standard
            base = (ref.status == "ok") & (ref.neff_min >= q.min_neff_reweighted)
            gg, rr = g[base], ref[base]
            ratio = (gg.neff_min / rr.neff_min).replace([np.inf], np.nan)
            rows.append(dict(level=level, source_ctau_cm=src, target_ctau_cm=tgt, mode=mode, n_cells=len(g),
                             n_direct_ok=int(base.sum()), n_reliable=int((gg.sf_class == "reliable").sum()),
                             n_low_stats=int((gg.sf_class == "low_stats").sum()),
                             n_undefined=int((gg.sf_class == "undefined").sum()),
                             n_weight_concentrated=int((gg.status == "weight_concentrated").sum()),
                             frac_reliable=float((gg.sf_class == "reliable").mean()) if len(gg) else np.nan,
                             median_neff_ratio=float(ratio.median()) if len(ratio) else np.nan,
                             p10_neff_ratio=float(ratio.quantile(0.1)) if len(ratio) else np.nan,
                             median_full_top1=float(gg.full_top1_sumw2_frac.median()) if "full_top1_sumw2_frac" in gg else np.nan,
                             max_full_top1=float(gg.full_top1_sumw2_frac.max()) if "full_top1_sumw2_frac" in gg else np.nan))
    t = pd.DataFrame(rows)
    verdicts = []
    for r in t.itertuples():
        if np.isclose(r.target_ctau_cm, r.source_ctau_cm):
            verdicts.append("direct_reference")
            continue
        ch = t[(t.level == "channel") & (t.source_ctau_cm == r.source_ctau_cm) & (t.target_ctau_cm == r.target_ctau_cm)
               & (t["mode"] == r.mode)].frac_reliable
        bn = t[(t.level == "bin") & (t.source_ctau_cm == r.source_ctau_cm) & (t.target_ctau_cm == r.target_ctau_cm)
               & (t["mode"] == r.mode)].frac_reliable
        fc, fb = (float(ch.iloc[0]) if len(ch) else np.nan), (float(bn.iloc[0]) if len(bn) else np.nan)
        if fc >= q.target_usable_frac_channel and fb >= q.target_usable_frac_bin:
            verdicts.append("usable")
        elif fc >= q.target_limited_frac_channel:
            verdicts.append("limited")
        else:
            verdicts.append("unreliable")
    t["verdict"] = verdicts
    return t


def lifetime_checks(y: pd.DataFrame, yr, pts_frame: pd.DataFrame) -> dict:
    """Identity: the target equal to the generated lifetime must reproduce the no-reweight yields exactly."""
    d = y[(y.variant != "noreweight") & np.isclose(y.target_ctau_cm.astype(float), y.source_ctau_cm.astype(float))]
    n = y[y.variant == "noreweight"]
    k = ["mode", "bin", "kind", "point", "sim", "hlt"]
    m = d.merge(n, on=k, suffixes=("_d", "_n"))
    exact = (m.n_raw_d == m.n_raw_n) & (m.sumw_d == m.sumw_n) & (m.sumw2_d == m.sumw2_n)
    out = dict(direct_target_equals_noreweight=dict(rows=int(len(m)), bitwise_equal=int(exact.sum()),
                                                    max_rel_diff_sumw=float(((m.sumw_d - m.sumw_n).abs()
                                                                            / m.sumw_n.where(m.sumw_n != 0)).max())
                                                    if len(m) else None),
               anomaly_totals=yr.files.filter(like="n_").sum(numeric_only=True).to_dict() if len(yr.files) else {},
               source_lifetime=pts_frame[["key", "source_ctau_fullsim_cm", "source_ctau_fastsim_cm", "source_ctau_ok",
                                          "source_ctau_problem"]].to_dict("records")
               if "source_ctau_ok" in pts_frame else [])
    return out


def write(run: Path, tables: dict, extra_json: dict | None = None) -> list[Path]:
    paths = []
    for name, t in tables.items():
        if isinstance(t, pd.DataFrame):
            p = run / f"{name}.csv"
            t.to_csv(p, index=False)
        else:
            p = run / f"{name}.json"
            p.write_text(json.dumps(t, indent=1, default=str))
        paths.append(p)
    (run / "OUTPUTS.md").write_text(OUTPUTS_MD)
    for k, v in (extra_json or {}).items():
        (run / f"{k}.json").write_text(json.dumps(v, indent=1, default=str))
    return paths


def overlap_matrices(run: Path, yr, bins, benchmarks: list[str]) -> list[Path]:
    bmeta = bin_frame(bins)
    out = run / "overlap_matrices"
    out.mkdir(exist_ok=True)
    paths = []
    have = set(yr.yields.point)
    for pt in benchmarks:
        if pt not in have:
            continue
        for sim in ("fullsim", "fastsim"):
            for hlt in ("parity", "symmetric"):
                for sxy in ("high", "low"):
                    M = ov.bin_matrix(yr.overlaps, yr.yields, bmeta, pt, sim, hlt, sxy)
                    p = out / f"bins_{sxy}Sxy_{pt}_{sim}_{hlt}.csv"
                    M.to_csv(p)
                    paths.append(p)
                G = ov.group_matrix(yr.overlaps, yr.yields, bmeta, pt, sim, hlt)
                p = out / f"channels_highSxy_{pt}_{sim}_{hlt}.csv"
                G.to_csv(p)
                paths.append(p)
    return paths
