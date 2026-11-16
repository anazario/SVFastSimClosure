"""Scale factors SF = Y_FullSim / Y_FastSim,corr with statistical uncertainties.

FullSim and FastSim are independent samples, so the relative statistical
uncertainties add in quadrature.  Deliberately no intrinsic-scatter model or
Combine nuisance here: this module only measures.

Statistical conventions
* sigma(Y) = sqrt(sum w^2), N_eff = (sum w)^2 / sum w^2 (per sample).
* SF error: first-order (Gaussian) propagation.  It is only meaningful when both yields
  are non-zero and N_eff is not small; ``status`` says when it is not:
    ok                   both samples populated, N_eff >= quality.min_neff
    low_stats            N_eff < min_neff in either sample
    weight_concentrated  (reweighted targets only) one / five events carry more than
                         max_top1 / max_top5 of sum w^2 in either sample
    fullsim_empty        FullSim has no events (SF -> 0, no Gaussian error)
    fastsim_empty        FastSim has no events (SF undefined)
    both_empty
  ``sf_class`` condenses this for plotting: reliable | low_stats | undefined.  Reweighted
  points are ``reliable`` only with N_eff >= quality.min_neff_reweighted in both samples.
  Nothing is dropped: every bin stays in the tables for count-based modelling later.
* Pools are **ratios of summed yields** (``estimator = ratio_of_sums``) over a stated set
  of hypotheses: a diagnostic summary, not a fitted common SF.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LOW_NEFF = 10.0
YIELD_COLS = ["n_raw", "sumw", "sumw2"]
CONC_COLS = ["w_max", "top1_sumw2_frac", "top3_sumw2_frac", "top5_sumw2_frac"]
OPTIONAL_KEYS = ["pool", "regime", "estimator", "n_hypotheses", "variant", "source_ctau_cm", "target_ctau_cm"]


def _keys(df: pd.DataFrame) -> list[str]:
    return ["mode", "bin", "point", "hlt"] + [k for k in OPTIONAL_KEYS if k in df.columns]


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["err"] = np.sqrt(df["sumw2"])
    with np.errstate(divide="ignore", invalid="ignore"):
        df["n_eff"] = np.where(df["sumw2"] > 0, df["sumw"] ** 2 / df["sumw2"], 0.0)
    return df


def sf_table(yields: pd.DataFrame, bin_meta: pd.DataFrame, quality=None) -> pd.DataFrame:
    """One row per (mode, bin, point, hlt[, pool, variant, lifetimes]) with FullSim and FastSim side by side."""
    min_neff = getattr(quality, "min_neff", LOW_NEFF)
    y = add_derived(yields)
    keys = _keys(y)
    cols = ["n_raw", "sumw", "sumw2", "err", "n_eff"] + [c for c in CONC_COLS if c in y.columns]
    full = y[y.sim == "fullsim"].set_index(keys)[cols].add_prefix("full_")
    fast = y[y.sim == "fastsim"].set_index(keys)[cols].add_prefix("fast_")
    t = full.join(fast, how="outer").reset_index()
    with np.errstate(divide="ignore", invalid="ignore"):
        t["sf"] = np.where(t.fast_sumw > 0, t.full_sumw / t.fast_sumw, np.nan)
        rel = np.sqrt(
            np.where(t.full_sumw > 0, (t.full_err / t.full_sumw) ** 2, np.nan)
            + np.where(t.fast_sumw > 0, (t.fast_err / t.fast_sumw) ** 2, np.nan)
        )
        t["sf_err"] = t.sf * rel
    t["status"] = "ok"
    t.loc[(t.full_n_eff < min_neff) | (t.fast_n_eff < min_neff), "status"] = "low_stats"
    reweighted = _reweighted(t)
    if quality is not None and "full_top1_sumw2_frac" in t.columns:
        conc = ((t.full_top1_sumw2_frac > quality.max_top1_sumw2_frac) | (t.fast_top1_sumw2_frac > quality.max_top1_sumw2_frac))
        if "full_top5_sumw2_frac" in t.columns:
            conc |= ((t.full_top5_sumw2_frac > quality.max_top5_sumw2_frac)
                     | (t.fast_top5_sumw2_frac > quality.max_top5_sumw2_frac))
        t.loc[reweighted & conc & (t.status == "ok"), "status"] = "weight_concentrated"
    t.loc[t.full_n_raw == 0, "status"] = "fullsim_empty"
    t.loc[t.fast_n_raw == 0, "status"] = "fastsim_empty"
    t.loc[(t.full_n_raw == 0) & (t.fast_n_raw == 0), "status"] = "both_empty"
    t["sf_class"] = np.where(t.status.isin(["fullsim_empty", "fastsim_empty", "both_empty"]), "undefined",
                             np.where(t.status == "ok", "reliable", "low_stats"))
    if quality is not None:
        weak = reweighted & (np.minimum(t.full_n_eff, t.fast_n_eff) < quality.min_neff_reweighted)
        t.loc[weak & (t.sf_class == "reliable"), "sf_class"] = "low_stats"
    with np.errstate(divide="ignore", invalid="ignore"):
        t["ln_sf"] = np.where(t.sf > 0, np.log(t.sf), np.nan)
        t["ln_sf_err"] = np.where(t.sf > 0, rel, np.nan)
    t["reweighted"] = reweighted
    meta = bin_meta[[c for c in ["mode", "name", "sxy", "role", "group", "group_display", "order", "kin_label",
                                 "partner", "short"] if c in bin_meta.columns]].rename(columns={"name": "bin"})
    t = t.merge(meta, on=["mode", "bin"], how="left")
    first = [c for c in LEGACY_SF_COLS if c in t.columns]
    return t[first + [c for c in t.columns if c not in first]]


LEGACY_SF_COLS = ["mode", "bin", "point", "hlt", "full_n_raw", "full_sumw", "full_sumw2", "full_err", "full_n_eff",
                  "fast_n_raw", "fast_sumw", "fast_sumw2", "fast_err", "fast_n_eff", "sf", "sf_err", "status", "sxy",
                  "role", "group", "group_display", "order", "kin_label", "partner"]


def _reweighted(t: pd.DataFrame) -> pd.Series:
    if "variant" in t.columns and "source_ctau_cm" in t.columns:
        return (t.variant != "noreweight") & ~np.isclose(t.target_ctau_cm.astype(float), t.source_ctau_cm.astype(float))
    return pd.Series(False, index=t.index)


# ----------------------------------------------------------------------------- pooling (ratio of sums)


def pooled(yields: pd.DataFrame, points: list[str], label: str = "pooled") -> pd.DataFrame:
    """Yields summed over the given signal points: the input of a **ratio of sums**, not a fit."""
    y = yields[yields.point.isin(points)]
    keys = ["mode", "bin", "sim", "hlt"] + [k for k in ("variant", "target_ctau_cm") if k in y.columns]
    if "source_ctau_cm" in y.columns and y.source_ctau_cm.nunique(dropna=False) == 1:
        keys.append("source_ctau_cm")
    g = y.groupby(keys, as_index=False, dropna=False)[YIELD_COLS].sum()
    if "w_max" in y.columns:
        wm = y.groupby(keys, as_index=False, dropna=False)["w_max"].max()
        g = g.merge(wm, on=keys)
        with np.errstate(divide="ignore", invalid="ignore"):
            g["top1_sumw2_frac"] = np.where(g.sumw2 > 0, g.w_max ** 2 / g.sumw2, np.nan)
    g["point"] = "pooled"
    g["estimator"] = "ratio_of_sums"
    g["pool"] = label
    g["n_hypotheses"] = len(set(points) & set(y.point))
    return g


def regime_pools(yields: pd.DataFrame, points_meta: pd.DataFrame, regimes, by_source: bool = True) -> pd.DataFrame:
    """Ratio-of-sums pools restricted to each analysis mode's own signal regime, per generated lifetime.

    pool label: ``<regime>_ct<source cm>`` (or ``<regime>`` when by_source=False).  Points whose
    regime is ambiguous between the configured conventions are left out (recorded in the manifest).
    """
    out = []
    pm = points_meta[points_meta.regime_unambiguous]
    for regime, mode in regimes.mode_of.items():
        sel = pm[pm.regime == regime]
        groups = sel.groupby("ctau_cm_num") if by_source else [(None, sel)]
        for src, g in groups:
            y = yields[(yields["mode"] == mode)]
            if src is not None and "source_ctau_cm" in y.columns:
                y = y[np.isclose(y.source_ctau_cm.astype(float), src)]
            label = f"{regime}_ct{src:g}" if src is not None else regime
            p = pooled(y, list(g.key), label)
            p["regime"] = regime
            out.append(p)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# ----------------------------------------------------------------------------- channel integration


def channel_meta(bin_meta: pd.DataFrame) -> pd.DataFrame:
    """Pseudo-bin metadata for the channel-integrated selections ``<group>`` / ``<group>:SR``."""
    rows = []
    for (mode, group), g in bin_meta.groupby(["mode", "group"], sort=False):
        sxy = g.sxy.iloc[0]
        for subset, name, role in (("all", group, "channel"), ("sr", f"{group}:SR", "channel_sr")):
            if subset == "sr" and sxy != "high":
                continue
            rows.append(dict(mode=mode, name=name, sxy=sxy, role=role, group=group, group_display=g.group_display.iloc[0],
                             order=int(g.order.min()), kin_label="", partner=None,
                             short="all bins" if subset == "all" else "SR bins", subset=subset))
    m = pd.DataFrame(rows)
    hi = m[m.sxy == "high"]
    for i, r in hi.iterrows():  # partner of a high-S_xy channel: the low-S_xy channel of its bins' partners
        b = bin_meta[(bin_meta["mode"] == r["mode"]) & (bin_meta.group == r.group)]
        lo_groups = bin_meta[(bin_meta["mode"] == r["mode"]) & bin_meta.name.isin(b.partner)].group.unique()
        if len(lo_groups) == 1 and r.subset == "all":
            m.loc[i, "partner"] = lo_groups[0]
            m.loc[(m["mode"] == r["mode"]) & (m.name == lo_groups[0]), "partner"] = r["name"]
    return m


def channel_yields(yields: pd.DataFrame, bin_meta: pd.DataFrame) -> pd.DataFrame:
    """Sum of the (disjoint, verified) bins of each channel group, per point/sim/hlt/variant.

    ``<group>`` = every bin of the group (anchor + SR for high-S_xy), ``<group>:SR`` = SR bins only.
    Valid as a yield sum because bins within a group are required to be disjoint (structural
    check + event-level overlap check); then sum w^2 adds too.
    """
    y = yields[yields.get("kind", "bin") == "bin"] if "kind" in yields.columns else yields
    meta = bin_meta[["mode", "name", "group", "role"]].rename(columns={"name": "bin"})
    y = y.merge(meta, on=["mode", "bin"], how="inner")
    keys = ["mode", "group", "point", "sim", "hlt"] + [k for k in OPTIONAL_KEYS if k in y.columns]
    out = []
    for subset, sel in (("all", y), ("sr", y[y.role == "sr"])):
        g = sel.groupby(keys, as_index=False, dropna=False)[YIELD_COLS].sum()
        if "w_max" in sel.columns:
            g = g.merge(sel.groupby(keys, as_index=False, dropna=False)["w_max"].max(), on=keys)
            with np.errstate(divide="ignore", invalid="ignore"):
                g["top1_sumw2_frac"] = np.where(g.sumw2 > 0, g.w_max ** 2 / g.sumw2, np.nan)
        g["bin"] = g.group if subset == "all" else g.group + ":SR"
        out.append(g.drop(columns="group"))
    return pd.concat(out, ignore_index=True)


def channel_cross(cross: pd.DataFrame, bin_meta: pd.DataFrame) -> pd.DataFrame:
    """Cross moments sum w_i w_j summed over the disjoint bins of each channel."""
    c = cross[cross.kind == "bin"].merge(bin_meta[["mode", "name", "group", "role"]].rename(columns={"name": "bin"}),
                                         on=["mode", "bin"])
    keys = ["mode", "group", "point", "sim", "hlt", "variant_i", "variant_j", "target_i_cm", "target_j_cm"]
    out = []
    for subset, sel in (("all", c), ("sr", c[c.role == "sr"])):
        g = sel.groupby(keys, as_index=False)["sumw_ij"].sum()
        g["bin"] = g.group if subset == "all" else g.group + ":SR"
        g["kind"] = "channel"
        out.append(g.drop(columns="group"))
    return pd.concat(out, ignore_index=True)


# ----------------------------------------------------------------------------- covariance across targets


def logsf_covariance(cross: pd.DataFrame, sf: pd.DataFrame) -> pd.DataFrame:
    """Covariance of ln SF between target lifetimes of the SAME generated sample.

    Cov(Y_i, Y_j) = sum_events w_i w_j   (same events, different weights; per sample)
    Cov(ln SF_i, ln SF_j) ~= Cov_F(i,j)/(F_i F_j) + Cov_S(i,j)/(S_i S_j)   (FullSim, FastSim independent)
    First-order (Gaussian) approximation: ``gaussian_ok`` only when both points have status ok.
    """
    if cross is None or not len(cross):
        return pd.DataFrame()
    c = cross[cross.variant_i != "noreweight"]
    c = c[c.variant_j != "noreweight"]
    k = ["mode", "bin", "point", "hlt"]
    piv = c.pivot_table(index=k + ["variant_i", "variant_j", "target_i_cm", "target_j_cm"], columns="sim",
                        values="sumw_ij", aggfunc="first").reset_index()
    piv = piv.rename(columns={"fullsim": "cov_full", "fastsim": "cov_fast"})
    s = sf[sf.variant != "noreweight"][k + ["variant", "full_sumw", "fast_sumw", "full_sumw2", "fast_sumw2", "status",
                                             "source_ctau_cm"]]
    a = s.rename(columns={c_: c_ + "_i" for c_ in ["variant", "full_sumw", "fast_sumw", "full_sumw2", "fast_sumw2", "status"]})
    b = s.drop(columns="source_ctau_cm").rename(
        columns={c_: c_ + "_j" for c_ in ["variant", "full_sumw", "fast_sumw", "full_sumw2", "fast_sumw2", "status"]})
    t = piv.merge(a, on=k + ["variant_i"]).merge(b, on=k + ["variant_j"])
    with np.errstate(divide="ignore", invalid="ignore"):
        t["corr_full"] = t.cov_full / np.sqrt(t.full_sumw2_i * t.full_sumw2_j)
        t["corr_fast"] = t.cov_fast / np.sqrt(t.fast_sumw2_i * t.fast_sumw2_j)
        t["cov_ln_sf"] = t.cov_full / (t.full_sumw_i * t.full_sumw_j) + t.cov_fast / (t.fast_sumw_i * t.fast_sumw_j)
        vi = t.full_sumw2_i / t.full_sumw_i ** 2 + t.fast_sumw2_i / t.fast_sumw_i ** 2
        vj = t.full_sumw2_j / t.full_sumw_j ** 2 + t.fast_sumw2_j / t.fast_sumw_j ** 2
        t["corr_ln_sf"] = t.cov_ln_sf / np.sqrt(vi * vj)
    t["gaussian_ok"] = (t.status_i == "ok") & (t.status_j == "ok")
    cols = k + ["source_ctau_cm", "target_i_cm", "target_j_cm", "variant_i", "variant_j", "cov_full", "cov_fast",
                "corr_full", "corr_fast", "cov_ln_sf", "corr_ln_sf", "status_i", "status_j", "gaussian_ok"]
    return t[cols].sort_values(k + ["target_i_cm", "target_j_cm"]).reset_index(drop=True)


# ----------------------------------------------------------------------------- high / low S_xy


def _pair_overlap(overlaps: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Symmetric (sel_a, sel_b) overlap rows -> sums keyed by (…, high, low) in both orders."""
    if overlaps is None or not len(overlaps):
        return pd.DataFrame(columns=keys + ["sel_a", "sel_b", "n_ab", "sumw_ab", "sumw2_ab"])
    o = overlaps[(overlaps.kind_a == "bin") & (overlaps.kind_b == "bin") & (overlaps.mode_a == overlaps.mode_b)]
    o = o.rename(columns={"mode_a": "mode"})
    both = pd.concat([o, o.rename(columns={"sel_a": "sel_b", "sel_b": "sel_a", "n_a": "n_b", "n_b": "n_a"})])
    return both


def highlow(yields: pd.DataFrame, overlaps: pd.DataFrame, bin_meta: pd.DataFrame, quality=None,
            references: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per matched high/low-S_xy pair: both SFs and the double ratio DR = SF_high / SF_low.

    Var(ln DR) = sum of the four relative variances - 2 Cov(F_h,F_l)/(F_h F_l) - 2 Cov(S_h,S_l)/(S_h S_l),
    Cov(Y_h, Y_l) = sum w^2 over events in BOTH selections (zero when they are disjoint).
    ``exhaust_frac``: fraction of the S_xy-inclusive reference population (same cuts, >= 1 SV of the
    type in any S_xy class) that falls in high OR low; < 1 means high + low is not a total acceptance.
    """
    min_neff = getattr(quality, "min_neff", LOW_NEFF)
    y = add_derived(yields)
    keys = ["point", "hlt"] + [k for k in OPTIONAL_KEYS if k in y.columns]
    meta = bin_meta[(bin_meta.sxy == "high") & bin_meta.partner.notna()]
    yk = y.set_index(["mode", "bin", "sim"] + keys)
    ov = _pair_overlap(overlaps, [])  # overlaps of the same point set / pool as ``yields``
    okeys = ["mode", "sel_a", "sel_b", "sim", "point", "hlt"] + [k for k in ("variant",) if k in ov.columns]
    ov = ov.groupby(okeys, as_index=False)[["n_ab", "sumw_ab", "sumw2_ab"]].sum() if len(ov) else ov
    ovk = ov.set_index(okeys) if len(ov) else None
    rows = []
    combos = y[["sim"] + keys].drop_duplicates().drop(columns="sim").drop_duplicates()
    for _, r in meta.iterrows():
        for _, c in combos.iterrows():
            ck = tuple(c[k] for k in keys)
            rec = dict(mode=r["mode"], bin=r["name"], partner=r.partner, role=r.role, group=r.group,
                       group_display=r.get("group_display"), short=r.get("short"), order=r.order, **dict(zip(keys, ck)))
            ok = True
            for sim, tag in (("fullsim", "full"), ("fastsim", "fast")):
                for which, bn in (("h", r["name"]), ("l", r.partner)):
                    try:
                        q = yk.loc[(r["mode"], bn, sim) + ck]
                    except KeyError:
                        ok = False
                        break
                    rec[f"{tag}_{which}_n"] = int(q.n_raw)
                    rec[f"{tag}_{which}_sumw"] = float(q.sumw)
                    rec[f"{tag}_{which}_sumw2"] = float(q.sumw2)
                    rec[f"{tag}_{which}_neff"] = float(q.n_eff)
                okey = (r["mode"], r["name"], r.partner, sim, ck[0], ck[1])
                if "variant" in okeys:
                    okey += (c["variant"] if "variant" in keys else "noreweight",)
                if ovk is not None and okey in ovk.index:
                    o = ovk.loc[okey]
                    rec[f"{tag}_n_both"], rec[f"{tag}_cov_hl"] = int(o.n_ab), float(o.sumw2_ab)
                else:
                    rec[f"{tag}_n_both"], rec[f"{tag}_cov_hl"] = 0, 0.0
            if ok:
                rows.append(rec)
    t = pd.DataFrame(rows)
    if not len(t):
        return t
    with np.errstate(divide="ignore", invalid="ignore"):
        for which in ("h", "l"):
            t[f"sf_{which}"] = np.where(t[f"fast_{which}_sumw"] > 0, t[f"full_{which}_sumw"] / t[f"fast_{which}_sumw"], np.nan)
            t[f"sf_{which}_err"] = t[f"sf_{which}"] * np.sqrt(t[f"full_{which}_sumw2"] / t[f"full_{which}_sumw"] ** 2
                                                               + t[f"fast_{which}_sumw2"] / t[f"fast_{which}_sumw"] ** 2)
        t["double_ratio"] = t.sf_h / t.sf_l
        var_unc = sum(t[f"{s}_{w}_sumw2"] / t[f"{s}_{w}_sumw"] ** 2 for s in ("full", "fast") for w in ("h", "l"))
        cov = (t.full_cov_hl / (t.full_h_sumw * t.full_l_sumw) + t.fast_cov_hl / (t.fast_h_sumw * t.fast_l_sumw))
        t["ln_dr_err"] = np.sqrt(var_unc - 2 * cov)
        t["double_ratio_err"] = t.double_ratio * t.ln_dr_err
        t["double_ratio_err_uncorrelated"] = t.double_ratio * np.sqrt(var_unc)
    t["disjoint"] = (t.full_n_both == 0) & (t.fast_n_both == 0)
    nmin = t[[f"{s}_{w}_neff" for s in ("full", "fast") for w in ("h", "l")]].min(axis=1)
    t["min_neff"] = nmin
    t["status"] = np.where(~np.isfinite(t.double_ratio) | (t.double_ratio <= 0), "undefined",
                           np.where(nmin < min_neff, "low_stats", "ok"))
    if references is not None and len(references):
        ref = references[references.bin.str.startswith("sxyincl:")].copy()
        ref["hb"] = ref.bin.str.slice(len("sxyincl:"))
        rk = ["mode", "hb", "sim"] + keys
        ref = ref.set_index(rk)
        for tag, sim in (("full", "fullsim"), ("fast", "fastsim")):
            ids = [tuple([m, b, sim] + [row[k] for k in keys]) for m, b, row in
                   zip(t["mode"], t.bin, t[keys].to_dict("records"))]
            nref = np.array([ref.n_raw.get(i, np.nan) for i in ids], dtype=float)
            t[f"{tag}_n_ref"] = nref
            with np.errstate(divide="ignore", invalid="ignore"):
                t[f"{tag}_exhaust_frac"] = (t[f"{tag}_h_n"] + t[f"{tag}_l_n"] - t[f"{tag}_n_both"]) / nref
    return t


def highlow_legacy(hl_pooled: pd.DataFrame, sf_points: pd.DataFrame) -> pd.DataFrame:
    """The columns of the earlier ad-hoc ``sf_high_vs_low_pooled.csv`` (+ the new correlated quantities).

    Same definitions as before: pooled SFs (ratio of sums over all hypotheses, HLT parity), DR with
    uncorrelated error (``ratio_err`` now includes the high/low covariance; identical when the pair
    is disjoint), spread of per-point SFs excluding low_stats.
    """
    p = hl_pooled[hl_pooled.hlt == "parity"]
    rows = []
    for _, r in p.iterrows():
        d = sf_points[(sf_points["mode"] == r["mode"]) & (sf_points.bin == r.bin) & (sf_points.hlt == "parity")
                      & np.isfinite(sf_points.sf) & (sf_points.status != "low_stats")]
        chi2 = ((d.sf - r.sf_h) / d.sf_err) ** 2
        rows.append(dict(mode=r["mode"], bin=r.bin, role=r.role, sf_high=r.sf_h, sf_low=r.sf_l, ratio=r.double_ratio,
                         ratio_err=r.double_ratio_err, ratio_err_uncorrelated=r.double_ratio_err_uncorrelated,
                         n_pts=len(d), sf_min=d.sf.min(), sf_max=d.sf.max(), chi2_ndf=chi2.sum() / max(len(d) - 1, 1),
                         partner=r.partner, disjoint=r.disjoint, full_n_both=r.full_n_both, fast_n_both=r.fast_n_both,
                         full_exhaust_frac=r.get("full_exhaust_frac"), fast_exhaust_frac=r.get("fast_exhaust_frac"),
                         estimator="ratio_of_sums", pool=r.get("pool")))
    return pd.DataFrame(rows)


def pool_overlaps(overlaps: pd.DataFrame, points: list[str], label: str) -> pd.DataFrame:
    if overlaps is None or not len(overlaps):
        return overlaps
    o = overlaps[overlaps.point.isin(points)]
    keys = ["sim", "hlt", "mode_a", "sel_a", "kind_a", "mode_b", "sel_b", "kind_b"] + [
        k for k in ("variant", "target_ctau_cm") if k in o.columns]
    g = o.groupby(keys, as_index=False, dropna=False)[["n_a", "n_b", "n_ab", "sumw_ab", "sumw2_ab"]].sum()
    g["point"] = "pooled"
    g["pool"] = label
    return g


# ----------------------------------------------------------------------------- source cross-check


def source_crosscheck(sf: pd.DataFrame, quality) -> pd.DataFrame:
    """Direct MC at lifetime T vs a sample generated at another lifetime reweighted to T.

    Independent samples: z = (ln SF_direct - ln SF_rw) / sqrt(var_direct + var_rw).
    direction ``down`` (e.g. 50 -> 10 cm, a moderate reweighting) is the cross-check;
    ``up`` (10 -> 50 cm, weights grow like exp(t (1/tau0 - 1/tau))) is a stress test only.
    """
    s = sf[(sf.variant != "noreweight")].copy()
    s["mass_key"] = s.point.str.rsplit("_", n=1).str[0]
    direct = s[np.isclose(s.target_ctau_cm.astype(float), s.source_ctau_cm.astype(float))]
    rw = s[~np.isclose(s.target_ctau_cm.astype(float), s.source_ctau_cm.astype(float))]
    k = ["mode", "bin", "mass_key", "hlt", "target_ctau_cm"]
    cols = ["point", "source_ctau_cm", "sf", "sf_err", "ln_sf", "ln_sf_err", "status", "sf_class", "full_n_eff",
            "fast_n_eff"]
    m = direct[k + cols].merge(rw[k + cols], on=k, suffixes=("_direct", "_rw"))
    m["direction"] = np.where(m.source_ctau_cm_rw > m.target_ctau_cm, "down", "up")
    m["role"] = np.where(m.direction == "down", "cross_check", "stress_test")
    with np.errstate(divide="ignore", invalid="ignore"):
        m["delta_ln_sf"] = m.ln_sf_direct - m.ln_sf_rw
        m["delta_err"] = np.hypot(m.ln_sf_err_direct, m.ln_sf_err_rw)
        m["z"] = m.delta_ln_sf / m.delta_err
    m["comparable"] = (m.status_direct == "ok") & (m.sf_class_rw == "reliable")
    return m
