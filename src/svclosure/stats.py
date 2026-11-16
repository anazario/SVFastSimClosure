"""Scale factors SF = Y_FullSim / Y_FastSim,corr with statistical uncertainties.

FullSim and FastSim are independent samples, so the relative statistical
uncertainties add in quadrature.  Deliberately no intrinsic-scatter model or
Combine nuisance here: this module only measures.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LOW_NEFF = 10.0


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["err"] = np.sqrt(df["sumw2"])
    with np.errstate(divide="ignore", invalid="ignore"):
        df["n_eff"] = np.where(df["sumw2"] > 0, df["sumw"] ** 2 / df["sumw2"], 0.0)
    return df


def sf_table(yields: pd.DataFrame, bin_meta: pd.DataFrame) -> pd.DataFrame:
    """One row per (mode, bin, point, hlt) with FullSim and FastSim columns side by side."""
    y = add_derived(yields)
    keys = ["mode", "bin", "point", "hlt"]
    cols = ["n_raw", "sumw", "sumw2", "err", "n_eff"]
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
    t.loc[(t.full_n_eff < LOW_NEFF) | (t.fast_n_eff < LOW_NEFF), "status"] = "low_stats"
    t.loc[t.full_n_raw == 0, "status"] = "fullsim_empty"
    t.loc[t.fast_n_raw == 0, "status"] = "fastsim_empty"
    t.loc[(t.full_n_raw == 0) & (t.fast_n_raw == 0), "status"] = "both_empty"
    meta = bin_meta[["mode", "name", "sxy", "role", "group", "group_display", "order", "kin_label", "partner"]].rename(
        columns={"name": "bin"}
    )
    return t.merge(meta, on=["mode", "bin"], how="left")


def pooled(yields: pd.DataFrame, points: list[str]) -> pd.DataFrame:
    """Yields summed over the given signal points (a ratio of sums, not a fit)."""
    y = yields[yields.point.isin(points)]
    g = y.groupby(["mode", "bin", "sim", "hlt"], as_index=False)[["n_raw", "sumw", "sumw2"]].sum()
    g["point"] = "pooled"
    return g
