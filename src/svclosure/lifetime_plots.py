"""Lifetime-reweighting figures (Matplotlib + mplhep, same conventions as plotting.py).

1. SF vs target c*tau per analysis mode and channel (regime ratio-of-sums pools), one curve per
   generated sample; the directly generated points are stars.
2. The same for representative mass hypotheses.
3. Heatmaps: hypothesis x target lifetime, one panel per channel, one figure per mode.
4. Reweighting validity: N_eff(target)/N_eff(direct) and the largest single-event share of sum w^2.
5. Direct MC vs the other generated sample reweighted to the same lifetime (cross-check / stress test).
Points from the same generated sample are correlated across targets (lifetime_cov_logsf.csv): the
curves are drawn as connected points to make that visible, and no figure treats them as independent.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, NullFormatter

from .plotting import (MODE_ORDER, MODE_TITLE, REGIME_PUBLIC, SF_LABEL, _fig_cms, _ordered, _save, _style, _unity_band,
                       auto_ylim, class_legend_handles, draw_measurements, plt)

SOURCE_STYLE = {10.0: dict(color="#3f90da", marker="o"), 50.0: dict(color="#bd1f01", marker="D")}
DIRECT_STYLE = dict(marker="*", ms=19, mec="black", mew=1.2, zorder=6)
NOTE_CORR = "points of one generated sample are correlated (same events); see lifetime_cov_logsf.csv"


def _src_style(src: float) -> dict:
    return SOURCE_STYLE.get(float(src), dict(color="#832db6", marker="s"))


def _logx(ax, targets):
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    lo, hi = min(targets) / 1.35, max(targets) * 1.35
    ax.set_xticks([v for v in (1, 2, 5, 10, 20, 50, 100, 200, 500) if lo <= v <= hi])
    ax.set_xlim(lo, hi)


def _curves(ax, g: pd.DataFrame, ylim, offsets: bool = True) -> dict:
    """One connected series per generated sample; direct point as a star."""
    counts = dict(clipped=0, undefined=0)
    srcs = sorted(g.source_ctau_cm.unique())
    for k, src in enumerate(srcs):
        st = _src_style(src)
        s = g[np.isclose(g.source_ctau_cm, src)].sort_values("target_ctau_cm")
        x = s.target_ctau_cm.to_numpy() * ((1 + 0.025 * (k - (len(srcs) - 1) / 2)) if offsets else 1)
        fin = np.isfinite(s.sf.to_numpy()) & (s.sf_class != "undefined").to_numpy()
        ax.plot(x[fin], np.clip(s.sf.to_numpy()[fin], *ylim), color=st["color"], lw=1, alpha=0.6, zorder=2)
        rw = ~np.isclose(s.target_ctau_cm, s.source_ctau_cm)
        c = draw_measurements(ax, x[rw], s.sf[rw], s.sf_err[rw], s.sf_class[rw], s.status[rw], st["color"], st["marker"],
                              ylim, ms=8)
        counts = {kk: counts[kk] + c[kk] for kk in counts}
        dct = ~rw
        if dct.any():
            ax.errorbar(x[dct], s.sf[dct], yerr=s.sf_err[dct], fmt="none", ecolor=st["color"], elinewidth=2.2, zorder=5)
            ax.plot(x[dct], s.sf[dct], color=st["color"], ls="none", **DIRECT_STYLE)
    return counts


def _legend_sources(srcs) -> list:
    h = []
    for src in srcs:
        st = _src_style(src)
        h.append(Line2D([], [], color=st["color"], marker=st["marker"], lw=1, label=f"reweighted from {src:g} cm sample"))
        h.append(Line2D([], [], color=st["color"], ls="none", label=f"direct MC, generated {src:g} cm",
                        **{k: v for k, v in DIRECT_STYLE.items() if k != "zorder"}))
    return h


def sf_vs_ctau_grid(t: pd.DataFrame, out: Path, stem: str, rlabel: str, title: str, q, hlt: str = "parity",
                    modes=MODE_ORDER) -> list[Path]:
    """Rows = analysis modes, columns = channels (high-S_xy, all bins of the channel)."""
    _style()
    d = t[(t.hlt == hlt) & (t.role == "channel") & (t.sxy == "high")]
    if not len(d):
        return []
    rows = [m for m in modes if m in set(d["mode"])]
    chans = {m: list(_ordered(d[d["mode"] == m]).drop_duplicates("bin")[["bin", "group_display"]].itertuples(index=False))
             for m in rows}
    ncol = max(len(v) for v in chans.values())
    fig, axes = plt.subplots(len(rows), ncol, figsize=(6.2 * ncol, 5.4 * len(rows)), squeeze=False)
    fig.subplots_adjust(hspace=0.42, wspace=0.25, top=0.84, bottom=0.14)
    targets = sorted(d.target_ctau_cm.unique())
    counts = dict(clipped=0, undefined=0)
    for i, m in enumerate(rows):
        for j in range(ncol):
            ax = axes[i, j]
            if j >= len(chans[m]):
                ax.axis("off")
                continue
            ch, disp = chans[m][j]
            g = d[(d["mode"] == m) & (d.bin == ch)]
            ylim = auto_ylim(g.sf, g.sf_err, g.sf_class)
            ylim = (ylim[0], ylim[1] + 0.12 * (ylim[1] - ylim[0]))
            ax.set_ylim(*ylim)
            _unity_band(ax)
            c = _curves(ax, g, ylim)
            counts = {kk: counts[kk] + c[kk] for kk in counts}
            _logx(ax, targets)
            ax.set_title(f"{MODE_TITLE[m]}: {disp}", fontsize=15)
            ax.set_xlabel(r"target c$\tau$ [cm]", fontsize=14)
            ax.set_ylabel("SF", fontsize=15)
    _fig_cms(fig, rlabel, y=0.93)
    fig.text(0.125, 0.905, title + f"\n{NOTE_CORR}; open: low statistics or reweighted with "
             rf"$N_{{\mathrm{{eff}}}}<{q.min_neff_reweighted:g}$", fontsize=12, ha="left", va="top")
    fig.legend(handles=_legend_sources(sorted(d.source_ctau_cm.unique())) + class_legend_handles(counts)[1:],
               loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False, fontsize=12)
    return _save(fig, out, stem)


def heatmap_lifetime(t: pd.DataFrame, pts: pd.DataFrame, out: Path, stem: str, rlabel: str, mode: str, regime: str,
                     hlt: str = "parity") -> list[Path]:
    """Hypotheses of the mode's regime (rows, by Delta m) x target lifetime (columns, grouped by source),
    one panel per channel.  Direct-MC columns are outlined."""
    _style()
    keep = set(pts[(pts.regime == regime) & pts.regime_unambiguous].mass_key)
    d = t[(t.hlt == hlt) & (t["mode"] == mode) & (t.role == "channel") & (t.sxy == "high")].copy()
    d["mass_key"] = d.point.str.rsplit("_", n=1).str[0]
    d = d[d.mass_key.isin(keep)]
    if not len(d):
        return []
    pm = pts.drop_duplicates("mass_key").set_index("mass_key").loc[sorted(keep)].sort_values(["dm", "mgl", "mn2"])
    rows = [k for k in pm.index if k in set(d.mass_key)]
    cols = d[["source_ctau_cm", "target_ctau_cm"]].drop_duplicates().sort_values(["source_ctau_cm", "target_ctau_cm"])
    cols = list(cols.itertuples(index=False, name=None))
    chans = list(_ordered(d).drop_duplicates("bin")[["bin", "group_display"]].itertuples(index=False))
    fig, axes = plt.subplots(1, len(chans), figsize=(4.6 * len(chans) + 2.5, 0.55 * len(rows) + 4.2), squeeze=False)
    fig.subplots_adjust(wspace=0.08, top=0.8, bottom=0.2)
    fin = d[(d.sf_class == "reliable") & (d.sf > 0)].sf
    r_ = float(np.nanmax(np.abs(np.log(fin)))) if len(fin) else np.log(2)
    r_ = min(max(r_, np.log(1.5)), np.log(10))
    norm = LogNorm(vmin=np.exp(-r_), vmax=np.exp(r_))
    im = None
    for ax, (ch, disp) in zip(axes[0], chans):
        g = d[d.bin == ch].set_index(["mass_key", "source_ctau_cm", "target_ctau_cm"])
        grid = np.full((len(rows), len(cols)), np.nan)
        kl = np.full(grid.shape, "", dtype=object)
        for i, mk in enumerate(rows):
            for j, (s, tg) in enumerate(cols):
                if (mk, s, tg) in g.index:
                    r = g.loc[(mk, s, tg)]
                    grid[i, j], kl[i, j] = r.sf, r.sf_class
        ax.set_facecolor("0.85")
        im = ax.imshow(np.where((grid > 0) & (kl != "undefined"), grid, np.nan), cmap="RdBu_r", norm=norm, aspect="auto",
                       interpolation="nearest")
        for i in range(len(rows)):
            for j, (s, tg) in enumerate(cols):
                v, k = grid[i, j], kl[i, j]
                if k == "low_stats":
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, hatch="///", ec="0.35", lw=0))
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, style="italic", color="0.2")
                elif k == "reliable":
                    dark = abs(np.log(max(v, 1e-9))) > 0.6 * r_
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8.5, color="white" if dark else "black")
                else:
                    ax.text(j, i, "–", ha="center", va="center", fontsize=10, color="0.3")
                if np.isclose(s, tg):
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="black", lw=2))
        for j in range(1, len(cols)):
            if cols[j][0] != cols[j - 1][0]:
                ax.axvline(j - 0.5, color="black", lw=2.5)
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels([f"{tg:g}" for _, tg in cols], fontsize=11)
        ax.set_xlabel(r"target c$\tau$ [cm]", fontsize=13)
        for s in dict.fromkeys(c[0] for c in cols):
            js = [j for j, c in enumerate(cols) if c[0] == s]
            ax.text(np.mean(js), -0.9, f"from {s:g} cm", ha="center", va="bottom", fontsize=11)
        ax.set_title(disp, fontsize=15, pad=34, fontweight="bold")
        ax.tick_params(which="both", length=0)
        ax.set_yticks(range(len(rows)))
        if ax is axes[0, 0]:
            ax.set_yticklabels([f"({pm.loc[k, 'mgl']}, {pm.loc[k, 'mn2']}, {pm.loc[k, 'mn1']})" for k in rows], fontsize=11)
            ax.set_ylabel(r"$(m_{\tilde{g}}, m_{\tilde{\chi}_2^0}, m_{\tilde{\chi}_1^0})$ [GeV]", fontsize=13)
        else:
            ax.set_yticklabels([])
    cb = fig.colorbar(im, ax=list(axes[0]), pad=0.01, fraction=0.03)
    cb.set_label(SF_LABEL + " (channel)", fontsize=13)
    ticks = [v for v in (0.2, 0.33, 0.5, 0.67, 0.8, 1, 1.25, 1.5, 2, 3, 5) if norm.vmin <= v <= norm.vmax]
    cb.set_ticks(ticks)
    cb.set_ticklabels([f"{v:g}" for v in ticks])
    cb.minorticks_off()
    _fig_cms(fig, rlabel, y=0.93, x1=0.86)
    fig.text(0.125, 0.885, f"{MODE_TITLE[mode]} analysis, {REGIME_PUBLIC[regime]} hypotheses; outlined: direct MC; "
             "hatched: low statistics / concentrated weights; '–': undefined", fontsize=11, ha="left")
    return _save(fig, out, stem)


def validity_plot(sfl: pd.DataFrame, pts: pd.DataFrame, regimes, q, out: Path, stem: str, rlabel: str,
                  level_label: str = "channel") -> list[Path]:
    """N_eff(target)/N_eff(direct) and the largest single-event share of sum w^2, vs target lifetime.

    Regime-matched hypotheses only; median over hypotheses x channels with the 10-90% range.
    """
    _style()
    d = sfl[(sfl.hlt == "parity") & (sfl.role == "channel") & (sfl.sxy == "high")].copy()
    allowed = {(regimes.mode_of[r], k) for r, k in zip(pts.regime, pts.key) if r in regimes.mode_of}
    d = d[[(m, p) in allowed for m, p in zip(d["mode"], d.point)]]
    if not len(d):
        return []
    d["neff_min"] = np.minimum(d.full_n_eff, d.fast_n_eff)
    ref = d[np.isclose(d.target_ctau_cm, d.source_ctau_cm)].set_index(["mode", "bin", "point"]).neff_min
    d["neff_ratio"] = d.neff_min.to_numpy() / ref.reindex(pd.MultiIndex.from_frame(d[["mode", "bin", "point"]])).to_numpy()
    d["top1"] = np.maximum(d.full_top1_sumw2_frac, d.fast_top1_sumw2_frac)
    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(10, 14), sharex=True)
    fig.subplots_adjust(hspace=0.08, top=0.9)
    targets = sorted(d.target_ctau_cm.unique())
    for (src, mode), g in d.groupby(["source_ctau_cm", "mode"]):
        st = _src_style(src)
        ls = "-" if mode == "compressed" else "--"
        agg = g.groupby("target_ctau_cm").agg(r_med=("neff_ratio", "median"), r_lo=("neff_ratio", lambda v: v.quantile(.1)),
                                              r_hi=("neff_ratio", lambda v: v.quantile(.9)), n_med=("neff_min", "median"),
                                              t_med=("top1", "median"), t_max=("top1", "max"))
        lab = f"from {src:g} cm, {MODE_TITLE[mode]}"
        a1.plot(agg.index, agg.r_med, ls=ls, marker=st["marker"], color=st["color"], label=lab)
        a1.fill_between(agg.index, agg.r_lo, agg.r_hi, color=st["color"], alpha=0.12, lw=0)
        a2.plot(agg.index, agg.n_med, ls=ls, marker=st["marker"], color=st["color"])
        a3.plot(agg.index, agg.t_med, ls=ls, marker=st["marker"], color=st["color"])
        a3.plot(agg.index, agg.t_max, ls=":", marker=st["marker"], mfc="white", color=st["color"])
    a1.axhline(1, color="0.4", lw=1)
    a1.set_ylabel(r"$N_{\mathrm{eff}}$(target) / $N_{\mathrm{eff}}$(direct)", fontsize=16)
    a1.legend(fontsize=11, frameon=False, ncol=2, loc="lower center")
    a2.axhline(q.min_neff_reweighted, color="0.3", ls="-.", lw=1.2, label=f"trust threshold ({q.min_neff_reweighted:g})")
    a2.axhline(q.min_neff, color="0.5", ls=":", lw=1.2, label=f"low-stats threshold ({q.min_neff:g})")
    a2.set_yscale("log")
    a2.set_ylabel(r"median min($N_{\mathrm{eff}}$ Full, Fast)", fontsize=16)
    a2.legend(fontsize=11, frameon=False)
    a3.axhline(q.max_top1_sumw2_frac, color="0.3", ls="-.", lw=1.2, label=f"concentration threshold ({q.max_top1_sumw2_frac:g})")
    a3.set_ylabel(r"largest $w^2$ / $\sum w^2$", fontsize=16)
    a3.legend(handles=[Line2D([], [], color="0.3", label="median"), Line2D([], [], color="0.3", ls=":", mfc="white",
                                                                              marker="o", label="maximum"),
                       Line2D([], [], color="0.3", ls="-.", label=f"threshold ({q.max_top1_sumw2_frac:g})")],
               fontsize=11, frameon=False)
    a3.set_ylim(0, None)
    _logx(a3, targets)
    a3.set_xlabel(r"target c$\tau$ [cm]")
    _fig_cms(fig, rlabel, y=0.925, x0=0.125, x1=0.9)
    fig.text(0.125, 0.905, f"{level_label}-integrated high-$S_{{xy}}$ yields, regime-matched hypotheses "
             "(band: 10-90% of hypotheses x channels)", fontsize=12)
    return _save(fig, out, stem)


def crosscheck_plot(cc: pd.DataFrame, out: Path, stem: str, rlabel: str, role: str) -> list[Path]:
    """ln SF(direct) vs ln SF(reweighted other sample) and the pull distribution."""
    _style()
    d = cc[(cc.role == role) & (cc.hlt == "parity")]
    if not len(d):
        return []
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(17, 7.5), gridspec_kw=dict(width_ratios=[1.25, 1]))
    fig.subplots_adjust(top=0.84, wspace=0.25)
    comp = d[d.comparable]
    groups = list(dict.fromkeys(d.group_display_)) if "group_display_" in d else []
    pal = ["#3f90da", "#ffa90e", "#bd1f01", "#94a4a2", "#832db6", "#a96b59"]
    keyc = (d["mode"] + ":" + d.bin)
    cmap = {k: pal[i % len(pal)] for i, k in enumerate(dict.fromkeys(keyc))}
    for k, g in comp.groupby(comp["mode"] + ":" + comp.bin):
        a1.errorbar(g.sf_direct, g.sf_rw, xerr=g.sf_err_direct, yerr=g.sf_err_rw, fmt="o", ms=6, color=cmap[k],
                    label=_chan_label(k), alpha=0.85)
    nc = d[~d.comparable & np.isfinite(d.sf_direct) & np.isfinite(d.sf_rw)]
    a1.plot(nc.sf_direct, nc.sf_rw, "o", mfc="white", mec="0.5", ms=5, label="not comparable (low statistics)")
    lim = [0.8 * np.nanmin(np.r_[comp.sf_direct, comp.sf_rw, 0.8]), 1.15 * np.nanmax(np.r_[comp.sf_direct, comp.sf_rw, 1.2])]
    a1.plot(lim, lim, color="0.3", lw=1)
    a1.set_xlim(lim)
    a1.set_ylim(lim)
    tgt = d.target_ctau_cm.iloc[0]
    src_rw = d.source_ctau_cm_rw.iloc[0]
    a1.set_xlabel(f"SF, direct MC at {tgt:g} cm")
    a1.set_ylabel(f"SF, {src_rw:g} cm sample reweighted to {tgt:g} cm")
    a1.legend(fontsize=9.5, frameon=False, loc="upper left")
    z = comp.z[np.isfinite(comp.z)]
    a2.hist(z, bins=np.linspace(-5, 5, 21), color="#3f90da", alpha=0.7, label=f"{len(z)} comparable measurements")
    xs = np.linspace(-5, 5, 200)
    a2.plot(xs, len(z) * 0.5 * np.exp(-xs ** 2 / 2) / np.sqrt(2 * np.pi), color="0.2", label="unit Gaussian")
    a2.set_xlabel(r"$z = (\ln\mathrm{SF}_{\mathrm{direct}} - \ln\mathrm{SF}_{\mathrm{rw}})/\sigma$")
    a2.set_ylabel("measurements")
    if len(z):
        a2.text(0.03, 0.97, f"mean {z.mean():+.2f}\nRMS {np.sqrt((z ** 2).mean()):.2f}\n|z|>3: {(z.abs() > 3).sum()}",
                transform=a2.transAxes, va="top", fontsize=12)
    a2.legend(fontsize=11, frameon=False, loc="upper right")
    _fig_cms(fig, rlabel, y=0.93)
    what = "cross-check" if role == "cross_check" else "STRESS TEST (extrapolation to longer lifetime; not a nominal estimate)"
    fig.text(0.125, 0.885, f"{what}: independent samples, same mass hypothesis; "
             "z ignores correlations between bins of one hypothesis", fontsize=12)
    return _save(fig, out, stem)


def _chan_label(k: str) -> str:
    from .bins import GROUP_DISPLAY

    mode, b = k.split(":", 1)
    g = b.split(":")[0]
    return f"{MODE_TITLE[mode]}: {GROUP_DISPLAY.get(g, g)}" + (" (SR)" if b.endswith(":SR") else "")


def all_figures(T: dict, pts: pd.DataFrame, regimes, scan, out: Path, rlabel: str) -> list[Path]:
    made = []
    q = scan.quality

    def nominal(t):  # cross-check / stress-test targets only enter the dedicated comparison figures
        return t[t.target_role.isin(["direct", "nominal"])] if "target_role" in t.columns else t

    T = dict(T, lifetime_sf_channel_pooled_regime=nominal(T["lifetime_sf_channel_pooled_regime"]),
             lifetime_sf_channel_all=T["lifetime_sf_channel"], lifetime_sf_channel=nominal(T["lifetime_sf_channel"]))
    reg = T["lifetime_sf_channel_pooled_regime"]
    parts = [reg[(reg["mode"] == m) & reg.pool.str.startswith(r + "_ct")] for r, m in regimes.mode_of.items()]
    pooled = pd.concat(parts)
    for hlt in ("parity", "symmetric"):
        sfx = "" if hlt == "parity" else "_hltSymmetric"
        made += sf_vs_ctau_grid(pooled, out, f"sf_vs_ctau_channels{sfx}", rlabel,
                                "Channel-integrated SF, ratio of summed yields over each mode's own signal regime"
                                + ("" if hlt == "parity" else " [HLT dropped in both]"), q, hlt)
    sfc = T["lifetime_sf_channel"].copy()
    sfc["mass_key"] = sfc.point.str.rsplit("_", n=1).str[0]
    for mk in scan.representative_points:
        g = sfc[sfc.mass_key == mk]
        if not len(g):
            continue
        r = pts[pts.mass_key == mk].iloc[0]
        regime = r.regime
        made += sf_vs_ctau_grid(g, out / "per_hypothesis", f"sf_vs_ctau_{mk}", rlabel,
                                f"gogoGZ ({r.mgl}, {r.mn2}, {r.mn1}) GeV, $\\Delta m$ = {r.dm} GeV "
                                f"({REGIME_PUBLIC[regime]}); its own regime's analysis mode is the nominal one", q)
    for regime, mode in regimes.mode_of.items():
        made += heatmap_lifetime(sfc, pts, out, f"sf_heatmap_ctau_{mode}", rlabel, mode, regime)
    made += validity_plot(sfc, pts, regimes, q, out, "reweight_validity_channels", rlabel)
    cc = T["lifetime_source_crosscheck_channel"]
    if len(cc):
        high = set(T["lifetime_sf_channel_all"].query("sxy == 'high' and role == 'channel'").bin)
        cc = cc[cc.bin.isin(high)]  # whole high-S_xy channels only (":SR" subsets share their events)
        for role in ("cross_check", "stress_test"):
            for tgt, g in cc[cc.role == role].groupby("target_ctau_cm"):
                made += crosscheck_plot(g, out, f"direct_vs_reweighted_{role}_{tgt:g}cm", rlabel, role)
    return made
