"""Matplotlib/mplhep figures.  No ROOT import anywhere in this module.

Conventions borrowed from LLPStandardPlots (style.py / fit_plotter.py / unrolled.py),
re-implemented in matplotlib:
  * CMS label in bold sans + "Simulation Preliminary" italic, energy label right-aligned,
  * channel-group separators: dashed gray (fit_plotter sub_sep_bins style 2),
  * analysis-mode separator: solid black, width 3 (fit_plotter separator_bins in *_final schemes),
  * group labels drawn above each group; the clean ``*_noBinNames`` figures label each bin by its
    kinematic definition only (full analysis-bin-name figures are a debugging option),
  * gray +-10% band around SF = 1: a *reference tolerance*, not a fitted interval.
Bin labels and kinematic ranges come from the resolved-bin manifest (LLPCombine YAML), never
from LLPStandardPlots' unrolled binning tables.

Measurement classes (``sf_class`` from stats.sf_table) are always drawn differently:
  reliable   filled marker, solid error bar
  low_stats  open marker, dashed error bar (first-order Gaussian error not trustworthy)
  undefined  no ratio: hollow down-triangle at the lower edge (FullSim empty, SF -> 0),
             hollow up-triangle at the upper edge (FastSim empty), x (both empty)
Finite values outside the automatic y range are drawn as filled arrows at the edge, with a note.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mplhep as hep  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

# CMS-recommended categorical colours (Petroff 10-colour scheme)
SR_COLOR = "#3f90da"
ANCHOR_COLOR = "#e76300"
PETROFF = ["#3f90da", "#ffa90e", "#bd1f01", "#94a4a2", "#832db6", "#a96b59", "#e76300", "#b9ac70", "#717581", "#92dadd"]
MODE_ORDER = ("compressed", "noncompressed")
MODE_TITLE = {"compressed": "Compressed", "noncompressed": r"High-$\Delta m$"}  # public names
REGIME_PUBLIC = {"compressed": "compressed", "high_dm": r"high-$\Delta m$"}
BAND_FRAC = 0.10  # reference tolerance drawn around SF = 1
SF_LABEL = r"SF = $Y_{\mathrm{FullSim}}\,/\,Y_{\mathrm{FastSim,corr}}$"


def _style():
    plt.style.use(hep.style.CMS)
    plt.rcParams.update({"axes.labelsize": 20, "legend.fontsize": 13})


def _cms(ax, rlabel: str, loc: int = 0, fontsize: int = 20):
    hep.cms.label("Preliminary", data=False, rlabel=rlabel, ax=ax, loc=loc, fontsize=fontsize)


def _fig_cms(fig, rlabel: str, y: float = 0.955, x0: float = 0.125, x1: float = 0.9):
    fig.text(x0, y, "CMS", fontsize=24, fontweight="bold", ha="left", va="bottom")
    fig.text(x0 + 0.95 / fig.get_size_inches()[0], y, "Simulation Preliminary", fontsize=20, style="italic", ha="left",
             va="bottom")
    fig.text(x1, y, rlabel, fontsize=20, ha="right", va="bottom")


def _unity_band(ax, label: bool = False):
    ax.axhspan(1 - BAND_FRAC, 1 + BAND_FRAC, color="0.6", alpha=0.25, lw=0, zorder=0,
               label=(rf"$\pm{BAND_FRAC * 100:.0f}\%$ reference tolerance (not a fit)") if label else None)
    ax.axhline(1.0, color="0.35", lw=1.2, ls="-", zorder=1)


def _ordered(t: pd.DataFrame) -> pd.DataFrame:
    t = t.copy()
    t["_m"] = t["mode"].map({m: i for i, m in enumerate(MODE_ORDER)})
    return t.sort_values(["_m", "order"]).drop(columns="_m").reset_index(drop=True)


def _save(fig, out: Path, stem: str) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        p = out / f"{stem}.{ext}"
        fig.savefig(p, dpi=200 if ext == "png" else None, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    return paths


# ----------------------------------------------------------------------------- shared measurement drawing


def auto_ylim(sf, err, cls, lo_floor: float = 0.0, pad: float = 0.15, min_half_span: float = 0.2,
              max_factor: float = 2.0) -> tuple[float, float]:
    """y range from the reliable points (+-1 sigma) and the low-statistics central values.

    Low-statistics central values extend the range only up to ``max_factor`` x the reliable span
    around 1 (wild values are then drawn as clipped arrows, never silently dropped).
    """
    sf, err, cls = np.asarray(sf, float), np.asarray(err, float), np.asarray(cls)
    rel = (cls == "reliable") & np.isfinite(sf)
    e = np.where(np.isfinite(err), err, 0)
    if rel.any():
        lo, hi = np.min(sf[rel] - e[rel]), np.max(sf[rel] + e[rel])
    else:
        lo, hi = 1 - min_half_span, 1 + min_half_span
    lo, hi = min(lo, 1 - BAND_FRAC), max(hi, 1 + BAND_FRAC)
    span = max(hi - 1, 1 - lo, min_half_span)
    low = (cls == "low_stats") & np.isfinite(sf)
    if low.any():
        lo = min(lo, max(np.min(sf[low]), 1 - max_factor * span))
        hi = max(hi, min(np.max(sf[low]), 1 + max_factor * span))
    width = hi - lo
    return max(lo_floor, lo - pad * width), hi + pad * width


def draw_measurements(ax, x, sf, err, cls, status, color, marker, ylim, ms=9, zorder=3, label=None) -> dict:
    """Draw one series with the class conventions; returns counts of clipped / undefined points."""
    x, sf, err = np.asarray(x, float), np.asarray(sf, float), np.asarray(err, float)
    cls, status = np.asarray(cls), np.asarray(status)
    lo, hi = ylim
    edge_lo, edge_hi = lo + 0.025 * (hi - lo), hi - 0.025 * (hi - lo)
    n = dict(clipped=0, undefined=0)
    finite = np.isfinite(sf) & (cls != "undefined")
    inside = finite & (sf >= lo) & (sf <= hi)
    for c, mfc, ls in (("reliable", color, "-"), ("low_stats", "white", "--")):
        m = inside & (cls == c)
        if m.any():
            eb = ax.errorbar(x[m], sf[m], yerr=np.where(np.isfinite(err[m]), err[m], 0), fmt=marker, ms=ms, color=color,
                             mfc=mfc, mec=color, mew=1.6, elinewidth=2 if c == "reliable" else 1.4, capsize=0,
                             zorder=zorder, label=label if c == "reliable" else None)
            if ls != "-":
                for coll in eb[2]:
                    coll.set_linestyle(ls)
    if label and not (inside & (cls == "reliable")).any():  # keep the series in the legend
        ax.plot([], [], marker=marker, ms=ms, color=color, ls="none", label=label)
    for side, m0, y, mk in (("lo", finite & (sf < lo), edge_lo, "v"), ("hi", finite & (sf > hi), edge_hi, "^")):
        for c, mfc in (("reliable", color), ("low_stats", "white")):  # clipped arrows keep the class convention
            m = m0 & (cls == c)
            if m.any():
                ax.plot(x[m], np.full(m.sum(), y), marker=mk, ms=ms + 1, color=color, mfc=mfc, mec=color, mew=1.6,
                        ls="none", zorder=zorder + 1)
                n["clipped"] += int(m.sum())
    und = cls == "undefined"
    for st, y, mk in (("fullsim_empty", edge_lo, "v"), ("fastsim_empty", edge_hi, "^"), ("both_empty", edge_lo, "x")):
        m = und & (status == st)
        if m.any():
            ax.plot(x[m], np.full(m.sum(), y), marker=mk, ms=ms, mfc="none", mec="black" if mk != "x" else None,
                    color="black", mew=1.4, ls="none", zorder=zorder + 1)
            n["undefined"] += int(m.sum())
    return n


def class_legend_handles(counts: dict, min_neff: float | None = None, reweighted_neff: float | None = None) -> list:
    h = [Line2D([], [], marker="o", color="0.3", mfc="0.3", ls="-", lw=2, label="reliable (stat. error)"),
         Line2D([], [], marker="o", color="0.3", mfc="white", ls="--", lw=1.4,
                label="low statistics" + (rf" ($N_{{\mathrm{{eff}}}}<{min_neff:g}$)" if min_neff else "")
                + " or concentrated weights")]
    if reweighted_neff:
        h[-1].set_label(h[-1].get_label() + rf"; reweighted: $N_{{\mathrm{{eff}}}}<{reweighted_neff:g}$")
    if counts.get("undefined"):
        h += [Line2D([], [], marker="v", color="black", mfc="none", ls="none", label="FullSim empty (SF$\\to$0)"),
              Line2D([], [], marker="^", color="black", mfc="none", ls="none", label="FastSim empty (undefined)")]
    if counts.get("clipped"):
        h.append(Line2D([], [], marker="^", color="0.3", ls="none", label="value outside y range"))
    return h


# ----------------------------------------------------------------------------- primary SF plot


def sf_plot(t: pd.DataFrame, out: Path, stem: str, title: str, rlabel: str, hlt: str = "parity",
            bin_names: bool = True, notes: list[str] | None = None, min_neff: float | None = 10) -> list[Path]:
    """Primary SF plot: the high-S_xy bins of both analysis modes.

    bin_names=False labels each x tick with its kinematic definition only (no analysis bin name).
    """
    _style()
    d = _ordered(t[(t.sxy == "high") & (t.hlt == hlt)])
    n = len(d)
    x = np.arange(n, dtype=float)
    fig, ax = plt.subplots(figsize=(max(14, 0.9 * n + 4), 8.5))
    _unity_band(ax, label=True)
    cls = d.sf_class if "sf_class" in d else np.where(d.status == "ok", "reliable", "low_stats")
    ylo, yhi = auto_ylim(d.sf, d.sf_err, cls)
    span = yhi - ylo
    ymax = yhi + 0.55 * span  # headroom for group / mode labels and the legend
    counts = dict(clipped=0, undefined=0)
    for role, color, marker, lab in (
        ("sr", SR_COLOR, "o", r"High-$S_{xy}$ SR bin"),
        ("anchor", ANCHOR_COLOR, "s", r"High-$S_{xy}$ normalization-anchor CR bin"),
    ):
        m = (d.role == role).to_numpy()
        c = draw_measurements(ax, x[m], d.sf[m], d.sf_err[m], np.asarray(cls)[m], d.status[m], color, marker,
                              (ylo, ymax), label=lab)
        counts = {k: counts[k] + c[k] for k in counts}
    ax.set_ylim(ylo, ymax)
    ax.set_xlim(-0.6, n - 0.4)

    prev_mode, prev_group, gstart, prev_glab = None, None, 0, None
    ytxt = yhi + 0.06 * span

    def label_group(s, e, text):
        ax.text((s + e - 1) / 2, ytxt, text, ha="center", va="bottom", fontsize=15)

    for i, r in d.iterrows():
        if prev_mode is not None and r["mode"] != prev_mode:
            label_group(gstart, i, prev_glab)
            ax.axvline(i - 0.5, color="black", lw=3, zorder=2)
            gstart = i
        elif prev_group is not None and r["group"] != prev_group:
            label_group(gstart, i, prev_glab)
            ax.axvline(i - 0.5, color="0.45", lw=1.6, ls="--", zorder=2)
            gstart = i
        prev_mode, prev_group, prev_glab = r["mode"], r["group"], r["group_display"]
    if n:
        label_group(gstart, n, prev_glab)
    for mode in MODE_ORDER:
        idx = np.flatnonzero(d["mode"].to_numpy() == mode)
        if len(idx):
            ax.text((idx[0] + idx[-1]) / 2, yhi + 0.17 * span, MODE_TITLE[mode], ha="center", va="bottom",
                    fontsize=17, fontweight="bold")
    ax.set_xticks(x)
    kin = [k.replace("; ", "\n") for k in d.kin_label]
    ticks = [f"{b}\n{k}" for b, k in zip(d.bin, kin)] if bin_names else kin
    ax.set_xticklabels(ticks, rotation=90, fontsize=11 if bin_names else 13, linespacing=1.25)
    ax.tick_params(axis="x", which="minor", bottom=False, top=False)
    ax.set_ylabel(SF_LABEL)
    h, l = ax.get_legend_handles_labels()
    h += class_legend_handles(counts, min_neff)
    ax.legend(handles=h, loc="upper left", bbox_to_anchor=(0.0, 1.0), frameon=True, framealpha=0.9, edgecolor="none",
              ncol=2, fontsize=12)
    note = [title + ("" if hlt == "parity" else "  [HLT dropped in both samples]")] + list(notes or [])
    if counts["clipped"]:
        note.append(f"{counts['clipped']} value(s) outside the y range drawn as arrows at the edge")
    ax.text(0.99, 0.02, "\n".join(note), transform=ax.transAxes, ha="right", va="bottom", fontsize=12)
    _cms(ax, rlabel)
    return _save(fig, out, stem)


# ----------------------------------------------------------------------------- heatmaps


def _mass_sort(pm: pd.DataFrame) -> pd.DataFrame:
    return pm.sort_values(["dm", "mgl", "mn2", "mn1"])


def heatmap_mode(t: pd.DataFrame, points_meta: pd.DataFrame, out: Path, stem: str, rlabel: str, mode: str,
                 ctau: float, hlt: str = "parity", sxy: str = "high", regime_note: str = "") -> list[Path]:
    """Readable heatmap: one analysis mode, one generated lifetime; rows = hypotheses (by Delta m),
    columns = bins grouped by channel with short labels.  Reliable cells show the SF, low-statistics
    cells are hatched (value in small italics), undefined cells are gray with a dash."""
    _style()
    d = _ordered(t[(t.sxy == sxy) & (t.hlt == hlt) & (t["mode"] == mode) & (t.point != "pooled")])
    pm = _mass_sort(points_meta[np.isclose(points_meta.ctau_cm_num, ctau)])
    pts = [p for p in pm.key if p in set(d.point)]
    if not pts or not len(d):
        return []
    cols = d.drop_duplicates("bin")[["bin", "group", "group_display", "short", "role", "kin_label"]].reset_index(drop=True)
    grid = np.full((len(pts), len(cols)), np.nan)
    klass = np.full(grid.shape, "", dtype=object)
    ci = {b: j for j, b in enumerate(cols.bin)}
    ri = {p: i for i, p in enumerate(pts)}
    for r in d.itertuples():
        if r.point in ri:
            grid[ri[r.point], ci[r.bin]] = r.sf
            klass[ri[r.point], ci[r.bin]] = r.sf_class
    finite = grid[np.isfinite(grid) & (grid > 0) & (klass == "reliable")]
    r_ = float(np.nanmax(np.abs(np.log(finite)))) if finite.size else np.log(2)
    r_ = min(max(r_, np.log(1.5)), np.log(10))
    norm = LogNorm(vmin=np.exp(-r_), vmax=np.exp(r_))
    fig, ax = plt.subplots(figsize=(max(9, 1.05 * len(cols) + 5), max(6, 0.55 * len(pts) + 3.5)))
    ax.set_facecolor("0.85")
    show = np.where((grid > 0) & np.isfinite(grid), grid, np.nan)
    im = ax.imshow(show, cmap="RdBu_r", norm=norm, aspect="auto", interpolation="nearest")
    cb = fig.colorbar(im, ax=ax, pad=0.01)
    cb.set_label(SF_LABEL, fontsize=16)
    ticks = [v for v in (0.1, 0.2, 0.25, 0.33, 0.5, 0.67, 0.8, 1, 1.25, 1.5, 2, 3, 4, 5, 10) if norm.vmin <= v <= norm.vmax]
    cb.set_ticks(ticks)
    cb.set_ticklabels([f"{v:g}" for v in ticks])
    cb.minorticks_off()
    for i in range(len(pts)):
        for j in range(len(cols)):
            v, k = grid[i, j], klass[i, j]
            if k == "undefined" or not np.isfinite(v):
                ax.text(j, i, "–", ha="center", va="center", fontsize=11, color="0.3")
            elif k == "low_stats":
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, hatch="///", ec="0.35", lw=0))
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, style="italic", color="0.15")
            else:
                dark = abs(np.log(max(v, 1e-9))) > 0.6 * r_
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=10, color="white" if dark else "black")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols.short, rotation=0, fontsize=12)
    for lab, role in zip(ax.get_xticklabels(), cols.role):
        if role == "anchor":
            lab.set_color(ANCHOR_COLOR)
            lab.set_fontweight("bold")
    lab_map = dict(zip(pm.key, pm.mass_label if "mass_label" in pm else pm.short_label))
    dm_map = dict(zip(pm.key, pm.dm))
    ax.set_yticks(range(len(pts)))
    ax.set_yticklabels([f"{lab_map[p]}  $\\Delta m$={dm_map[p]}" for p in pts], fontsize=11)
    ax.tick_params(which="both", length=0)
    for j in range(1, len(cols)):
        if cols.group[j] != cols.group[j - 1]:
            ax.axvline(j - 0.5, color="black", lw=2)
    for g, gg in cols.groupby("group", sort=False):
        ax.text((gg.index.min() + gg.index.max()) / 2, -0.55, gg.group_display.iloc[0], ha="center", va="bottom",
                fontsize=14, fontweight="bold")
    ax.set_ylabel(r"$(m_{\tilde{g}}, m_{\tilde{\chi}_2^0}, m_{\tilde{\chi}_1^0})$ [GeV]")
    tag = "High" if sxy == "high" else "Low"
    fig.subplots_adjust(top=1 - 1.6 / fig.get_size_inches()[1])
    pos = ax.get_position()
    ytop = 1 - 0.55 / fig.get_size_inches()[1]
    fig.text(pos.x0, ytop, "CMS", fontsize=20, fontweight="bold", ha="left", va="bottom")
    fig.text(pos.x0 + 0.06, ytop, "Simulation Preliminary", fontsize=17, style="italic", ha="left", va="bottom")
    fig.text(pos.x1, ytop, rlabel, fontsize=17, ha="right", va="bottom")
    fig.text(pos.x0, ytop - 0.1 / fig.get_size_inches()[1], f"{MODE_TITLE[mode]} analysis, {tag}-$S_{{xy}}$ bins, generated c$\\tau$ = {ctau:g} cm"
             + ("" if hlt == "parity" else ", HLT dropped in both") + (f"; {regime_note}" if regime_note else ""),
             fontsize=13, ha="left", va="top")
    fig.text(0.01, 0.005, "hatched: low statistics (value not reliable); gray '–': SF undefined (empty sample)",
             fontsize=10, ha="left", va="bottom")
    return _save(fig, out, stem)


def heatmap(t: pd.DataFrame, points_meta: pd.DataFrame, out: Path, stem: str, rlabel: str, hlt: str = "parity",
            sxy: str = "high") -> list[Path]:
    """Debug layout (all hypotheses x all bins, full analysis bin names)."""
    _style()
    d = _ordered(t[(t.sxy == sxy) & (t.hlt == hlt) & (t.point != "pooled")])
    bins = d.drop_duplicates(["mode", "bin"])[["mode", "bin", "group", "role"]].reset_index(drop=True)
    pm = points_meta.sort_values(["ctau_cm_num", "mgl", "mn2", "mn1"])
    pts = [p for p in pm.key if p in set(d.point)]
    grid = np.full((len(pts), len(bins)), np.nan)
    col = {(m, b): j for j, (m, b) in enumerate(zip(bins["mode"], bins["bin"]))}
    row = {p: i for i, p in enumerate(pts)}
    for r in d.itertuples():
        grid[row[r.point], col[(r.mode, r.bin)]] = r.sf
    finite = grid[np.isfinite(grid) & (grid > 0)]
    r_ = max(2.0, float(np.nanmax(np.abs(np.log(finite))))) if finite.size else 2.0
    r_ = min(r_, np.log(10))
    norm = LogNorm(vmin=np.exp(-r_), vmax=np.exp(r_))
    fig, ax = plt.subplots(figsize=(max(12, 0.62 * len(bins) + 5), max(8, 0.36 * len(pts) + 4)))
    ax.set_facecolor("0.85")
    im = ax.imshow(np.where(grid > 0, grid, np.nan), cmap="RdBu_r", norm=norm, aspect="auto", interpolation="nearest")
    cb = fig.colorbar(im, ax=ax, pad=0.01)
    cb.set_label(SF_LABEL)
    ax.set_xticks(range(len(bins)))
    ax.set_xticklabels(bins.bin, rotation=90, fontsize=10)
    ax.set_yticks(range(len(pts)))
    ax.set_yticklabels([dict(zip(pm.key, pm.short_label)).get(p, p) for p in pts], fontsize=10)
    for j in range(1, len(bins)):
        if bins["mode"][j] != bins["mode"][j - 1]:
            ax.axvline(j - 0.5, color="black", lw=3)
        elif bins["group"][j] != bins["group"][j - 1]:
            ax.axvline(j - 0.5, color="0.3", lw=1.2, ls="--")
    _cms(ax, rlabel)
    return _save(fig, out, stem)


# ----------------------------------------------------------------------------- SF vs mass splitting


def sf_vs_dm(t: pd.DataFrame, points_meta: pd.DataFrame, out: Path, stem: str, rlabel: str, hlt: str = "parity",
             ctau: float | None = None, regimes=None, min_neff: float | None = 10):
    """Diagnostic: SF per bin vs mass splitting, one panel per channel (no model drawn).

    Hypotheses outside the analysis mode's own signal regime are drawn on a shaded background.
    """
    _style()
    pm = points_meta if ctau is None else points_meta[np.isclose(points_meta.ctau_cm_num, ctau)]
    d = t[(t.sxy == "high") & (t.hlt == hlt) & (t.point != "pooled")]
    d = d.drop(columns=[c for c in ("dm", "key") if c in d.columns]).merge(pm[["key", "dm"]], left_on="point",
                                                                          right_on="key")
    if not len(d):
        return []
    groups = list(_ordered(d).drop_duplicates(["mode", "group"])[["mode", "group", "group_display"]].itertuples(index=False))
    ncol = 3
    nrow = int(np.ceil(len(groups) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.6 * ncol, 5.8 * nrow), squeeze=False)
    fig.subplots_adjust(hspace=0.45, wspace=0.22, top=0.88)
    from matplotlib.ticker import FuncFormatter, NullFormatter

    sr_colors = [SR_COLOR, "#832db6", "#5a9c3d", "#717581", "#a96b59"]  # never the anchor orange
    counts = dict(clipped=0, undefined=0)
    dms = np.sort(pm.dm.unique())
    for ax, (mode, group, gdisp) in zip(axes.flat, groups):
        g = d[(d["mode"] == mode) & (d.group == group)]
        ylim = auto_ylim(g.sf, g.sf_err, g.sf_class)
        ylim = (ylim[0], ylim[1] + 0.45 * (ylim[1] - ylim[0]))  # headroom for the in-panel legend
        ax.set_ylim(*ylim)
        if regimes is not None and len(dms):
            regime = [r for r, m in regimes.mode_of.items() if m == mode][0]
            lo_dm, hi_dm = dms.min() / 1.4, dms.max() * 1.4
            cut = regimes.dm_max_compressed
            if regime == "compressed":
                ax.axvspan(cut * 1.0001, hi_dm, color="0.9", zorder=-1, lw=0)
            else:
                ax.axvspan(lo_dm, cut, color="0.9", zorder=-1, lw=0)
            ax.set_xlim(lo_dm, hi_dm)
        i_sr = 0
        nb = g.bin.nunique()
        for k, bn in enumerate(_ordered(g).bin.unique()):
            gb = g[g.bin == bn].sort_values("dm")
            anchor = gb.role.iloc[0] == "anchor"
            color = ANCHOR_COLOR if anchor else sr_colors[i_sr % len(sr_colors)]
            i_sr += 0 if anchor else 1
            xoff = gb.dm * (1 + 0.03 * (k - (nb - 1) / 2))  # small horizontal offsets between bins
            c = draw_measurements(ax, xoff, gb.sf, gb.sf_err, gb.sf_class, gb.status, color, "s" if anchor else "o", ylim,
                                  ms=7, label=f"{gb.short.iloc[0]}: {gb.kin_label.iloc[0]}")
            counts = {kk: counts[kk] + c[kk] for kk in counts}
        _unity_band(ax)
        ax.set_xscale("log")
        ax.set_xticks([v for v in (20, 50, 100, 200, 300, 500, 1000, 2000) if dms.min() / 1.4 <= v <= dms.max() * 1.4])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(axis="x", which="minor", bottom=False, top=False)
        ax.tick_params(axis="x", labelsize=13)
        ax.set_title(f"{MODE_TITLE[mode]}: {gdisp}", fontsize=15)
        ax.legend(fontsize=8.5, frameon=False, loc="upper left", handlelength=1.2)
        ax.set_xlabel(r"$\Delta m = m_{\tilde{\chi}_2^0} - m_{\tilde{\chi}_1^0}$ [GeV]", fontsize=14)
        ax.set_ylabel("SF", fontsize=16)
    for ax in axes.flat[len(groups):]:
        ax.axis("off")
    _fig_cms(fig, rlabel, y=0.93)
    txt = (r"High-$S_{xy}$ bins" + (f", generated c$\\tau$ = {ctau:g} cm" if ctau is not None else "")
           + ("" if hlt == "parity" else ", HLT dropped in both")
           + "; shaded: hypotheses outside the mode's signal regime")
    fig.text(0.125, 0.905, txt, fontsize=13, ha="left")
    fig.subplots_adjust(bottom=0.13)
    fig.legend(handles=class_legend_handles(counts, min_neff), loc="lower center", bbox_to_anchor=(0.5, 0.0),
               ncol=4, frameon=False, fontsize=12)
    return _save(fig, out, stem)


# ----------------------------------------------------------------------------- channel and high/low summaries


def channel_summary(t: pd.DataFrame, out: Path, stem: str, rlabel: str, hlt: str = "parity",
                    title: str = "", min_neff: float | None = 10) -> list[Path]:
    """Channel-integrated SF (ratio of sums over the regime pool) per channel, both generated lifetimes."""
    _style()
    d = _ordered(t[(t.hlt == hlt) & (t.role == "channel") & (t.sxy == "high")])
    if not len(d):
        return []
    chans = d.drop_duplicates(["mode", "bin"])[["mode", "bin", "group_display"]].reset_index(drop=True)
    x0 = {(m, b): i for i, (m, b) in enumerate(zip(chans["mode"], chans.bin))}
    fig, ax = plt.subplots(figsize=(13, 7.5))
    _unity_band(ax, label=True)
    srcs = sorted(d.source_ctau_cm.dropna().unique()) if "source_ctau_cm" in d else [None]
    ylim = auto_ylim(d.sf, d.sf_err, d.sf_class)
    ylim = (ylim[0], ylim[1] + 0.3 * (ylim[1] - ylim[0]))
    counts = dict(clipped=0, undefined=0)
    for k, src in enumerate(srcs):
        g = d if src is None else d[np.isclose(d.source_ctau_cm, src)]
        xs = np.array([x0[(m, b)] for m, b in zip(g["mode"], g.bin)]) + (k - (len(srcs) - 1) / 2) * 0.18
        c = draw_measurements(ax, xs, g.sf, g.sf_err, g.sf_class, g.status, PETROFF[[0, 2][k % 2]], "o", ylim,
                              label=None if src is None else f"generated c$\\tau$ = {src:g} cm")
        counts = {kk: counts[kk] + c[kk] for kk in counts}
    ax.set_ylim(*ylim)
    ax.set_xticks(range(len(chans)))
    ax.set_xticklabels(chans.group_display, fontsize=14)
    ax.set_xlim(-0.6, len(chans) - 0.4)
    for i in range(1, len(chans)):
        if chans["mode"][i] != chans["mode"][i - 1]:
            ax.axvline(i - 0.5, color="black", lw=3)
    for mode in MODE_ORDER:
        idx = np.flatnonzero(chans["mode"].to_numpy() == mode)
        if len(idx):
            ax.text((idx[0] + idx[-1]) / 2, ylim[1] - 0.12 * (ylim[1] - ylim[0]), MODE_TITLE[mode], ha="center",
                    fontsize=16, fontweight="bold")
    ax.set_ylabel(SF_LABEL)
    h, _ = ax.get_legend_handles_labels()
    ax.legend(handles=h + class_legend_handles(counts, min_neff), loc="upper left", ncol=2, frameon=False, fontsize=11)
    ax.text(0.99, 0.02, title, transform=ax.transAxes, ha="right", va="bottom", fontsize=12)
    _cms(ax, rlabel)
    return _save(fig, out, stem)


def highlow_plot(hl: pd.DataFrame, out: Path, stem: str, rlabel: str, hlt: str = "parity", title: str = "") -> list[Path]:
    """Double ratio DR = SF(high-S_xy) / SF(low-S_xy) per matched pair, per pool."""
    _style()
    d = _ordered(hl[hl.hlt == hlt].rename(columns={"status": "status_"}))
    if not len(d):
        return []
    bins = d.drop_duplicates(["mode", "bin"])[["mode", "bin", "group", "group_display", "short", "role"]].reset_index(drop=True)
    xi = {(m, b): i for i, (m, b) in enumerate(zip(bins["mode"], bins.bin))}
    pools = list(dict.fromkeys(d.pool))
    fig, ax = plt.subplots(figsize=(max(14, 0.75 * len(bins) + 4), 8))
    _unity_band(ax, label=True)
    cls = np.where(d.status_ == "ok", "reliable", np.where(d.status_ == "low_stats", "low_stats", "undefined"))
    d = d.assign(cls=cls, st=np.where(d.status_ == "undefined", "both_empty", d.status_))
    ylim = auto_ylim(d.double_ratio, d.double_ratio_err, d.cls)
    ylim = (ylim[0], ylim[1] + 0.5 * (ylim[1] - ylim[0]))
    counts = dict(clipped=0, undefined=0)
    for k, pool in enumerate(pools):
        g = d[d.pool == pool]
        xs = np.array([xi[(m, b)] for m, b in zip(g["mode"], g.bin)]) + (k - (len(pools) - 1) / 2) * 0.2
        c = draw_measurements(ax, xs, g.double_ratio, g.double_ratio_err, g.cls, g.st, PETROFF[[0, 2, 4, 5][k % 4]], "o",
                              ylim, ms=8, label=pool_label(pool))
        counts = {kk: counts[kk] + c[kk] for kk in counts}
    ax.set_ylim(*ylim)
    ax.set_xlim(-0.6, len(bins) - 0.4)
    ax.set_xticks(range(len(bins)))
    ax.set_xticklabels([f"{gd}\n{s}" for gd, s in zip(bins.group_display, bins.short)], rotation=90, fontsize=11)
    for i in range(1, len(bins)):
        if bins["mode"][i] != bins["mode"][i - 1]:
            ax.axvline(i - 0.5, color="black", lw=3)
        elif bins.group[i] != bins.group[i - 1]:
            ax.axvline(i - 0.5, color="0.45", lw=1.4, ls="--")
    import matplotlib.transforms as mtrans

    for mode in MODE_ORDER:
        idx = np.flatnonzero(bins["mode"].to_numpy() == mode)
        if len(idx):
            ax.text((idx[0] + idx[-1]) / 2, 0.74, MODE_TITLE[mode], ha="center", fontsize=16, fontweight="bold",
                    transform=mtrans.blended_transform_factory(ax.transData, ax.transAxes))
    ax.set_ylabel(r"$\mathrm{SF}(\mathrm{high}\text{-}S_{xy})\,/\,\mathrm{SF}(\mathrm{low}\text{-}S_{xy})$")
    h, _ = ax.get_legend_handles_labels()
    ax.legend(handles=h + class_legend_handles(counts), loc="upper left", ncol=3, frameon=False, fontsize=11)
    ax.text(0.99, 0.02, title, transform=ax.transAxes, ha="right", va="bottom", fontsize=12)
    _cms(ax, rlabel)
    return _save(fig, out, stem)


def pool_label(pool: str) -> str:
    if pool == "all_hypotheses":
        return "all hypotheses (diagnostic)"
    if "_ct" in pool:
        reg, ct = pool.split("_ct")
        return REGIME_PUBLIC.get(reg, reg) + f" hyp., c$\\tau$ = {ct} cm"
    return pool


# ----------------------------------------------------------------------------- the nominal figure set


def study_figures(T: dict, pts: pd.DataFrame, regimes, fig: Path, rl: str, per_point: bool = True,
                  all_hypotheses: bool = True, full_names: bool = False, min_neff: float = 10) -> list[Path]:
    made = []
    sf, reg = T["sf"], T["sf_pooled_regime"]
    for hlt in ("parity", "symmetric"):
        sfx = "" if hlt == "parity" else "_hltSymmetric"
        for src in sorted(reg.source_ctau_cm.dropna().unique()):
            parts, notes = [], []
            for regime, mode in regimes.mode_of.items():
                g = reg[(reg.pool == f"{regime}_ct{src:g}") & (reg["mode"] == mode)]
                parts.append(g)
                nh = int(g.n_hypotheses.iloc[0]) if len(g) and "n_hypotheses" in g else 0
                notes.append(f"{MODE_TITLE[mode]} mode: {nh} {REGIME_PUBLIC[regime]} hypothes{'is' if nh == 1 else 'es'}")
            t = pd.concat(parts)
            title = f"Ratio of summed yields, generated c$\\tau$ = {src:g} cm"
            stem = f"sf_highSxy_regime_ct{src:g}{sfx}"
            made += sf_plot(t, fig, stem + "_noBinNames", title, rl, hlt, bin_names=False, notes=notes, min_neff=min_neff)
            if full_names:
                made += sf_plot(t, fig, stem, title, rl, hlt, bin_names=True, notes=notes, min_neff=min_neff)
        if all_hypotheses:
            sp = T["sf_pooled"]
            n = int(sp.n_hypotheses.iloc[0]) if "n_hypotheses" in sp and len(sp) else 0
            title = f"Diagnostic: ratio of summed yields over all {n} hypotheses (both lifetimes, both regimes)"
            made += sf_plot(sp, fig, f"sf_highSxy_pooled{sfx}_noBinNames", title, rl, hlt, bin_names=False,
                            min_neff=min_neff)
            if full_names:
                made += sf_plot(sp, fig, f"sf_highSxy_pooled{sfx}", title, rl, hlt, bin_names=True, min_neff=min_neff)
        if sf.point.nunique() > 1:
            for src in sorted(pts.ctau_cm_num.unique()):
                for mode in MODE_ORDER:
                    regime = [r for r, m in regimes.mode_of.items() if m == mode][0]
                    note = f"signal regime of this mode: {REGIME_PUBLIC[regime]}"
                    for sxy in ("high", "low"):
                        made += heatmap_mode(sf, pts, fig, f"sf_heatmap_{sxy}Sxy_{mode}_ct{src:g}{sfx}", rl, mode, src,
                                             hlt, sxy, note)
                made += sf_vs_dm(sf, pts, fig, f"sf_vs_dm_ct{src:g}{sfx}", rl, hlt, src, regimes, min_neff)
            if full_names:
                made += heatmap(sf, pts, fig, f"sf_heatmap_highSxy_allBins{sfx}", rl, hlt, "high")
        cr = T.get("sf_channel_pooled_regime")
        if cr is not None and len(cr):
            parts = [cr[(cr["mode"] == m) & cr.pool.str.startswith(r + "_ct")] for r, m in regimes.mode_of.items()]
            made += channel_summary(pd.concat(parts), fig, f"sf_channel_regime{sfx}", rl, hlt,
                                    "Channel-integrated, ratio of summed yields over each mode's own regime", min_neff)
        hp = T.get("highlow_pooled_regime")
        if hp is not None and len(hp):
            made += highlow_plot(hp, fig, f"sf_high_vs_low_regime{sfx}", rl, hlt,
                                 "Pools restricted to each mode's signal regime; errors include high/low covariance")
    if per_point:
        lab = dict(zip(pts.key, pts.short_label))
        for pk in sorted(sf.point.unique()):
            made += sf_plot(sf[sf.point == pk], fig / "per_point", f"sf_highSxy_{pk}_noBinNames",
                            f"gogoGZ {lab.get(pk, pk)}", rl, "parity", bin_names=False, min_neff=min_neff)
            if full_names:
                made += sf_plot(sf[sf.point == pk], fig / "per_point", f"sf_highSxy_{pk}", f"gogoGZ {lab.get(pk, pk)}",
                                rl, "parity", bin_names=True, min_neff=min_neff)
    return made
