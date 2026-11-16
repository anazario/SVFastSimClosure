"""Matplotlib/mplhep figures.  No ROOT import anywhere in this module.

Conventions borrowed from LLPStandardPlots (style.py / fit_plotter.py / unrolled.py),
re-implemented in matplotlib:
  * CMS label in bold sans + "Simulation Preliminary" italic, energy label right-aligned,
  * channel-group separators: dashed gray (fit_plotter sub_sep_bins style 2),
  * analysis-mode separator: solid black, width 3 (fit_plotter separator_bins in *_final schemes),
  * group labels drawn above each group, analysis bin names on the axis
    (a *_noBinNames variant shows only the kinematic definitions),
  * shaded ±10% band around SF = 1 to guide the eye.
Bin labels and kinematic ranges come from the resolved-bin manifest (LLPCombine
YAML), never from LLPStandardPlots' unrolled binning tables.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mplhep as hep  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LogNorm, TwoSlopeNorm  # noqa: E402

# CMS-recommended categorical colours (Petroff 10-colour scheme): blue, orange
SR_COLOR = "#3f90da"
ANCHOR_COLOR = "#e76300"
MODE_ORDER = ("compressed", "noncompressed")
MODE_TITLE = {"compressed": "Compressed", "noncompressed": r"High-$\Delta m$"}
BAND_FRAC = 0.10  # guide-the-eye band drawn around SF = 1


def _style():
    plt.style.use(hep.style.CMS)
    plt.rcParams.update({"axes.labelsize": 20, "legend.fontsize": 14})


def _cms(ax, rlabel: str, loc: int = 0):
    hep.cms.label("Preliminary", data=False, rlabel=rlabel, ax=ax, loc=loc, fontsize=20)


def _unity_band(ax, label: bool = False):
    ax.axhspan(1 - BAND_FRAC, 1 + BAND_FRAC, color="0.6", alpha=0.25, lw=0, zorder=0,
               label=rf"$\pm{BAND_FRAC:.0%}$".replace("%", r"\%") if label else None)


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


def sf_plot(t: pd.DataFrame, out: Path, stem: str, title: str, rlabel: str, hlt: str = "parity",
            bin_names: bool = True) -> list[Path]:
    """Primary SF plot: all high-S_xy bins of both modes for one hypothesis (or the pooled set).

    bin_names=False labels each x tick with its kinematic definition only (no analysis bin name).
    """
    _style()
    d = _ordered(t[(t.sxy == "high") & (t.hlt == hlt)])
    n = len(d)
    x = np.arange(n, dtype=float)
    fig, ax = plt.subplots(figsize=(max(14, 0.9 * n + 4), 8.5))
    fig.subplots_adjust(bottom=0.42)

    _unity_band(ax, label=True)
    ax.axhline(1.0, color="0.35", lw=1.5, ls="-", zorder=1)
    for role, color, marker, lab in (
        ("sr", SR_COLOR, "o", r"High-$S_{xy}$ SR bin"),
        ("anchor", ANCHOR_COLOR, "s", r"High-$S_{xy}$ normalization-anchor CR bin"),
    ):
        m = (d.role == role).to_numpy()
        ok = m & np.isfinite(d.sf.to_numpy())
        ax.errorbar(
            x[ok], d.sf[ok], yerr=d.sf_err[ok], fmt=marker, ms=9, color=color, mec="white", mew=1.0,
            elinewidth=2, capsize=0, label=lab, zorder=3,
        )
        empty = m & ~np.isfinite(d.sf.to_numpy())
        if empty.any():
            ax.plot(x[empty], np.ones(empty.sum()), marker=marker, ms=9, mfc="none", mec=color, ls="none", zorder=3)

    sf = d.sf.to_numpy()
    lo = np.nanmin(np.where(np.isfinite(sf), sf - d.sf_err.to_numpy(), np.nan)) if np.isfinite(sf).any() else 0.5
    hi = np.nanmax(np.where(np.isfinite(sf), sf + d.sf_err.to_numpy(), np.nan)) if np.isfinite(sf).any() else 1.5
    span = max(hi - 1, 1 - lo, 0.2)
    ymin, ymax = max(0.0, 1 - 1.25 * span), 1 + 1.7 * span
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(-0.6, n - 0.4)

    # separators + group labels
    prev_mode, prev_group, gstart = None, None, 0
    ytxt = ymax - 0.03 * (ymax - ymin)

    def label_group(s, e, text):
        ax.text((s + e - 1) / 2, ytxt, text, ha="center", va="top", fontsize=15)

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
            ax.text((idx[0] + idx[-1]) / 2, ymax - 0.10 * (ymax - ymin), MODE_TITLE[mode], ha="center", va="top",
                    fontsize=16, fontweight="bold")

    ax.set_xticks(x)
    kin = [k.replace("; ", "\n") for k in d.kin_label]
    ticks = [f"{b}\n{k}" for b, k in zip(d.bin, kin)] if bin_names else kin
    ax.set_xticklabels(ticks, rotation=90, fontsize=11 if bin_names else 13, linespacing=1.25)
    ax.tick_params(axis="x", which="minor", bottom=False, top=False)
    ax.set_ylabel(r"SF = $Y_{\mathrm{FullSim}}\,/\,Y_{\mathrm{FastSim,corr}}$")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.83), frameon=False, ncol=3)
    ax.text(0.99, 0.02, title + ("" if hlt == "parity" else "  [HLT dropped in both]"), transform=ax.transAxes,
            ha="right", va="bottom", fontsize=13)
    _cms(ax, rlabel)
    return _save(fig, out, stem)


def heatmap(t: pd.DataFrame, points_meta: pd.DataFrame, out: Path, stem: str, rlabel: str, hlt: str = "parity",
            sxy: str = "high") -> list[Path]:
    """SF for every matched hypothesis (rows) x bin (columns), diverging around SF = 1."""
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
    ax.set_facecolor("0.85")  # empty / undefined cells
    im = ax.imshow(np.where(grid > 0, grid, np.nan), cmap="RdBu_r", norm=norm, aspect="auto", interpolation="nearest")
    cb = fig.colorbar(im, ax=ax, pad=0.01)
    tag = "High" if sxy == "high" else "Low"
    cb.set_label(tag + r"-$S_{xy}$ bins:  SF = $Y_{\mathrm{FullSim}}/Y_{\mathrm{FastSim,corr}}$"
                 + ("" if hlt == "parity" else "  (HLT dropped in both)"))
    ticks = [t for t in (0.1, 0.2, 0.25, 0.5, 0.67, 0.8, 1, 1.25, 1.5, 2, 4, 5, 10) if norm.vmin <= t <= norm.vmax]
    cb.set_ticks(ticks)
    cb.set_ticklabels([f"{t:g}" for t in ticks])
    cb.minorticks_off()
    ax.set_xticks(range(len(bins)))
    ax.set_xticklabels(bins.bin, rotation=90, fontsize=10)
    for lab, role in zip(ax.get_xticklabels(), bins.role):
        if role == "anchor":
            lab.set_color(ANCHOR_COLOR)
            lab.set_fontweight("bold")
    lab_map = dict(zip(pm.key, pm.short_label))
    ax.set_yticks(range(len(pts)))
    ax.set_yticklabels([lab_map.get(p, p) for p in pts], fontsize=10)
    ax.tick_params(which="minor", left=False, right=False, top=False, bottom=False)
    for j in range(1, len(bins)):
        if bins["mode"][j] != bins["mode"][j - 1]:
            ax.axvline(j - 0.5, color="black", lw=3)
        elif bins["group"][j] != bins["group"][j - 1]:
            ax.axvline(j - 0.5, color="0.3", lw=1.2, ls="--")
    for i in range(1, len(pts)):
        if pts[i].rsplit("_", 1)[1] != pts[i - 1].rsplit("_", 1)[1]:
            ax.axhline(i - 0.5, color="black", lw=2)
    for i in range(len(pts)):
        for j in range(len(bins)):
            v = grid[i, j]
            if np.isfinite(v):
                dark = abs(np.log(max(v, 1e-9))) > 0.6 * r_
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5, color="white" if dark else "black")
    ax.set_ylabel(r"$(m_{\tilde{g}}, m_{\tilde{\chi}_2^0}, m_{\tilde{\chi}_1^0})$ [GeV], c$\tau$")
    _cms(ax, rlabel)
    return _save(fig, out, stem)


def sf_vs_dm(t: pd.DataFrame, points_meta: pd.DataFrame, out: Path, stem: str, rlabel: str, hlt: str = "parity"):
    """Diagnostic: SF per group vs mass splitting (no model drawn)."""
    _style()
    d = t[(t.sxy == "high") & (t.hlt == hlt) & (t.point != "pooled")].merge(
        points_meta[["key", "dm", "ctau_cm"]], left_on="point", right_on="key"
    )
    groups = list(_ordered(d).drop_duplicates(["mode", "group"])[["mode", "group", "group_display"]].itertuples(index=False))
    ncol = 3
    nrow = int(np.ceil(len(groups) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.5 * ncol, 5.6 * nrow), sharey=True, squeeze=False)
    fig.subplots_adjust(hspace=0.42, wspace=0.08, top=0.9)
    sr_colors = [SR_COLOR, "#832db6", "#5a9c3d", "#717581", "#a96b59"]  # never the anchor orange
    from matplotlib.ticker import FuncFormatter, NullFormatter

    for ax, (mode, group, gdisp) in zip(axes.flat, groups):
        g = d[(d["mode"] == mode) & (d.group == group)]
        i_sr = 0
        for bn in _ordered(g).bin.unique():
            gb = g[g.bin == bn]
            anchor = gb.role.iloc[0] == "anchor"
            color = ANCHOR_COLOR if anchor else sr_colors[i_sr % len(sr_colors)]
            i_sr += 0 if anchor else 1
            ok = np.isfinite(gb.sf)
            ax.errorbar(gb.dm[ok], gb.sf[ok], yerr=gb.sf_err[ok], fmt="s" if anchor else "o", ms=7, color=color,
                        alpha=0.9, label=bn, capsize=0, elinewidth=1.5)
        _unity_band(ax)
        ax.axhline(1, color="0.35", lw=1)
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_title(f"{MODE_TITLE[mode]}: {gdisp}", fontsize=15)
        ax.legend(fontsize=9, frameon=False, loc="upper right")
        ax.set_xlabel(r"$\Delta m = m_{\tilde{\chi}_2^0} - m_{\tilde{\chi}_1^0}$ [GeV]", fontsize=14)
    for ax in axes.flat[len(groups):]:
        ax.axis("off")
    for r in range(nrow):
        axes[r, 0].set_ylabel("SF", fontsize=16)
    axes[0, 0].set_ylim(0, 3)
    fig.text(0.125, 0.955, "CMS", fontsize=24, fontweight="bold", ha="left", va="bottom")
    fig.text(0.18, 0.955, "Simulation Preliminary", fontsize=20, style="italic", ha="left", va="bottom")
    fig.text(0.9, 0.955, ("high-$S_{xy}$ bins" + ("" if hlt == "parity" else ", HLT dropped in both")),
             fontsize=16, ha="right", va="bottom")
    return _save(fig, out, stem)
