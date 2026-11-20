"""Resolved-bin manifest: which bins are SV bins, their S_xy class, role, group and labels.

Everything is derived from the YAML selections and FitConfig associations:

* S_xy class: the SV flag the bin *requires* (``passN...HighDxySigSR == 1`` -> high,
  ``passN...LowDxySigCR == 1`` -> low).  Bins requiring neither are not SV bins.
* group: the shape-transfer channel the bin belongs to in the FitConfig.
* role: ``anchor`` (high-S_xy bin named ``...CR..`` inside an SR channel, i.e. the
  lowest-sensitivity normalization anchor, e.g. ``Ch2CRGeLep100``/``SVonly_CR00``),
  ``sr`` (other high-S_xy bins) or ``lowsxy_cr`` (ordinary low-S_xy control bins,
  e.g. ``Ch1CRGeLep1xx``/``SVonly_AnchorCRxx``; note the compressed naming where
  ``AnchorCR`` is the *low*-S_xy CR).
* low-S_xy partner: the bin whose cut multiset equals this one's with the
  High/Low SV flag swapped.
* kinematic label: intervals intersected from the bin's own top-level conjuncts.
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np
from dataclasses import asdict, dataclass, field

from . import expr
from .llpcombine import AnalysisConfig, FitConfig

SV_FLAG_RE = re.compile(r"^passN\w+?(High|Low)DxySig(SR|CR)$")

# Display names only (taken from LLPStandardPlots fit_plotter.py conventions, translated
# from ROOT TLatex to mathtext).  They never influence a selection.
GP, GD = r"$\gamma_{\mathrm{p}}$", r"$\gamma_{\mathrm{d}}$"
GROUP_DISPLAY = {  # the single authoritative group -> display-label mapping (group names are mode specific)
    # noncompressed (High-Delta-m): SR channel -> label
    "Ch2SRGeLep1": "Lep SV",
    "Ch4SRGeHad1": "Had SV",
    "Ch14SRgeq1SVHighDxygeq1PhoNotBHLate": f"SV + {GD}",
    "Ch1CRGeLep1": "Lep SV",
    "Ch3CRGeHad1": "Had SV",
    "Ch13CRgeq1SVLowDxygeq1PhoNotBHLate": f"SV + {GD}",
    # compressed
    "SVonly_SR": "SV only",
    "MixDel_SR": f"SV + {GD}",
    "MixPrompt_SR": f"SV + {GP}",
    "SVonly_AnchorCR": "SV only",
    "MixDel_AnchorCR": f"SV + {GD}",
    "MixPrompt_AnchorCR": f"SV + {GP}",
}
GROUP_MODE = {  # analysis mode of each channel group (for display_label's consistency check)
    **{g: "noncompressed" for g in ("Ch2SRGeLep1", "Ch4SRGeHad1", "Ch14SRgeq1SVHighDxygeq1PhoNotBHLate", "Ch1CRGeLep1",
                                    "Ch3CRGeHad1", "Ch13CRgeq1SVLowDxygeq1PhoNotBHLate")},
    **{g: "compressed" for g in ("SVonly_SR", "MixDel_SR", "MixPrompt_SR", "SVonly_AnchorCR", "MixDel_AnchorCR",
                                 "MixPrompt_AnchorCR")},
}


def display_label(mode: str, group_or_bin: str) -> str:
    """Display label of an (analysis mode, channel group); a bin name ``group:...`` is accepted.

    Looked up live from GROUP_DISPLAY (never from the ``group_display`` column cached in a run's CSVs)."""
    g = group_or_bin.split(":")[0]
    if GROUP_MODE.get(g, mode) != mode:
        raise ValueError(f"group {g!r} belongs to {GROUP_MODE[g]!r}, not {mode!r}")
    return GROUP_DISPLAY.get(g, g)


GROUP_CHANNEL_TYPE = {  # SV-only vs SV+photon, for bookkeeping in the report
    "Ch2SRGeLep1": "SV-only",
    "Ch4SRGeHad1": "SV-only",
    "Ch14SRgeq1SVHighDxygeq1PhoNotBHLate": "SV+photon",
    "SVonly_SR": "SV-only",
    "MixDel_SR": "SV+photon",
    "MixPrompt_SR": "SV+photon",
}

# Kinematic variables to label, in display order: (expression, mathtext symbol, scale, unit)
LABEL_VARS = [
    ("rjr_Ms[0]", r"M_{S}", 1e-3, "TeV"),
    ("rjr_Rs[0]", r"R_{S}", 1.0, ""),
    ("rjrIsr_PtIsr", r"p_{T}^{ISR}", 1.0, "GeV"),
    ("rjrIsr_RIsr", r"R_{ISR}", 1.0, ""),
]


@dataclass
class ResolvedBin:
    mode: str
    name: str
    group: str  # fit channel (shape_transfer_fit)
    group_display: str
    channel_type: str
    sxy: str  # "high" | "low"
    role: str  # "anchor" | "sr" | "lowsxy_cr"
    sv_flags: list[str]
    partner: str | None  # low-S_xy counterpart of a high-S_xy bin (or vice versa)
    kin_label: str
    kin_intervals: dict[str, list]
    expression: str  # LLPCombine combined cut string (FullSim / fit as written)
    expression_no_hlt: str  # Trigger_* -> true (LLPCombine FastSim behaviour)
    branches: list[str]
    branches_no_hlt: list[str]
    order: int = 0
    short: str = ""  # concise public label, e.g. "SR10", "CR00" (low-S_xy: "SR10 low")
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def required_sv_flags(node: expr.Node) -> list[tuple[str, str]]:
    """(flag, 'high'|'low') for every top-level ``flag == 1`` SV requirement."""
    out = []
    for term in expr.conjuncts(node):
        if (
            isinstance(term, expr.Cmp)
            and term.op == "=="
            and isinstance(term.left, expr.Var)
            and isinstance(term.right, expr.Num)
            and term.right.value == 1
        ):
            m = SV_FLAG_RE.match(term.left.name)
            if m:
                out.append((term.left.name, m.group(1).lower()))
    return out


def _fmt(v: float, unit: str = "") -> str:
    if unit == "TeV":
        return f"{v:.1f}" if round(v, 1) == v else f"{v:g}"
    return f"{v:g}"


def interval_label(iv: expr.Interval, symbol: str, scale: float, unit: str) -> str:
    lo, hi = iv.lo * scale, iv.hi * scale
    f = lambda v: _fmt(v, unit)  # noqa: E731
    u = f"\\,\\mathrm{{{unit}}}" if unit else ""
    if iv.lo == iv.hi:
        return f"${symbol} = {f(lo)}{u}$"
    if not (iv.hi < float("inf")):
        op = r"\geq" if iv.lo_closed else ">"
        return f"${symbol} {op} {f(lo)}{u}$"
    if not (iv.lo > float("-inf")):
        op = r"\leq" if iv.hi_closed else "<"
        return f"${symbol} {op} {f(hi)}{u}$"
    lb = "[" if iv.lo_closed else "("
    rb = "]" if iv.hi_closed else ")"
    return f"${symbol} \\in {lb}{f(lo)},{f(hi)}{rb}{u}$"


def kinematic_labels(node: expr.Node) -> tuple[str, dict[str, list]]:
    parts, raw = [], {}
    for var, symbol, scale, unit in LABEL_VARS:
        iv = expr.variable_interval(node, var)
        if iv.bounded:
            parts.append(interval_label(iv, symbol, scale, unit))
            raw[var] = [iv.lo, iv.lo_closed, iv.hi, iv.hi_closed]
    return "; ".join(parts), raw


def resolve_bins(mode: str, cfg: AnalysisConfig, fit: FitConfig) -> list[ResolvedBin]:
    """All SV-containing bins of one analysis mode (both S_xy classes)."""
    resolved: dict[str, ResolvedBin] = {}
    sv_counter: dict[str, Counter] = {}
    for name in cfg.bins:
        full = cfg.combined_cuts(name)
        node = expr.parse(full)
        flags = required_sv_flags(node)
        if not flags:
            continue
        classes = {c for _, c in flags}
        if len(classes) != 1:
            raise ValueError(f"{mode}/{name}: requires both high and low S_xy flags {flags}")
        sxy = classes.pop()
        group = fit.channel_of(name)
        if group is None:
            raise ValueError(f"{mode}/{name}: SV bin not found in shape_transfer_fit.bin_association")
        stripped = expr.strip_triggers(full)
        if sxy == "high":
            role = "anchor" if ("CR" in name and "SR" not in name) else "sr"
        else:
            role = "lowsxy_cr"
        label, ivs = kinematic_labels(node)
        resolved[name] = ResolvedBin(
            mode=mode,
            name=name,
            group=group,
            group_display=GROUP_DISPLAY.get(group, group),
            channel_type=GROUP_CHANNEL_TYPE.get(group) or GROUP_CHANNEL_TYPE.get(
                (fit.channel_association.get("shape_transfer_fit", {}).get(group) or [group])[0], ""
            ),
            sxy=sxy,
            role=role,
            sv_flags=[f for f, _ in flags],
            partner=None,
            kin_label=label,
            kin_intervals=ivs,
            expression=full,
            expression_no_hlt=stripped,
            branches=sorted(node.branches()),
            branches_no_hlt=sorted(expr.parse(stripped).branches()),
        )
        sv_counter[name] = Counter(cfg.bins[name])

    # partners: identical cut multiset after swapping the SV flag
    def swapped(cuts: Counter, to_low: bool) -> Counter:
        out = Counter()
        for c, n in cuts.items():
            c2 = (
                re.sub(r"HighDxySigSR", "LowDxySigCR", c) if to_low else re.sub(r"LowDxySigCR", "HighDxySigSR", c)
            )
            out[c2] += n
        return out

    for name, b in resolved.items():
        target = swapped(sv_counter[name], to_low=(b.sxy == "high"))
        matches = [o for o in resolved if o != name and sv_counter[o] == target]
        if len(matches) > 1:
            raise ValueError(f"{mode}/{name}: ambiguous S_xy partner {matches}")
        b.partner = matches[0] if matches else None

    for b in resolved.values():
        b.short = short_label(b)
    for b in resolved.values():  # a low-S_xy bin is named after its high-S_xy partner
        if b.sxy == "low" and b.partner in resolved:
            b.short = resolved[b.partner].short + " low"

    # order: follow the FitConfig bin_association order, high-S_xy groups first
    order = []
    for ch, bins in fit.bin_association.get("shape_transfer_fit", {}).items():
        for bn in bins:
            if bn in resolved:
                order.append(bn)
    for i, bn in enumerate(order):
        resolved[bn].order = i
    return sorted(resolved.values(), key=lambda b: b.order)


def short_label(b: ResolvedBin) -> str:
    """Concise label from the bin role and its two-digit kinematic index (never used to select).

    High-S_xy anchor -> "CR00", high-S_xy SR -> "SR10"; a low-S_xy bin gets its high-S_xy
    partner's label with a "low" suffix once partners are known (see :func:`resolve_bins`).
    """
    idx = b.name[-2:] if b.name[-2:].isdigit() else b.name
    if b.sxy == "high":
        return ("CR" if b.role == "anchor" else "SR") + idx
    return "CR" + idx + " low"


def _kin_vars(node: expr.Node) -> set[str]:
    out = set()
    for t in expr.conjuncts(node):
        if isinstance(t, expr.Cmp):
            for side, other in ((t.left, t.right), (t.right, t.left)):
                if isinstance(side, expr.Var) and isinstance(other, expr.Num):
                    out.add(side.cpp())
    return out


def provably_disjoint(a: str, b: str) -> bool:
    """True if some variable's top-level interval requirements of ``a`` and ``b`` cannot both hold."""
    na, nb = expr.parse(a), expr.parse(b)
    for var in _kin_vars(na) & _kin_vars(nb):
        ia, ib = expr.variable_interval(na, var), expr.variable_interval(nb, var)
        lo, lo_c = (ia.lo, ia.lo_closed) if (ia.lo > ib.lo or (ia.lo == ib.lo and not ia.lo_closed)) else (ib.lo, ib.lo_closed)
        hi, hi_c = (ia.hi, ia.hi_closed) if (ia.hi < ib.hi or (ia.hi == ib.hi and not ia.hi_closed)) else (ib.hi, ib.hi_closed)
        if lo > hi or (lo == hi and not (lo_c and hi_c)):
            return True
    return False


@dataclass
class ReferenceSelection:
    """An inclusive reference selection used to look for events that enter no expected category.

    * ``group_envelope``: the conjunction of every top-level cut shared by all bins of one
      channel group.  The group's bins should partition it (they differ only by kinematic
      intervals); events inside it but in no bin are reported as omissions.
    * ``sxy_inclusive``: all cuts of a high/low-S_xy partner pair except the S_xy flag, AND
      ">= 1 SV of the same type in any S_xy class" (OR of every ``passN<type>Ge1Selection*DxySig*``
      flag in the file).  High + low are not expected to exhaust it (mid-S_xy / validation
      classes exist); the fraction they cover is reported, never assumed.
    """

    mode: str
    name: str
    kind: str
    expression: str
    expression_no_hlt: str
    members: list[str]
    note: str = ""
    gap_expression: str = ""  # events of the reference outside every member's kinematic range (static)
    gaps: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _iv_tuple(iv: expr.Interval) -> tuple:
    return (iv.lo, iv.lo_closed, iv.hi, iv.hi_closed)


def coverage_gaps(reference: str, members: list[str]) -> list[tuple[str, tuple]]:
    """1-D kinematic gaps: for each variable bounded by the members, the parts of the reference's
    range that no member's interval covers.  Returns [(variable, (lo, lo_closed, hi, hi_closed))]."""
    ref = expr.parse(reference)
    nodes = [expr.parse(m) for m in members]
    out = []
    for var, *_ in LABEL_VARS:
        ivs = [expr.variable_interval(n, var) for n in nodes]
        if not all(iv.bounded for iv in ivs):
            continue
        r = expr.variable_interval(ref, var)
        cur_lo, cur_closed = r.lo, r.lo_closed  # sweep from the reference's lower edge
        for iv in sorted(ivs, key=lambda i: (i.lo, not i.lo_closed)):
            starts_after = iv.lo > cur_lo or (iv.lo == cur_lo and cur_closed and not iv.lo_closed)
            if starts_after:
                out.append((var, (cur_lo, cur_closed, iv.lo, not iv.lo_closed)))
            if iv.hi > cur_lo or (iv.hi == cur_lo and iv.hi_closed):
                cur_lo, cur_closed = iv.hi, not iv.hi_closed
        if cur_lo < r.hi or (cur_lo == r.hi and cur_closed and r.hi_closed):
            out.append((var, (cur_lo, cur_closed, r.hi, r.hi_closed)))
    return [(v, g) for v, g in out if not (g[0] == g[2] and not (g[1] and g[3])) and not (g[0] > g[2])]


def _gap_expr(var: str, g: tuple) -> str:
    lo, lc, hi, hc = g
    parts = []
    if np.isfinite(lo):
        parts.append(f"({var} {'>=' if lc else '>'} {lo:g})")
    if np.isfinite(hi):
        parts.append(f"({var} {'<=' if hc else '<'} {hi:g})")
    return "(" + " && ".join(parts) + ")" if parts else "true"


def _common_conjuncts(exprs: list[str]) -> list[expr.Node]:
    nodes = [expr.parse(e) for e in exprs]
    first = expr.conjuncts(nodes[0])
    common_keys = set.intersection(*({c.cpp() for c in expr.conjuncts(n)} for n in nodes))
    seen, out = set(), []
    for c in first:
        k = c.cpp()
        if k in common_keys and k not in seen:
            seen.add(k)
            out.append(c)
    return out


def _join(nodes: list[expr.Node]) -> str:
    return " && ".join(n.cpp() for n in nodes) if nodes else "true"


def reference_selections(bins: list[ResolvedBin], available_branches: set[str] | None = None) -> list[ReferenceSelection]:
    refs = []
    groups: dict[tuple[str, str], list[ResolvedBin]] = {}
    for b in bins:
        groups.setdefault((b.mode, b.group), []).append(b)
    for (mode, group), bs in groups.items():
        common = _common_conjuncts([b.expression for b in bs])
        e = _join(common)
        gaps = coverage_gaps(e, [b.expression for b in bs])
        gexpr = " || ".join(_gap_expr(v, g) for v, g in gaps)
        refs.append(ReferenceSelection(mode, f"envelope:{group}", "group_envelope", e, expr.strip_triggers(e),
                                       [b.name for b in bs], f"cuts shared by all {len(bs)} bins of {group}",
                                       ("(" + gexpr + ")") if gexpr else "", [_gap_expr(v, g) for v, g in gaps]))
    by = {(b.mode, b.name): b for b in bins}
    for b in bins:
        if b.sxy != "high" or b.partner is None:
            continue
        lo = by[(b.mode, b.partner)]
        common = [c for c in _common_conjuncts([b.expression, lo.expression])]
        m = SV_FLAG_RE.match(b.sv_flags[0])
        obj = re.match(r"passN(\w+?)Ge1Selection", b.sv_flags[0])
        if not (m and obj):
            continue
        pat = re.compile(rf"^passN{obj.group(1)}Ge1Selection\w*DxySig\w*$")
        flags = sorted(x for x in (available_branches or set(b.sv_flags + lo.sv_flags)) if pat.match(x))
        if not flags:
            continue
        anysv = "(" + " || ".join(f"({f} == 1)" for f in flags) + ")"
        e = _join(common) + " && " + anysv
        refs.append(ReferenceSelection(b.mode, f"sxyincl:{b.name}", "sxy_inclusive", e, expr.strip_triggers(e),
                                       [b.name, lo.name], f">=1 {obj.group(1)} SV in any S_xy class: {flags}"))
    return refs


def check_structure(bins: list[ResolvedBin]) -> list[str]:
    """Consistency checks; returns a list of problems (empty if fine)."""
    problems = []
    by_group: dict[tuple[str, str], list[ResolvedBin]] = {}
    for b in bins:
        by_group.setdefault((b.mode, b.group), []).append(b)
    for (mode, g), bs in by_group.items():  # kinematic bins of one channel must be disjoint by construction
        for i, a in enumerate(bs):
            for c in bs[i + 1:]:
                if not provably_disjoint(a.expression_no_hlt, c.expression_no_hlt):
                    problems.append(f"{mode}/{g}: bins {a.name} and {c.name} are not provably disjoint")
    groups: dict[str, list[ResolvedBin]] = {}
    for b in bins:
        if b.sxy == "high":
            groups.setdefault(b.group, []).append(b)
    for g, bs in groups.items():
        anchors = [b for b in bs if b.role == "anchor"]
        if len(anchors) != 1:
            problems.append(f"group {g}: expected exactly one anchor bin, found {[b.name for b in anchors]}")
        elif not anchors[0].name.endswith("00"):
            problems.append(f"group {g}: anchor {anchors[0].name} does not end in 00")
        elif bs[0] is not anchors[0]:
            problems.append(f"group {g}: anchor {anchors[0].name} is not first in bin_association")
    for b in bins:
        if b.partner is None:
            problems.append(f"{b.mode}/{b.name}: no S_xy partner bin found")
    return problems
