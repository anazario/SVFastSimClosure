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
from dataclasses import asdict, dataclass, field

from . import expr
from .llpcombine import AnalysisConfig, FitConfig

SV_FLAG_RE = re.compile(r"^passN\w+?(High|Low)DxySig(SR|CR)$")

# Display names only (taken from LLPStandardPlots fit_plotter.py conventions, translated
# from ROOT TLatex to mathtext).  They never influence a selection.
GROUP_DISPLAY = {
    # noncompressed: SR channel -> label
    "Ch2SRGeLep1": "Lep SV",
    "Ch4SRGeHad1": "Had SV",
    "Ch14SRgeq1SVHighDxygeq1PhoNotBHLate": r"SV+$\gamma_{d}$",
    "Ch1CRGeLep1": "Lep SV",
    "Ch3CRGeHad1": "Had SV",
    "Ch13CRgeq1SVLowDxygeq1PhoNotBHLate": r"SV+$\gamma_{d}$",
    # compressed
    "SVonly_SR": "SV only",
    "MixDel_SR": r"$\gamma_{d}$+SV",
    "MixPrompt_SR": r"$\gamma_{p}$+SV",
    "SVonly_AnchorCR": "SV only",
    "MixDel_AnchorCR": r"$\gamma_{d}$+SV",
    "MixPrompt_AnchorCR": r"$\gamma_{p}$+SV",
}
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

    # order: follow the FitConfig bin_association order, high-S_xy groups first
    order = []
    for ch, bins in fit.bin_association.get("shape_transfer_fit", {}).items():
        for bn in bins:
            if bn in resolved:
                order.append(bn)
    for i, bn in enumerate(order):
        resolved[bn].order = i
    return sorted(resolved.values(), key=lambda b: b.order)


def check_structure(bins: list[ResolvedBin]) -> list[str]:
    """Consistency checks; returns a list of problems (empty if fine)."""
    problems = []
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
