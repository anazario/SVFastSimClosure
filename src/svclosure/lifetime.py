"""Lifetime-scan configuration: generated (source) lifetimes, target lifetimes, quality criteria.

Units: every lifetime here is a proper decay length c*tau in **cm**.

* The *source* (generated) lifetime of each sample is read from its file name, independently
  for FullSim and FastSim: the ``ct<m>p<mm>`` token is c*tau in **metres** (``ct0p1`` = 0.1 m =
  10 cm), the convention ``BFTool::NormalizeCtauToken`` relies on.  It must agree with the
  LLPCombine signal-key token (cm) and between the FullSim and FastSim file of a pair; any
  disagreement makes the point unusable for the scan.
* The *target* lifetimes come from ``lifetime_scan.targets`` in the study YAML (or ``--targets``):
  ``{source_cm: [target_cm, ...]}``.  The target equal to the source is the direct-MC reference.
* The signal key (``point``, e.g. ``gogoGZ_2300_1300_1000_10``) always keeps the *generated*
  lifetime, exactly as LLPCombine names the process; the target lives in its own column.
"""
from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field

from .llpcombine import normalize_ctau_token

CT_TOKEN_RE = re.compile(r"_ct(-?\d+(?:p\d+)?)_")


def ctau_cm_from_filename(path: str) -> float:
    """c*tau [cm] from the ``ct<m>p<frac>`` token (metres) of a signal file name."""
    name = path.rsplit("/", 1)[-1]
    m = CT_TOKEN_RE.search(name)
    if not m:
        raise ValueError(f"no ct<m>p<frac> token in {name}")
    metres = float(m.group(1).lstrip("-").replace("p", "."))
    return round(metres * 100.0, 9)


def ctau_cm_from_key_token(token: str) -> float:
    """c*tau [cm] from the LLPCombine signal-key ctau token (already cm, e.g. "10", "0.1")."""
    return float(token)


@dataclass
class SourceLifetime:
    point: str
    fullsim_cm: float | None
    fastsim_cm: float | None
    key_cm: float | None
    ok: bool
    problem: str | None = None

    @property
    def cm(self) -> float | None:
        return self.fullsim_cm if self.ok else None


def source_lifetime(point) -> SourceLifetime:
    """Determine and cross-check the generated lifetime of a matched FullSim/FastSim point."""
    try:
        full = ctau_cm_from_filename(point.fullsim)
        fast = ctau_cm_from_filename(point.fastsim)
        tok = point.fullsim.rsplit("/", 1)[-1]
        key = ctau_cm_from_key_token(normalize_ctau_token("ct" + CT_TOKEN_RE.search(tok).group(1)))
        key_from_signal_key = ctau_cm_from_key_token(point.key.rsplit("_", 1)[1])
    except Exception as exc:  # recorded, the point is then excluded from the scan
        return SourceLifetime(point.key, None, None, None, False, f"{type(exc).__name__}: {exc}")
    problems = []
    if full != fast:
        problems.append(f"FullSim {full} cm != FastSim {fast} cm")
    if not math.isclose(full, key, rel_tol=0, abs_tol=1e-9):
        problems.append(f"file token {full} cm != NormalizeCtauToken {key} cm")
    if not math.isclose(key, key_from_signal_key, rel_tol=0, abs_tol=1e-9):
        problems.append(f"signal key says {key_from_signal_key} cm")
    return SourceLifetime(point.key, full, fast, key, not problems, "; ".join(problems) or None)


def fmt_cm(v: float) -> str:
    return f"{v:g}"


def variant_key(target_cm: float | None) -> str:
    """Stable identifier of one weight variant (never the LLPCombine process key)."""
    return "noreweight" if target_cm is None else f"tgt{fmt_cm(target_cm).replace('.', 'p')}cm"


@dataclass
class Quality:
    """Statistical-reliability criteria (configurable; recorded with every run)."""

    min_neff: float = 10.0  # below this in either sample: status low_stats (as the original study)
    min_neff_reweighted: float = 25.0  # reweighted targets need at least this to be drawn as trustworthy
    max_top1_sumw2_frac: float = 0.20  # one event carrying more than this of sum w^2 -> weight_concentrated
    max_top5_sumw2_frac: float = 0.50
    min_neff_ratio: float = 0.0  # optional: n_eff(target)/n_eff(source) floor (0 = off)
    topk: int = 10  # largest weights kept per (selection, HLT mode, target) for the diagnostics
    # per-target verdict from the per-hypothesis SFs whose direct reference is status ok:
    target_usable_frac_channel: float = 0.90  # usable: this fraction of channel-integrated SFs reliable ...
    target_usable_frac_bin: float = 0.75  # ... and this fraction of single-bin SFs reliable
    target_limited_frac_channel: float = 0.50  # limited: at least this fraction of channel SFs; else unreliable


@dataclass
class LifetimeScan:
    targets: dict[float, list[float]] = field(default_factory=dict)  # source cm -> target cm (nominal)
    crosscheck_targets: dict[float, list[float]] = field(default_factory=dict)  # extra, role cross_check
    invalid_ctau_policy: str = "llpcombine"  # llpcombine | exclude | fail
    quality: Quality = field(default_factory=Quality)
    representative_points: list[str] = field(default_factory=list)  # mass keys for per-hypothesis figures

    POLICIES = ("llpcombine", "exclude", "fail")

    def validate(self) -> list[str]:
        problems = []
        if self.invalid_ctau_policy not in self.POLICIES:
            problems.append(f"invalid_ctau_policy must be one of {self.POLICIES}")
        for src, tg in list(self.targets.items()) + list(self.crosscheck_targets.items()):
            if not (isinstance(src, (int, float)) and math.isfinite(src) and src > 0):
                problems.append(f"invalid source lifetime {src!r} (must be finite and > 0 cm)")
            if not tg:
                problems.append(f"source {src}: empty target list")
            for t in tg:
                if not (isinstance(t, (int, float)) and math.isfinite(t) and t > 0):
                    problems.append(f"source {src}: invalid target lifetime {t!r} (must be finite and > 0 cm)")
            if len(set(tg)) != len(tg):
                problems.append(f"source {src}: duplicate targets {tg}")
        q = self.quality
        if q.min_neff < 0 or q.min_neff_reweighted < 0 or not (0 < q.max_top1_sumw2_frac <= 1):
            problems.append("quality thresholds out of range")
        if q.topk < 5:
            problems.append("quality.topk must be >= 5 (top-5 diagnostics)")
        return problems

    def targets_for(self, source_cm: float, crosscheck: bool = True) -> list[float] | None:
        """Nominal targets of a generated lifetime (+ the cross-check targets); None if not configured."""
        out = None
        maps = [self.targets] + ([self.crosscheck_targets] if crosscheck else [])
        for mp in maps:
            for s, t in mp.items():
                if math.isclose(s, source_cm, rel_tol=0, abs_tol=1e-9):
                    out = sorted(set(out or []) | set(t))
        if out is not None and self.targets_for_nominal(source_cm) is None:
            return None  # cross-check targets alone do not enable a source
        return out

    def targets_for_nominal(self, source_cm: float) -> list[float] | None:
        for s, t in self.targets.items():
            if math.isclose(s, source_cm, rel_tol=0, abs_tol=1e-9):
                return sorted(t)
        return None

    def role(self, source_cm: float, target_cm: float) -> str:
        nom = self.targets_for_nominal(source_cm) or []
        if any(math.isclose(t, target_cm, abs_tol=1e-9) for t in nom):
            return "direct" if math.isclose(source_cm, target_cm, abs_tol=1e-9) else "nominal"
        return "cross_check" if target_cm < source_cm else "stress_test"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["targets"] = {fmt_cm(k): v for k, v in self.targets.items()}
        d["crosscheck_targets"] = {fmt_cm(k): v for k, v in self.crosscheck_targets.items()}
        return d


def parse_targets(spec: str) -> dict[float, list[float]]:
    """CLI syntax ``"10:1,5,10,15;50:20,30,40,50,70"`` -> {10.0: [...], 50.0: [...]}."""
    out: dict[float, list[float]] = {}
    for part in filter(None, (p.strip() for p in spec.split(";"))):
        if ":" not in part:
            raise ValueError(f"bad --targets element {part!r}; expected SOURCE:T1,T2,...")
        src, tg = part.split(":", 1)
        try:
            out[float(src)] = [float(t) for t in tg.split(",") if t.strip()]
        except ValueError as exc:
            raise ValueError(f"bad --targets element {part!r}: {exc}") from None
    return out


def from_config(raw: dict | None, targets_override: str | None = None, **quality_overrides) -> LifetimeScan:
    raw = raw or {}
    targets = {float(k): [float(x) for x in v] for k, v in (raw.get("targets") or {}).items()}
    cross = {float(k): [float(x) for x in v] for k, v in (raw.get("crosscheck_targets") or {}).items()}
    if targets_override:
        targets = parse_targets(targets_override)
    q = Quality(**(raw.get("quality") or {}))
    for k, v in quality_overrides.items():
        if v is not None:
            setattr(q, k, v)
    scan = LifetimeScan(targets=targets, crosscheck_targets=cross, invalid_ctau_policy=raw.get("invalid_ctau_policy", "llpcombine"), quality=q,
                        representative_points=list(raw.get("representative_points") or []))
    problems = scan.validate()
    if problems:
        raise ValueError("invalid lifetime_scan configuration:\n  " + "\n  ".join(problems))
    return scan


# ----------------------------------------------------------------------------- signal regimes


@dataclass
class Regimes:
    """Which signal hypotheses each analysis mode targets (for the nominal summaries).

    Primary rule: Delta m = m(N2) - m(N1) <= ``dm_max_compressed`` -> compressed, else high-Delta-m.
    The ``alternatives`` are other conventions found in the LLPCombine macros; a point on which
    they disagree is reported and left out of the regime-restricted summaries (it stays in the
    all-hypotheses diagnostic).  ``benchmarks`` must classify as listed, otherwise loading fails.
    """

    dm_max_compressed: float
    mode_of: dict[str, str]  # regime -> analysis mode
    alternatives: list[dict] = field(default_factory=list)
    benchmarks: dict[str, list[list[int]]] = field(default_factory=dict)
    source_note: str = ""

    def primary(self, mgl: int, mn2: int, mn1: int) -> str:
        return "compressed" if (mn2 - mn1) <= self.dm_max_compressed else "high_dm"

    @staticmethod
    def _alt(rule: dict, mgl: int, mn2: int, mn1: int) -> str:
        ok = True
        if "max_mgl_minus_mn2" in rule:
            ok &= (mgl - mn2) <= rule["max_mgl_minus_mn2"]
        if "max_mn2_minus_mn1" in rule:
            ok &= (mn2 - mn1) <= rule["max_mn2_minus_mn1"]
        return "compressed" if ok else "high_dm"

    def classify(self, mgl: int, mn2: int, mn1: int) -> tuple[str, bool, dict]:
        """(regime, unambiguous, {alternative name: regime})."""
        p = self.primary(mgl, mn2, mn1)
        alts = {a["name"]: self._alt(a, mgl, mn2, mn1) for a in self.alternatives}
        return p, all(v == p for v in alts.values()), alts

    def check_benchmarks(self) -> list[str]:
        problems = []
        for regime, pts in self.benchmarks.items():
            for m in pts:
                got = self.primary(*m)
                if got != regime:
                    problems.append(f"benchmark {tuple(m)} is listed as {regime} but classifies as {got}")
        return problems


REGIME_TITLE = {"compressed": "Compressed", "high_dm": r"High-$\Delta m$"}


def regimes_from_config(raw: dict) -> Regimes:
    r = Regimes(dm_max_compressed=float(raw["dm_max_compressed"]), mode_of=dict(raw["mode_of"]),
                alternatives=list(raw.get("alternatives") or []), benchmarks=dict(raw.get("benchmarks") or {}),
                source_note=raw.get("source_note", ""))
    problems = r.check_benchmarks()
    if set(r.mode_of) != {"compressed", "high_dm"}:
        problems.append(f"signal_regimes.mode_of must map compressed and high_dm, got {sorted(r.mode_of)}")
    if problems:
        raise ValueError("invalid signal_regimes:\n  " + "\n  ".join(problems))
    return r
