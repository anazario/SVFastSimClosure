"""Signal event weights exactly as LLPCombine builds ``evtrwt``.

BuildFitInput::LoadSig_KeyValue:   evtwt  = evtFillWgt * Lumi
BuildFitInput::BuildReweights:     evtrwt = evtwt * xawt * xbwt * decayWeight   (lifetime on)
                                   evtrwt = evtwt * decayWeight                 (lifetime off)
main.cpp: with no ``decayWeights`` block in the YAML, sample and target Z/G rates
default to 0.5, so decay reweighting is *always* switched on.  With equal source
and target rates ``ReweightTool::DecayReweight`` returns 1 for events flagged
``Evt_isZZ``/``Evt_isGZ``/``Evt_isGG`` and **0** for events with none of the flags.
Statistical error: sqrt(sum evtrwt^2).

Lifetime reweighting (``ReweightTool::LifetimeReweight(float ctau, float tau_old, float tau_new)``):

    w(t; tau0 -> tau) = (tau0 / tau) * exp[t * (1/tau0 - 1/tau)]      per LLP, t = Xa_ctau / Xb_ctau

All lifetimes are proper decay lengths c*tau in **cm** (``Xa_ctau``/``Xb_ctau`` are in
cm; LLPCombine's ``sampleLifetime``/``targetLifetime`` and the signal-key ctau token
are in cm).  The C++ evaluates ``tau_old / tau_new`` in *float*, the exponent in
*double*, and casts the product back to float; :func:`lifetime_weight` reproduces
that bit for bit (verified against the header in tests and in ``validate``).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DECAY_BRANCHES = ("Evt_isZZ", "Evt_isGZ", "Evt_isGG")
LIFETIME_BRANCHES = ("Xa_ctau", "Xb_ctau")


@dataclass(frozen=True)
class WeightSpec:
    luminosity: float
    sample_z: float | None = 0.5
    sample_g: float | None = 0.5
    target_z: float | None = 0.5
    target_g: float | None = 0.5
    sample_lifetime: float | None = None
    target_lifetime: float | None = None

    @property
    def decay_on(self) -> bool:
        return self.sample_z is not None and self.sample_z != -1

    @property
    def lifetime_on(self) -> bool:
        return self.sample_lifetime is not None and self.sample_lifetime != -1

    def branches(self) -> list[str]:
        b = ["evtFillWgt"]
        if self.decay_on:
            b += list(DECAY_BRANCHES)
        if self.lifetime_on:
            b += list(LIFETIME_BRANCHES)
        return b

    def lumi_cpp(self) -> str:
        """LLPCombine passes Lumi through std::to_string -> 6 decimals in the JIT string."""
        return f"{self.luminosity:.6f}"


def _binom2(p: float) -> float:
    # ROOT::Math::binomial_pdf(k=2, p, n=2) = p^2
    return p * p


def decay_weight(is_zz: np.ndarray, is_gz: np.ndarray, is_gg: np.ndarray, spec: WeightSpec) -> np.ndarray:
    """Vectorised ``ReweightTool::DecayReweight`` (later flags override earlier ones, default 0)."""
    p_zz0, p_gg0 = _binom2(spec.sample_z), _binom2(spec.sample_g)
    p_gz0 = 1.0 - (p_zz0 + p_gg0)
    p_zz1 = spec.target_z if spec.target_z in (0, 1) else _binom2(spec.target_z)
    p_gg1 = spec.target_g if spec.target_g in (0, 1) else _binom2(spec.target_g)
    p_gz1 = 1.0 - (p_zz1 + p_gg1)
    w = np.zeros(len(is_zz), dtype=np.float64)
    w[np.asarray(is_zz, dtype=bool)] = p_zz1 / p_zz0
    w[np.asarray(is_gz, dtype=bool)] = p_gz1 / p_gz0
    w[np.asarray(is_gg, dtype=bool)] = p_gg1 / p_gg0
    return w


def lifetime_weight(ctau: np.ndarray, tau_old: float, tau_new: float) -> np.ndarray:
    """Bit-exact vectorised ``ReweightTool::LifetimeReweight`` (float32 result).

    C++:  ``double inv_old = 1.0/tau_old, inv_new = 1.0/tau_new;``
          ``return float((tau_old / tau_new) * std::exp(ctau * (inv_old - inv_new)));``
    with float ``ctau``, ``tau_old``, ``tau_new``: the ratio is a *float* division, the
    exponent is evaluated in double.  (Computing the exponent in float32, as NumPy 2
    does for ``float32_array * python_float``, differs by up to ~100 ulp.)
    """
    to, tn = np.float32(tau_old), np.float32(tau_new)
    ratio = np.float64(to / tn)
    d = 1.0 / np.float64(to) - 1.0 / np.float64(tn)
    c = np.asarray(ctau, dtype=np.float32).astype(np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        return (ratio * np.exp(c * d)).astype(np.float32)


def base_weights(arrays, spec: WeightSpec) -> tuple[np.ndarray, np.ndarray | None]:
    """(evtwt, decayWeight or None): the lifetime-independent factors of ``evtrwt``."""
    lumi = float(spec.lumi_cpp())  # evtFillWgt is float; float * double -> double in the RDF JIT
    evtwt = np.asarray(arrays["evtFillWgt"], dtype=np.float64) * lumi
    dw = decay_weight(arrays["Evt_isZZ"], arrays["Evt_isGZ"], arrays["Evt_isGG"], spec) if spec.decay_on else None
    return evtwt, dw


def combine(evtwt: np.ndarray, dw: np.ndarray | None, xa: np.ndarray | None = None, xb: np.ndarray | None = None) -> np.ndarray:
    """``evtwt*xawt*xbwt*decayWeight`` evaluated left to right in double, as the RDF JIT does."""
    w = evtwt
    if xa is not None:
        w = w * xa.astype(np.float64) * xb.astype(np.float64)
    if dw is not None:
        w = w * dw
    return w


def event_weights(arrays, spec: WeightSpec) -> np.ndarray:
    evtwt, dw = base_weights(arrays, spec)
    if spec.lifetime_on:
        xa = lifetime_weight(arrays["Xa_ctau"], spec.sample_lifetime, spec.target_lifetime)
        xb = lifetime_weight(arrays["Xb_ctau"], spec.sample_lifetime, spec.target_lifetime)
        return combine(evtwt, dw, xa, xb)
    return combine(evtwt, dw)


def weight_matrix(arrays, spec: WeightSpec, source_ctau_cm: float | None, targets: list[float | None]) -> np.ndarray:
    """(n_targets, n_events) C-contiguous event weights; ``None`` = no lifetime factor at all.

    The lifetime-independent factors are computed once; only ``xawt*xbwt`` changes per
    target.  Row ``i`` is bitwise what LLPCombine's ``evtrwt`` would be for target ``i``.
    ``spec`` must not itself carry lifetime settings (the scan owns them), so lifetime
    weights can never be applied twice.
    """
    if spec.lifetime_on:
        raise ValueError("WeightSpec already has lifetime reweighting; refusing to apply it twice")
    evtwt, dw = base_weights(arrays, spec)
    out = np.empty((len(targets), len(evtwt)), dtype=np.float64)
    for i, t in enumerate(targets):
        if t is None:
            out[i] = combine(evtwt, dw)
        else:
            if source_ctau_cm is None:
                raise ValueError("lifetime target given without a source lifetime")
            xa = lifetime_weight(arrays["Xa_ctau"], source_ctau_cm, t)
            xb = lifetime_weight(arrays["Xb_ctau"], source_ctau_cm, t)
            out[i] = combine(evtwt, dw, xa, xb)
    return out


# ----------------------------------------------------------------------------- invalid-input policy
#
# Event-level anomaly bits (int8 mask; recorded for every event, never silently "fixed").
#
#   bit  class                 condition                                  lifetime weight w(t)
#   1    ctau_nonfinite        Xa_ctau or Xb_ctau is NaN or +-inf         NaN/inf for EVERY lifetime target,
#                                                                         including the direct one (exp(nan*0))
#   2    ctau_negative         Xa_ctau or Xb_ctau < 0 (finite or -inf)    finite, but outside the support of the
#                                                                         exponential decay density: unphysical
#   4    ctau_zero             Xa_ctau or Xb_ctau == 0                    finite and well defined: w = tau0/tau.
#                                                                         Physically allowed (the density peaks at
#                                                                         t = 0); an exact float zero is unusual
#                                                                         for continuous generation and is recorded,
#                                                                         but there is no generator-specific reason
#                                                                         to treat it as a sentinel, so it is VALID
#   8    decay_flags_none      no Evt_is{ZZ,GZ,GG} flag set               lifetime weight unaffected; LLPCombine's
#                                                                         DecayReweight returns 0 (event weight 0)
#   16   decay_flags_multiple  more than one Evt_is* flag set             DecayReweight: the last flag wins
#
# What each ``invalid_ctau_policy`` does with an event in a selection (POLICY_ACTIONS below):
#
#   class                 llpcombine (default)          exclude                       fail
#   ctau_nonfinite        weighted as LLPCombine -> the  removed from every variant    run fails
#                         weight is non-finite -> run   (incl. noreweight)
#                         fails (NONFINITE_WEIGHT rule)
#   ctau_negative         weighted as LLPCombine        removed from every variant    run fails
#   ctau_zero             weighted (valid)              weighted (valid)              weighted (valid)
#   decay_flags_none      weight 0, as LLPCombine       removed from every variant    run fails
#   decay_flags_multiple  last flag wins, as LLPCombine removed from every variant    run fails
#
# NONFINITE_WEIGHT rule (all policies): after the policy is applied, any event that enters any
# selection with a non-finite weight in any variant (whatever the cause: lifetime inputs, an
# exp overflow, a non-finite evtFillWgt) makes the task fail with a per-event report; NaN/inf is
# never accumulated into yields or covariances.  Under ``llpcombine`` the weights of every finite
# event are bitwise those of ``ReweightTool``.
ANOM_NONFINITE = 1
ANOM_NEGATIVE = 2
ANOM_ZERO = 4
ANOM_DECAY_NONE = 8
ANOM_DECAY_MULTI = 16
ANOM_DECAYFLAGS = ANOM_DECAY_NONE | ANOM_DECAY_MULTI  # "decay_flags_not_one" (backward-compatible union)
ANOM_NAMES = {ANOM_NONFINITE: "ctau_nonfinite", ANOM_NEGATIVE: "ctau_negative", ANOM_ZERO: "ctau_zero",
              ANOM_DECAY_NONE: "decay_flags_none", ANOM_DECAY_MULTI: "decay_flags_multiple"}
# (class name, bit mask) of the per-selection / per-file anomaly counters, in column order
ANOM_CLASSES = (("any", 0x7F), ("ctau_nonfinite", ANOM_NONFINITE), ("ctau_negative", ANOM_NEGATIVE),
                ("ctau_zero", ANOM_ZERO), ("decay_flags_not_one", ANOM_DECAYFLAGS),
                ("decay_flags_none", ANOM_DECAY_NONE), ("decay_flags_multiple", ANOM_DECAY_MULTI))
# per policy: bits that remove a selected event from every variant / that make the run fail
INVALID_INPUT_BITS = ANOM_NONFINITE | ANOM_NEGATIVE | ANOM_DECAY_NONE | ANOM_DECAY_MULTI
POLICY_ACTIONS = {
    "llpcombine": dict(exclude=0, fail=0),
    "exclude": dict(exclude=INVALID_INPUT_BITS, fail=0),
    "fail": dict(exclude=0, fail=INVALID_INPUT_BITS),
}
INVALID_LIFETIME_WEIGHT = ANOM_NONFINITE | ANOM_NEGATIVE  # inputs for which w(t) is not a valid lifetime weight


def policy_description() -> list[dict]:
    """Machine-readable form of the table above (written to run_config.json)."""
    rows = []
    for bit, name in ANOM_NAMES.items():
        rows.append(dict(cls=name, bit=bit, anomalous=True,
                         lifetime_weight_valid=not bool(bit & INVALID_LIFETIME_WEIGHT),
                         **{pol: ("fail" if a["fail"] & bit else "exclude" if a["exclude"] & bit else
                                  ("fail_nonfinite_weight" if bit == ANOM_NONFINITE else "keep"))
                            for pol, a in POLICY_ACTIONS.items()}))
    return rows


def lifetime_anomalies(arrays) -> np.ndarray:
    """int8 bit mask per event (0 = valid).  Requires Xa_ctau, Xb_ctau and the Evt_is* flags."""
    xa = np.asarray(arrays["Xa_ctau"], dtype=np.float64)
    xb = np.asarray(arrays["Xb_ctau"], dtype=np.float64)
    code = np.zeros(len(xa), dtype=np.int8)
    fin = np.isfinite(xa) & np.isfinite(xb)
    code[~fin] |= ANOM_NONFINITE
    with np.errstate(invalid="ignore"):
        code[(xa < 0) | (xb < 0)] |= ANOM_NEGATIVE
        code[(xa == 0) | (xb == 0)] |= ANOM_ZERO
    nflag = sum((np.asarray(arrays[b]) != 0).astype(np.int8) for b in DECAY_BRANCHES)
    code[nflag == 0] |= ANOM_DECAY_NONE
    code[nflag > 1] |= ANOM_DECAY_MULTI
    return code


def anomaly_names(code: int) -> str:
    return "|".join(n for b, n in ANOM_NAMES.items() if code & b) or "none"


def weight_spec_from_config(raw_yaml: dict, luminosity: float, defaults: dict) -> WeightSpec:
    """Apply main.cpp's default logic to the analysis YAML."""
    dw = raw_yaml.get("decayWeights") or {}
    lw = raw_yaml.get("lifetimeWeights") or {}
    tz, tg = dw.get("targetZrate", -1), dw.get("targetGrate", -1)
    if tz < 0 and tg < 0:
        tz, tg = defaults["target_z"], defaults["target_g"]
    sz, sg = dw.get("sampleZrate", -1), dw.get("sampleGrate", -1)
    if sz < 0:
        sz = defaults["sample_z"]
    if sg < 0:
        sg = defaults["sample_g"]
    return WeightSpec(
        luminosity=luminosity,
        sample_z=sz,
        sample_g=sg,
        target_z=tz,
        target_g=tg,
        sample_lifetime=lw.get("sampleLifetime"),
        target_lifetime=lw.get("targetLifetime"),
    )
