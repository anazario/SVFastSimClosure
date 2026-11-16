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


# event-level validity of the lifetime inputs (bit flags, recorded, never silently fixed)
ANOM_NONFINITE = 1  # Xa_ctau or Xb_ctau is NaN/inf
ANOM_NEGATIVE = 2  # Xa_ctau or Xb_ctau < 0  (unphysical proper decay length)
ANOM_ZERO = 4  # Xa_ctau or Xb_ctau == 0
ANOM_DECAYFLAGS = 8  # number of Evt_is{ZZ,GZ,GG} flags != 1 (DecayReweight -> 0 or last flag wins)
ANOM_NAMES = {ANOM_NONFINITE: "ctau_nonfinite", ANOM_NEGATIVE: "ctau_negative", ANOM_ZERO: "ctau_zero",
              ANOM_DECAYFLAGS: "decay_flags_not_one"}


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
    nflag = sum(np.asarray(arrays[b], dtype=np.int8) != 0 for b in DECAY_BRANCHES)
    code[nflag != 1] |= ANOM_DECAYFLAGS
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
