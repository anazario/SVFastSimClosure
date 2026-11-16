"""Signal event weights exactly as LLPCombine builds ``evtrwt``.

BuildFitInput::LoadSig_KeyValue:   evtwt  = evtFillWgt * Lumi
BuildFitInput::BuildReweights:     evtrwt = evtwt * decayWeight        (lifetime reweighting off)
main.cpp: with no ``decayWeights`` block in the YAML, sample and target Z/G rates
default to 0.5, so decay reweighting is *always* switched on.  With equal source
and target rates ``ReweightTool::DecayReweight`` returns 1 for events flagged
``Evt_isZZ``/``Evt_isGZ``/``Evt_isGG`` and **0** for events with none of the flags.
Statistical error: sqrt(sum evtrwt^2).
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
    """``ReweightTool::LifetimeReweight`` (float arithmetic in C++)."""
    ctau = np.asarray(ctau, dtype=np.float32)
    return ((tau_old / tau_new) * np.exp(ctau * (1.0 / tau_old - 1.0 / tau_new))).astype(np.float32)


def event_weights(arrays, spec: WeightSpec) -> np.ndarray:
    # evtFillWgt is float; float * double -> double in the RDF JIT
    lumi = float(spec.lumi_cpp())
    w = np.asarray(arrays["evtFillWgt"], dtype=np.float64) * lumi
    if spec.lifetime_on:
        w = w * lifetime_weight(arrays["Xa_ctau"], spec.sample_lifetime, spec.target_lifetime)
        w = w * lifetime_weight(arrays["Xb_ctau"], spec.sample_lifetime, spec.target_lifetime)
    if spec.decay_on:
        w = w * decay_weight(arrays["Evt_isZZ"], arrays["Evt_isGZ"], arrays["Evt_isGG"], spec)
    return w


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
