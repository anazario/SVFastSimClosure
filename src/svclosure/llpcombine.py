"""Read-only access to the LLPCombine sources of truth.

Nothing here is re-typed from LLPCombine: bins, selections, associations, sample
lists, the luminosity table and the signal-key convention are all parsed from the
checked-out repository at run time.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml


# ----------------------------------------------------------------------------- analysis config


@dataclass
class AnalysisConfig:
    path: Path
    name: str
    yaml_luminosity: float | None
    signals: list[str]
    data: list[str]
    bins: dict[str, list[str]]  # bin -> flattened cut list, YAML order
    sample_lumis: dict[str, float] = field(default_factory=dict)  # "18" -> fb^-1
    split: str = "none"
    raw: dict = field(default_factory=dict, repr=False)

    def combined_cuts(self, bin_name: str) -> str:
        """Exactly ``ConfigParser::GetCombinedCuts``: ``(c0) && (c1) && ...``."""
        cuts = self.bins[bin_name]
        if not cuts:
            return ""
        return "(" + cuts[0] + ")" + "".join(" && (" + c + ")" for c in cuts[1:])


def _flatten(cut_groups) -> list[str]:
    out: list[str] = []
    for g in cut_groups:
        if isinstance(g, list):
            out.extend(_flatten(g))
        elif isinstance(g, str):
            out.append(g)
        else:
            raise ValueError(f"unexpected cut entry {g!r}")
    return out


def load_analysis_config(path: str | Path) -> AnalysisConfig:
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    analysis = raw.get("analysis", {})
    samples = raw.get("samples", {})
    bins = {}
    for name, spec in (raw.get("bins") or {}).items():
        cuts = spec["cuts"] if isinstance(spec, dict) else spec
        bins[name] = _flatten(cuts)
    sample_lumis = {}
    for k, v in (raw.get("sampleLumis") or {}).items():
        k = str(k)
        sample_lumis[k[2:] if len(k) == 4 and k.startswith("20") else k] = float(v)
    return AnalysisConfig(
        path=path,
        name=analysis.get("name", path.stem),
        yaml_luminosity=analysis.get("luminosity"),
        signals=list(samples.get("signals", [])),
        data=list(samples.get("data", [])),
        bins=bins,
        sample_lumis=sample_lumis,
        split=str(samples.get("split", "none")),
        raw=raw,
    )


# ----------------------------------------------------------------------------- fit config


@dataclass
class FitConfig:
    path: Path
    # fit type -> channel -> bins
    bin_association: dict[str, dict[str, list[str]]]
    # fit type -> CR channel -> SR channels
    channel_association: dict[str, dict[str, list[str]]]

    def channel_of(self, bin_name: str, fit: str = "shape_transfer_fit") -> str | None:
        for ch, bins in self.bin_association.get(fit, {}).items():
            if bin_name in bins:
                return ch
        return None

    def cr_channel_for(self, sr_channel: str, fit: str = "shape_transfer_fit") -> str | None:
        for cr, srs in self.channel_association.get(fit, {}).items():
            if sr_channel in srs:
                return cr
        return None


def load_fit_config(path: str | Path) -> FitConfig:
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    ba, ca = {}, {}
    for fit in ("shape_transfer_fit", "ABCD_fit"):
        block = raw.get(fit) or {}
        ba[fit] = {ch: list(b) for ch, b in (block.get("bin_association") or {}).items()}
        assoc = {}
        for cr, tgt in (block.get("channel_association") or {}).items():
            if isinstance(tgt, list):
                assoc[cr] = list(tgt)
            elif isinstance(tgt, dict):  # {SR: transfer factor} or {proc: [CRs]}
                assoc[cr] = [k for k, v in tgt.items() if not isinstance(v, list)]
        ca[fit] = assoc
    return FitConfig(path=path, bin_association=ba, channel_association=ca)


# ----------------------------------------------------------------------------- LumiMap


def load_lumi_map(llpcombine_dir: str | Path) -> dict[str, tuple[str, float]]:
    """Parse ``LumiMap::Years()`` from include/ConfigParser.h -> {"16": ("Run2", 36.32), ...}."""
    text = (Path(llpcombine_dir) / "include" / "ConfigParser.h").read_text()
    block = re.search(r"Years\(\)\s*\{(.*?)\};", text, re.S)
    if not block:
        raise RuntimeError("could not locate LumiMap::Years() in ConfigParser.h")
    years = {}
    for yy, run, lumi in re.findall(r'\{\s*"(\d{2})"\s*,\s*\{\s*"(\w+)"\s*,\s*([\d.]+)\s*\}\s*\}', block.group(1)):
        years[yy] = (run, float(lumi))
    if not years:
        raise RuntimeError("LumiMap::Years() parsed empty")
    return years


def data_year_token(key: str, lumi_map: dict[str, tuple[str, float]]) -> str:
    """``LumiMap::DataYearToken``: last two characters if they are a known year."""
    yy = key[-2:] if len(key) >= 2 else ""
    return yy if yy in lumi_map else ""


def signal_year_token(key: str, lumi_map: dict[str, tuple[str, float]]) -> str:
    """``LumiMap::SignalYearToken``: suffix after the last underscore if it is a known year."""
    if "_" not in key:
        return ""
    yy = key.rsplit("_", 1)[1]
    return yy if yy in lumi_map else ""


def resolved_luminosity(cfg: AnalysisConfig, lumi_map: dict[str, tuple[str, float]]) -> float:
    """``ResolveLuminosities``: analysis.luminosity becomes the summed lumi of the data years."""
    years = {data_year_token(d, lumi_map) for d in cfg.data} - {""}
    if not years:
        if cfg.yaml_luminosity is None:
            raise RuntimeError(f"{cfg.path}: no data years and no analysis.luminosity")
        return float(cfg.yaml_luminosity)
    return sum(lumi_map[y][1] for y in sorted(years))


# ----------------------------------------------------------------------------- SampleTool


def parse_sample_tool(llpcombine_dir: str | Path) -> dict[str, list[str]]:
    """Reconstruct ``SampleTool::MasterDict`` from src/SampleTool.cpp.

    Follows the sequential ``pathPrefix = "...";`` reassignments and
    ``MasterDict["key"] = { pathPrefix+"file", ... };`` blocks (later assignments
    to the same key overwrite earlier ones, as in C++).
    """
    text = (Path(llpcombine_dir) / "src" / "SampleTool.cpp").read_text()
    # drop comments but never touch string literals (they contain "root://")
    text = re.sub(
        r'("(?:\\.|[^"\\])*")|/\*.*?\*/|//[^\n]*',
        lambda m: m.group(1) or "",
        text,
        flags=re.S,
    )
    prefixes: dict[str, str] = {}
    master: dict[str, list[str]] = {}
    stmt_re = re.compile(
        r'(?:string\s+)?(?P<pvar>pathPrefix\w*)\s*=\s*"(?P<pval>[^"]*)"\s*;'
        r'|MasterDict\[\s*"(?P<key>[^"]+)"\s*\]\s*=\s*\{(?P<body>.*?)\}\s*;',
        re.S,
    )
    for m in stmt_re.finditer(text):
        if m.group("pvar"):
            prefixes[m.group("pvar")] = m.group("pval")
            continue
        files = []
        for pvar, fname in re.findall(r'(?:(pathPrefix\w*)\s*\+\s*)?"([^"]+)"', m.group("body")):
            files.append((prefixes[pvar] if pvar else "") + fname)
        master[m.group("key")] = files
    return master


def normalize_ctau_token(tok: str) -> str:
    """Line-by-line port of ``BFTool::NormalizeCtauToken`` (include/BuildFitTools.h).

    ct0p1 -> "10", ct0p5 -> "50" (historical 0p* branch truncates via int()),
    ct3p0 / ct-3p0 -> "300" (llround), plain digits pass through.
    """
    ctau = tok[2:] if tok.startswith("ct") else tok
    if not ctau:
        raise ValueError(f"Could not parse empty ctau token: {tok}")
    if ctau.startswith("0p"):
        num = float(ctau.replace("p", ".", 1)) * 100
        if num >= 1:
            return str(int(num))
        s = f"{num:.6f}"  # std::to_string(double)
        dec, last = s.find("."), len(s.rstrip("0"))
        return s[:last] if last > dec else s
    if ctau.startswith("-"):
        ctau = ctau[1:]
    if "p" in ctau:
        meters = float(ctau.replace("p", "."))
        return str(int(_llround(meters * 100.0)))
    if not ctau.isdigit():
        raise ValueError(f"Could not parse ctau token: {tok}")
    return ctau


def _llround(x: float) -> int:
    """C ``llround``: half away from zero."""
    import math

    return int(math.floor(x + 0.5)) if x >= 0 else -int(math.floor(-x + 0.5))


def signal_key(path: str) -> str:
    """``BFTool::GetSignalTokens``: mode_mGl_mN2_mN1_ctau (ctau in cm)."""
    name = path.rsplit("/", 1)[-1]
    mode = mgo = mn2 = mn1 = None
    ctau = "10"
    for tok in name.split("_"):
        if tok in ("gogoGZ", "gogoG", "gogoZ", "sqsqG"):
            mode = tok
        elif tok.startswith(("mGl-", "mGo-")):
            mgo = tok.split("-")[1]
        elif tok.startswith("mN2-"):
            mn2 = tok.split("-")[1]
        elif tok.startswith("mN1-"):
            mn1 = tok.split("-")[1]
        elif tok.startswith("ct"):
            ctau = normalize_ctau_token(tok)
    if None in (mode, mgo, mn2, mn1):
        raise ValueError(f"could not parse signal filename {name}")
    return f"{mode}_{mgo}_{mn2}_{mn1}_{ctau}"
