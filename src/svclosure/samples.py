"""Sample manifest: FullSim / nominally-corrected FastSim pairs discovered from SampleTool.cpp."""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field

from .llpcombine import signal_key
from .rootio import FileInfo, inspect_file, read_config_metadata

MASS_RE = re.compile(r"mGl-(\d+)_mN2-(\d+)_mN1-(\d+)_(ct[0-9p]+)")


@dataclass
class SignalPoint:
    key: str  # LLPCombine signal key, e.g. gogoGZ_2300_1300_1000_10
    mgl: int
    mn2: int
    mn1: int
    ctau_token: str  # filename token, e.g. ct0p1
    ctau_cm: str  # LLPCombine key token, e.g. "10"
    fullsim: str | None
    fastsim: str | None
    checks: dict = field(default_factory=dict)
    fullsim_info: dict | None = None
    fastsim_info: dict | None = None

    @property
    def dm(self) -> int:
        return self.mn2 - self.mn1

    @property
    def label(self) -> str:
        return f"({self.mgl}, {self.mn2}, {self.mn1}) GeV, c$\\tau$ = {self.ctau_cm} cm"

    @property
    def paired(self) -> bool:
        return self.fullsim is not None and self.fastsim is not None

    # checks that must pass for the point to enter the measurement; the rest are reported warnings
    CRITICAL = (
        "mass_and_ctau_match",
        "fullsim_readable",
        "fastsim_readable",
        "fullsim_nTotEvts_nonzero",
        "fastsim_nTotEvts_nonzero",
        "fullsim_sMCType_consistent",
        "fastsim_sMCType_consistent",
    )

    @property
    def usable(self) -> bool:
        return self.paired and all(self.checks.get(c) is True for c in self.CRITICAL)

    @property
    def warnings(self) -> list[str]:
        return [k for k, v in self.checks.items() if v is not True and k not in self.CRITICAL]

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(dm=self.dm, paired=self.paired, usable=self.usable)
        return d


def _mass_tokens(path: str) -> tuple[int, int, int, str]:
    m = MASS_RE.search(path.rsplit("/", 1)[-1])
    if not m:
        raise ValueError(f"no mass/ctau tokens in {path}")
    return int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(4)


def build_pairs(master: dict[str, list[str]], full_key: str, fast_key: str, expected_prefix: str | None) -> list[SignalPoint]:
    full = {signal_key(p): p for p in master.get(full_key, [])}
    fast = {signal_key(p): p for p in master.get(fast_key, [])}
    if not full:
        raise RuntimeError(f"SampleTool has no MasterDict['{full_key}']")
    if not fast:
        raise RuntimeError(f"SampleTool has no MasterDict['{fast_key}']")
    points = []
    for key in sorted(set(full) | set(fast), key=lambda k: tuple(int(x) if x.isdigit() else x for x in k.split("_"))):
        ref = full.get(key) or fast.get(key)
        mgl, mn2, mn1, ct = _mass_tokens(ref)
        sp = SignalPoint(key, mgl, mn2, mn1, ct, key.rsplit("_", 1)[1], full.get(key), fast.get(key))
        if sp.paired:
            fmass, smass = _mass_tokens(sp.fullsim), _mass_tokens(sp.fastsim)
            sp.checks["mass_and_ctau_match"] = fmass == smass
            fname, sname = sp.fullsim.rsplit("/", 1)[-1], sp.fastsim.rsplit("/", 1)[-1]
            sp.checks["fullsim_tag"] = "_FullMini_" in fname
            sp.checks["fastsim_nominal_sxy_tag"] = "_Fast1_SxyScaleNominal_" in sname
            if expected_prefix:
                sp.checks["same_skim_directory"] = sp.fullsim.startswith(expected_prefix) and sp.fastsim.startswith(
                    expected_prefix
                )
        else:
            sp.checks["paired"] = False
        points.append(sp)
    return points


def inspect_points(points: list[SignalPoint], tree: str, config_tree: str, concurrency: int, timeout: float) -> None:
    """Open every file once (metadata only) and attach FileInfo + metadata checks."""
    jobs = []
    for sp in points:
        for which in ("fullsim", "fastsim"):
            path = getattr(sp, which)
            if path:
                jobs.append((sp, which, path))
    # metadata reads are latency-bound and release the GIL inside XRootD: threads are fine here
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        infos = list(pool.map(lambda j: inspect_file(j[2], tree, config_tree, timeout), jobs))
    meta = read_config_metadata([j[2] for j in jobs], n_procs=concurrency)
    for (sp, which, path), info in zip(jobs, infos):
        m = meta.get(path, {"ok": False, "error": "no metadata"})
        if info.ok and m.get("ok"):
            info.config = m["entry0"]  # what LLPCombine reads (GetEntry(0))
            info.config_sums = m["sums"]
            info.config_entries = m["entries"]
            info.extra_config = dict(branch_entries=m["branch_entries"], n_distinct=m["n_distinct"])
        elif info.ok:
            info.ok, info.error = False, f"config metadata: {m.get('error')}"
        setattr(sp, f"{which}_info", info.to_dict())
    for sp in points:
        for which, want_type in (("fullsim", "not_fastsim"), ("fastsim", "fastsim")):
            info = getattr(sp, f"{which}_info")
            if info is None:
                continue
            sp.checks[f"{which}_readable"] = bool(info["ok"])
            if not info["ok"]:
                continue
            cfg = info["config"]
            sp.checks[f"{which}_nTotEvts_nonzero"] = cfg.get("nTotEvts", 0) != 0
            mct = cfg.get("sMCType")
            # LLPCombine drops Trigger_* iff sMCType == 2
            sp.checks[f"{which}_sMCType_consistent"] = (mct == 2) if want_type == "fastsim" else (mct != 2)
        fi, si = sp.fullsim_info, sp.fastsim_info
        if fi and si and fi["ok"] and si["ok"]:
            sp.checks["same_cross_section"] = abs(fi["config"].get("sCrossSection", 0) - si["config"].get("sCrossSection", -1)) <= 1e-6 * max(
                1e-30, abs(fi["config"].get("sCrossSection", 0))
            )
