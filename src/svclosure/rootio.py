"""ROOT file access (uproot).  The only module that touches files.

Remote ``root://`` URLs are opened with the native XRootD source
(``uproot.source.xrootd.XRootDSource``) instead of uproot's default fsspec-xrootd
source.  On this machine (hep conda env, macOS arm64, uproot 5.7 / fsspec-xrootd
0.5.5) the fsspec source fails on close / hangs at exit whenever PyROOT has also
opened a remote file in the same process; the native source works in all
orders.  PyROOT is in any case only used in a separate validation process.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Iterator

import awkward as ak
import uproot

def _open_kwargs(path: str, timeout: float) -> dict:
    if path.startswith("root://"):
        return {"handler": uproot.source.xrootd.XRootDSource, "timeout": timeout}
    return {}


def open_file(path: str, timeout: float = 60.0):
    return uproot.open(path, **_open_kwargs(path, timeout))


@dataclass
class FileInfo:
    path: str
    ok: bool
    error: str | None = None
    tree_entries: int | None = None
    config_entries: int | None = None
    config: dict = field(default_factory=dict)  # entry 0 of kuSkimConfigTree, as LLPCombine reads it
    config_sums: dict = field(default_factory=dict)  # sums over all config entries (diagnostic)
    branches: list[str] = field(default_factory=list)
    extra_config: dict = field(default_factory=dict)
    file_bytes: int | None = None
    open_seconds: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def inspect_file(path: str, tree: str, config_tree: str, timeout: float = 60.0) -> FileInfo:
    """Event-tree schema and size via uproot.

    kuSkimConfigTree is *not* read here: in the skims_v52 signal files its branches
    claim 200 entries while only 101 basket entries exist (TTree::GetEntries() = 100),
    which uproot rejects.  Its metadata is attached by :func:`read_config_metadata`
    (PyROOT, as LLPCombine reads it).
    """
    t0 = time.perf_counter()
    try:
        with open_file(path, timeout) as f:
            if config_tree not in f:
                raise KeyError(f"no {config_tree}")
            t = f[tree]
            return FileInfo(
                path=path,
                ok=True,
                tree_entries=int(t.num_entries),
                branches=sorted(t.keys()),
                file_bytes=int(f.file.source.num_bytes) if hasattr(f.file.source, "num_bytes") else None,
                open_seconds=time.perf_counter() - t0,
            )
    except Exception as exc:  # recorded in the manifest, never silently dropped
        return FileInfo(path=path, ok=False, error=f"{type(exc).__name__}: {exc}", open_seconds=time.perf_counter() - t0)


def read_config_metadata(paths: list[str], n_procs: int = 4, timeout_s: int = 1800) -> dict[str, dict]:
    """kuSkimConfigTree metadata via PyROOT, in separate processes (never in the uproot process)."""
    import json
    import os
    import subprocess
    import sys
    import tempfile
    from pathlib import Path

    if not paths:
        return {}
    src = str(Path(__file__).resolve().parents[1])
    env = dict(os.environ, PYTHONPATH=src + os.pathsep + os.environ.get("PYTHONPATH", ""))
    n_procs = max(1, min(n_procs, len(paths)))
    groups = [paths[i::n_procs] for i in range(n_procs)]
    out: dict[str, dict] = {}
    with tempfile.TemporaryDirectory() as td:
        procs = []
        for i, g in enumerate(groups):
            o = Path(td) / f"meta_{i}.json"
            cmd = [sys.executable, "-m", "svclosure.root_meta", str(o), *g]
            procs.append((o, g, subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)))
        for o, g, pr in procs:
            _, err = pr.communicate(timeout=timeout_s)
            if pr.returncode != 0 or not o.exists():
                for p in g:
                    out[p] = dict(ok=False, error=f"root_meta failed ({pr.returncode}): {err[-500:]}")
            else:
                out.update(json.loads(o.read_text()))
    return out


@dataclass
class IOStats:
    bytes_requested: int = 0
    requests: int = 0
    seconds_read: float = 0.0
    entries: int = 0
    chunks: int = 0

    def add(self, other: "IOStats") -> None:
        for k in ("bytes_requested", "requests", "seconds_read", "entries", "chunks"):
            setattr(self, k, getattr(self, k) + getattr(other, k))


def iterate_chunks(
    path: str,
    tree: str,
    branches: list[str],
    entry_start: int,
    entry_stop: int,
    step_size: int,
    stats: IOStats,
    timeout: float = 60.0,
) -> Iterator[ak.Array]:
    """Yield awkward arrays of only ``branches`` for [entry_start, entry_stop), ``step_size`` entries at a time."""
    with open_file(path, timeout) as f:
        t = f[tree]
        src = f.file.source
        b0 = getattr(src, "num_requested_bytes", 0)
        r0 = getattr(src, "num_requests", 0)
        start = entry_start
        while start < entry_stop:
            stop = min(start + step_size, entry_stop)
            t_read = time.perf_counter()
            arrays = t.arrays(branches, entry_start=start, entry_stop=stop, library="ak")
            stats.seconds_read += time.perf_counter() - t_read
            stats.entries += stop - start
            stats.chunks += 1
            yield arrays
            start = stop
        stats.bytes_requested += getattr(src, "num_requested_bytes", 0) - b0
        stats.requests += getattr(src, "num_requests", 0) - r0
