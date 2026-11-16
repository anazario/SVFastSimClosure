"""Run provenance: git revisions (incl. dirty state), environment, timing."""
from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _git(repo: Path, *args: str) -> str:
    try:
        return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception as exc:  # pragma: no cover
        return f"<error: {exc}>"


def git_state(repo: str | Path, watch: list[str] | None = None) -> dict:
    repo = Path(repo).expanduser()
    if not (repo / ".git").exists():
        return {"path": str(repo), "git": False}
    status = _git(repo, "status", "--porcelain")
    st = {
        "path": str(repo),
        "git": True,
        "commit": _git(repo, "rev-parse", "HEAD"),
        "branch": _git(repo, "rev-parse", "--abbrev-ref", "HEAD"),
        "describe": _git(repo, "describe", "--always", "--dirty"),
        "dirty": bool(status),
        "n_modified": sum(1 for l in status.splitlines() if not l.startswith("??")),
        "n_untracked": sum(1 for l in status.splitlines() if l.startswith("??")),
        "remote": _git(repo, "config", "--get", "remote.origin.url"),
    }
    if watch:  # the exact files this study reads: their own status and content hash
        files = {}
        for rel in watch:
            p = repo / rel
            files[rel] = {
                "status": _git(repo, "status", "--porcelain", "--", rel) or "clean",
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None,
                "last_commit": _git(repo, "log", "-1", "--format=%h %cI", "--", rel),
            }
        st["files_read"] = files
    return st


def environment() -> dict:
    env = {"python": sys.version, "platform": platform.platform(), "machine": platform.machine()}
    for mod in ("uproot", "awkward", "numpy", "pandas", "matplotlib", "mplhep", "yaml", "fsspec_xrootd"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception:
            env[mod] = None
    try:
        from XRootD import client  # noqa: F401
        import XRootD

        env["XRootD_bindings"] = str(Path(XRootD.__file__).parent)
    except Exception:
        env["XRootD_bindings"] = None
    return env


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
