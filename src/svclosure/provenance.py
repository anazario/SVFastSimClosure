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


def tracked_sources(repo: Path) -> list[str]:
    """Every file of this package that can change a result (code, config, tools, tests)."""
    out = []
    for pat in ("src/svclosure/*.py", "configs/*.yaml", "tools/*", "tests/*.py", "pyproject.toml", "environment.yml"):
        out += [str(p.relative_to(repo)) for p in sorted(repo.glob(pat)) if p.is_file()]
    return out


def git_state(repo: str | Path, watch: list[str] | None = None, diff_hash: bool = False) -> dict:
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
    if diff_hash:  # pins a dirty tree: hash of the uncommitted diff (tracked files) + untracked file list
        diff = subprocess.run(["git", "-C", str(repo), "diff", "HEAD", "--binary"], capture_output=True).stdout
        st["uncommitted_diff_sha256"] = hashlib.sha256(diff).hexdigest() if diff else None
        st["uncommitted_diff_bytes"] = len(diff)
        raw = subprocess.run(["git", "-C", str(repo), "status", "--porcelain"], capture_output=True, text=True).stdout
        st["untracked"] = [l[3:] for l in raw.splitlines() if l.startswith("??")]
        st["modified"] = [l[3:] for l in raw.splitlines() if l and not l.startswith("??")]
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


def short_rev(state: dict) -> dict:
    """Compact revision columns appended to every result table."""
    return {"svclosure_commit": (state.get("commit") or "")[:12], "svclosure_dirty": bool(state.get("dirty")),
            "svclosure_diff_sha256": (state.get("uncommitted_diff_sha256") or "")[:12]}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
