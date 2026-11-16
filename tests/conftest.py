import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG = ROOT_DIR / "configs" / "study.yaml"


@pytest.fixture(scope="session")
def study():
    from svclosure.pipeline import load_study

    s = load_study(CONFIG)
    if not s.llpcombine_dir.exists():
        pytest.skip(f"LLPCombine checkout not found at {s.llpcombine_dir}")
    return s


@pytest.fixture(scope="session")
def all_bins(study):
    from svclosure.pipeline import resolve_all_bins

    bins, problems = resolve_all_bins(study)
    return bins, problems


@pytest.fixture(scope="session")
def cpp_dumper(tmp_path_factory, study):
    """Compile LLPCombine's real ConfigParser into a cut dumper (outside the LLPCombine tree)."""
    cxx = shutil.which("clang++", path="/usr/bin") or shutil.which("c++")
    if not cxx:
        pytest.skip("no C++ compiler")
    d = tmp_path_factory.mktemp("cppcuts")
    src = d / "dump_cuts.cpp"
    src.write_text((ROOT_DIR / "tools" / "dump_cuts.cpp").read_text())
    exe = d / "dump_cuts"
    r = subprocess.run([cxx, "-std=c++17", "-O1", f"-I{study.llpcombine_dir / 'include'}", str(src),
                        str(study.llpcombine_dir / "src" / "ConfigParser.cpp"), "-o", str(exe)],
                       capture_output=True, text=True, env={k: v for k, v in os.environ.items() if k != "LDFLAGS"})
    if r.returncode != 0:
        pytest.skip(f"could not compile ConfigParser: {r.stderr[-500:]}")
    return exe
