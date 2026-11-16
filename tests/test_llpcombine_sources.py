import json
import subprocess

import pytest

from svclosure import llpcombine as L


def test_resolved_luminosity_is_sum_of_data_years(study):
    for a in study.analyses:
        assert a.luminosity == pytest.approx(423.71, abs=1e-9)
        assert a.weights.lumi_cpp() == "423.710000"


@pytest.mark.parametrize(
    "tok,expected",
    [("ct0p1", "10"), ("ct0p5", "50"), ("ct0p07", "7"), ("ct0p001", "0.1"), ("ct3p0", "300"), ("ct-3p0", "300"),
     ("ct10", "10")],
)
def test_normalize_ctau_token_matches_cpp_port(tok, expected):
    assert L.normalize_ctau_token(tok) == expected


def test_signal_key():
    p = "root://x//a/gogoGZ_mGl-2300_mN2-1300_mN1-1000_ct0p1__SVHPM100_Fast1_SxyScaleNominal_v37__llpCombineSkim.root"
    assert L.signal_key(p) == "gogoGZ_2300_1300_1000_10"


def test_sample_tool_pairs(study):
    from svclosure.samples import build_pairs

    master = L.parse_sample_tool(study.llpcombine_dir)
    pts = build_pairs(master, "gogoGZ", "gogoGZFAST", study.samples["expected_prefix"])
    assert len(pts) == 28
    assert all(p.paired for p in pts)
    assert all(p.checks["mass_and_ctau_match"] for p in pts)
    assert all(p.checks["fastsim_nominal_sxy_tag"] and p.checks["fullsim_tag"] for p in pts)
    assert all(p.checks["same_skim_directory"] for p in pts)
    assert {p.ctau_cm for p in pts} == {"10", "50"}


def test_combined_cuts_identical_to_llpcombine_parser(study, cpp_dumper):
    """Every bin's combined cut string equals what LLPCombine's own C++ ConfigParser builds."""
    for a in study.analyses:
        out = subprocess.run([str(cpp_dumper), str(a.cfg.path)], capture_output=True, text=True, check=True).stdout
        cpp, lumi = {}, None
        for line in out.splitlines():
            if line.startswith('{"bin"'):
                d = json.loads(line)
                cpp[d["bin"]] = d["cuts"]
            elif "__luminosity__" in line:
                lumi = json.loads(line)["__luminosity__"]
        assert set(cpp) == set(a.cfg.bins)
        assert {b: a.cfg.combined_cuts(b) for b in a.cfg.bins} == cpp
        assert lumi == pytest.approx(a.luminosity)
