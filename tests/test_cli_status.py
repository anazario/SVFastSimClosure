"""End-to-end CLI on local synthetic samples: run status, fatal failures, direct references, plot config."""
import json
from pathlib import Path

import pytest

from svclosure import cli
from svclosure import pipeline as P

from test_lifetime import _synthetic_file

MASSES = [(2300, 1300, 1000), (2300, 2200, 2100)]  # one High-Delta-m, one Compressed hypothesis


@pytest.fixture
def synthetic_points(study, all_bins, tmp_path, monkeypatch):
    pytest.importorskip("ROOT")  # config metadata is read with PyROOT
    from svclosure.samples import build_pairs, inspect_points

    bins, _ = all_bins

    def make(nan_xa=0):
        d = tmp_path / f"samples_{nan_xa}"
        d.mkdir()
        master = {"gogoGZ": [], "gogoGZFAST": []}
        for i, (m, ct) in enumerate((m, ct) for m in MASSES for ct in ("ct0p1", "ct0p5")):
            for fast in (False, True):
                tag = "Fast1_SxyScaleNominal" if fast else "FullMini"
                name = f"gogoGZ_mGl-{m[0]}_mN2-{m[1]}_mN1-{m[2]}_{ct}__SVHPM100_{tag}_v37__llpCombineSkim.root"
                _synthetic_file(d / name, bins, n=2500, seed=100 + 2 * i + fast, fastsim=fast,
                                nan_xa=nan_xa if (i == 0 and not fast) else 0)
                master["gogoGZFAST" if fast else "gogoGZ"].append(str(d / name))
        pts = build_pairs(master, "gogoGZ", "gogoGZFAST", None)
        inspect_points(pts, "kuSkimTree", "kuSkimConfigTree", 2, 30)
        monkeypatch.setattr(P, "discover_points", lambda study, inspect: pts)
        return pts

    return make


def _scan(tmp_path, *extra):
    res = tmp_path / "results"
    res.mkdir(exist_ok=True)
    rc = cli.main(["--results", str(res), "lifetime-scan", "--targets", "10:5,15;50:20,30", "--workers", "1",
                   "--no-plots", *extra])
    runs = sorted(res.iterdir())
    return rc, runs[-1]


def test_complete_run_with_target_override(synthetic_points, tmp_path, monkeypatch):
    pts = synthetic_points()
    assert all(p.usable for p in pts)
    rc, run = _scan(tmp_path)
    assert rc == 0 and not run.name.endswith("_FAILED")
    st = json.loads((run / "run_status.json").read_text())
    assert st["status"] == "complete" and st["failures"] == [] and all(st["checks"].values())
    lc = json.loads((run / "lifetime_checks.json").read_text())
    assert lc["target_lifetimes"]["10"] == dict(requested=[5.0, 15.0], auto_direct_reference=True,
                                                crosscheck=[50.0], evaluated=[5.0, 10.0, 15.0, 50.0])
    assert lc["target_lifetimes"]["50"]["evaluated"] == [10.0, 20.0, 30.0, 50.0]
    d = lc["direct_target_equals_noreweight"]
    assert d["passed"] and d["rows"] == d["expected_rows"] == d["bitwise_equal"] > 0
    assert lc["covariance"]["passed"]
    cfg = json.loads((run / "run_config.json").read_text())["lifetime_scan"]
    assert cfg["auto_direct_references"] == ["10", "50"] and cfg["targets"] == {"10": [5.0, 15.0], "50": [20.0, 30.0]}
    import pandas as pd

    y = pd.read_csv(run / "lifetime_yields.csv")
    k = ["mode", "bin", "kind", "point", "sim", "hlt", "target_ctau_cm"]
    assert not y.duplicated(k).any()
    direct = y[y.target_role == "direct"]
    assert set(direct.target_origin) == {"auto_direct"} and set(direct.target_ctau_cm) == {10.0, 50.0}
    assert {"requested", "auto_direct", "crosscheck"} == set(y.target_origin)
    assert "top_w" in y.columns and (y.top_w.str.split().str.len() == 10).all()

    # plot uses the configuration saved with the run, not the current YAML defaults
    seen = {}
    monkeypatch.setattr(cli, "_plots", lambda study, run, tables, pts, per_point=True, lifetime_figs=False, scan=None:
                        seen.update(scan=scan, rl=study.raw["plot"]["rlabel"]) or [])
    assert cli.main(["plot", "--run", str(run)]) == 0
    assert seen["scan"].targets == {10.0: [5.0, 15.0], 50.0: [20.0, 30.0]}
    assert seen["scan"].targets_for(10.0) == [5.0, 10.0, 15.0, 50.0] and seen["rl"] == "(13.6 TeV)"


def test_nonfinite_weight_fails_the_run(synthetic_points, tmp_path):
    synthetic_points(nan_xa=400)
    rc, run = _scan(tmp_path)
    assert rc == 4 and run.name.endswith("_FAILED")
    st = json.loads((run / "run_status.json").read_text())
    assert st["status"] == "failed" and any("NonFiniteWeightError" in f for f in st["failures"])
    import pandas as pd

    fe = pd.read_csv(run / "diagnostics" / "failure_events.csv")
    assert len(fe) and set(fe.reason) == {"nonfinite_weight"} and fe.Xa_ctau.isna().all()
    assert {"point", "sim", "path", "entry", "selections_parity", "source_ctau_cm", "targets_cm", "evtrwt",
            "evtFillWgt", "Xb_ctau", "flags"} <= set(fe.columns)
    for name in ("yields.csv", "sf.csv", "lifetime_sf.csv", "lifetime_quality.csv", "figures", "OUTPUTS.md"):
        assert not (run / name).exists(), name  # no normal-looking result package
    with pytest.raises(SystemExit, match="refusing"):
        cli.main(["plot", "--run", str(run)])


def test_exclude_policy_completes_with_same_inputs(synthetic_points, tmp_path):
    synthetic_points(nan_xa=400)
    rc, run = _scan(tmp_path, "--invalid-ctau-policy", "exclude")
    assert rc == 0, json.loads((run / "run_status.json").read_text())
    lc = json.loads((run / "lifetime_checks.json").read_text())
    assert lc["anomaly_totals"]["n_ctau_nonfinite"] == 400 and lc["anomaly_totals"]["n_excluded"] >= 400


def test_required_check_failure_is_fatal(tmp_path):
    run = tmp_path / "x_lifetime"
    run.mkdir()
    tables = dict(overlap_summary=dict(structural_failures=[{"a": 1}], unexpected_overlaps=[], unexpected_omissions=[]),
                  lifetime_checks=dict(direct_target_equals_noreweight=dict(passed=False, bitwise_equal=1, rows=2,
                                                                            expected_rows=2, duplicated_direct_rows=0),
                                       covariance=dict(passed=True)))

    class S:
        raw = {"overlaps": {}}

    fails, checks = cli._required_checks(S, tables, scan=object())
    assert len(fails) == 2 and checks == dict(bin_structure_overlaps=False, direct_target_identity=False,
                                              covariance_consistency=True)
    code, failed = cli._fail_run(run, cli.RunFailed(5, fails))
    assert code == 5 and failed.name == "x_lifetime_FAILED" and not run.exists()
    assert json.loads((failed / "run_status.json").read_text())["status"] == "failed"


def test_missing_sample_fails_unless_partial_is_explicit(synthetic_points, tmp_path):
    pts = synthetic_points()
    pts[0].checks["fastsim_readable"] = False  # as for an unreachable / unreadable FastSim file
    rc, run = _scan(tmp_path)
    assert rc == 4 and run.name.endswith("_FAILED")
    assert any("required sample missing" in f for f in json.loads((run / "run_status.json").read_text())["failures"])
    rc, run = _scan(tmp_path, "--allow-missing-points")
    st = json.loads((run / "run_status.json").read_text())
    assert rc == 0 and st["status"] == "partial" and len(st["omitted_points"]) == 1 and pts[0].key in st["omitted_points"][0]
