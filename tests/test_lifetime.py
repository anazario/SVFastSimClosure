"""Lifetime reweighting: C++ parity of the weights, configuration, covariance, engine vs RDataFrame."""
import json
import os
import subprocess
import sys
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd
import pytest

from svclosure import lifetime as L
from svclosure import stats
from svclosure.weights import WeightSpec, lifetime_anomalies, lifetime_weight, weight_matrix

SRC = Path(__file__).resolve().parents[1] / "src"
PAIRS = [(10, 1), (10, 5), (10, 10), (10, 15), (10, 50), (50, 20), (50, 30), (50, 40), (50, 50), (50, 70), (50, 10),
         (10, 3.3), (0.1, 7)]


@pytest.fixture(scope="module")
def cpp_lw(study):
    ROOT = pytest.importorskip("ROOT")
    assert ROOT.gInterpreter.Declare(f'#include "{study.llpcombine_dir}/include/BuildFitTools.h"')
    ROOT.gInterpreter.Declare("""
    #include <vector>
    std::vector<float> svc_test_lw(const std::vector<float>& c, float a, float b){
        std::vector<float> o; o.reserve(c.size());
        for (float x : c) o.push_back(ReweightTool::LifetimeReweight(x, a, b));
        return o;
    }""")
    return lambda c, a, b: np.array(ROOT.svc_test_lw(ROOT.std.vector["float"](c.tolist()), a, b), dtype=np.float32)


def test_lifetime_weight_bitwise_equal_to_llpcombine_header(cpp_lw):
    rng = np.random.default_rng(3)
    c = np.concatenate([rng.exponential(10, 50000), rng.exponential(50, 50000), rng.uniform(0, 600, 20000),
                        [0.0, -0.0, 1e-30, -17.21, -12.11, 133.4, 537.6, 1e4]]).astype(np.float32)
    for a, b in PAIRS:
        ours, ref = lifetime_weight(c, a, b), cpp_lw(c, a, b)
        fin = np.isfinite(ref)
        assert np.array_equal(np.isfinite(ours), fin), (a, b)
        assert np.array_equal(ours[fin], ref[fin]), f"{a}->{b}: {np.sum(ours[fin] != ref[fin])} differ"
        assert np.array_equal(np.isinf(ours), np.isinf(ref))


def test_direct_target_is_exactly_one():
    c = np.array([0.0, 1e-3, 7.0, 133.0, -12.0], dtype=np.float32)
    for s in (10, 50, 0.1):
        assert np.all(lifetime_weight(c, s, s) == 1.0)
    assert np.isnan(lifetime_weight(np.array([np.nan], np.float32), 10, 10))[0]


def _arrays(n=1000, seed=1):
    rng = np.random.default_rng(seed)
    flags = rng.integers(0, 3, n)
    return {"evtFillWgt": rng.uniform(1e-3, 2e-3, n).astype(np.float32),
            "Xa_ctau": rng.exponential(10, n).astype(np.float32), "Xb_ctau": rng.exponential(10, n).astype(np.float32),
            "Evt_isZZ": flags == 0, "Evt_isGZ": flags == 1, "Evt_isGG": flags == 2}


def test_weight_matrix_direct_row_equals_noreweight_and_no_double_application():
    a = _arrays()
    W = weight_matrix(a, WeightSpec(423.71), 10.0, [None, 5.0, 10.0, 15.0])
    assert W.flags.c_contiguous and W.shape == (4, 1000)
    assert np.array_equal(W[0], W[2])  # target == source is bitwise the unreweighted weight
    assert not np.array_equal(W[0], W[1])
    with pytest.raises(ValueError, match="twice"):
        weight_matrix(a, WeightSpec(423.71, sample_lifetime=10, target_lifetime=5), 10.0, [None])


def test_anomaly_flags():
    a = {"Xa_ctau": np.array([1, np.nan, 2, 0, 3], np.float32), "Xb_ctau": np.array([1, 1, -2, 1, 3], np.float32),
         "Evt_isZZ": np.array([1, 1, 1, 1, 0]), "Evt_isGZ": np.array([0, 0, 0, 0, 0]), "Evt_isGG": np.array([0, 0, 0, 0, 0])}
    assert lifetime_anomalies(a).tolist() == [0, 1, 2, 4, 8]


def test_scan_config_validation_and_cli_syntax():
    assert L.parse_targets("10:1,5,10,15;50:20,30,40,50,70") == {10.0: [1, 5, 10, 15], 50.0: [20, 30, 40, 50, 70]}
    for bad in ("10:0,5", "10:-1", "10:nan", "10:inf", "10:5,5"):
        with pytest.raises(ValueError):
            L.from_config({}, bad)
    with pytest.raises(ValueError):
        L.parse_targets("10-5")
    s = L.from_config({"targets": {10: [5, 1, 10]}})
    assert s.targets_for(10.0) == [1, 5, 10] and s.targets_for(50.0) is None
    assert L.variant_key(None) == "noreweight" and L.variant_key(2.5) == "tgt2p5cm"


def test_source_lifetime_from_file_names(study):
    from svclosure.llpcombine import parse_sample_tool
    from svclosure.samples import build_pairs

    pts = build_pairs(parse_sample_tool(study.llpcombine_dir), "gogoGZ", "gogoGZFAST", None)
    got = {L.source_lifetime(p).cm for p in pts}
    assert got == {10.0, 50.0}
    assert all(L.source_lifetime(p).ok for p in pts)
    import copy

    q = copy.copy(next(p for p in pts if "_ct0p1_" in p.fastsim))
    q.fastsim = q.fastsim.replace("_ct0p1_", "_ct0p5_")  # a mislabelled pair must be rejected
    sl = L.source_lifetime(q)
    assert not sl.ok and "FastSim" in sl.problem
    assert L.ctau_cm_from_filename("x_mN1-1_ct3p0__y.root") == 300.0


def test_regimes_are_unambiguous_for_all_samples(study):
    from svclosure.llpcombine import parse_sample_tool
    from svclosure.samples import build_pairs

    r = study.regimes
    pts = build_pairs(parse_sample_tool(study.llpcombine_dir), "gogoGZ", "gogoGZFAST", None)
    cls = [r.classify(p.mgl, p.mn2, p.mn1) for p in pts]
    assert all(unamb for _, unamb, _ in cls)
    assert {c for c, _, _ in cls} == {"compressed", "high_dm"}
    with pytest.raises(ValueError):
        L.regimes_from_config(dict(study.raw["signal_regimes"], benchmarks={"compressed": [[2300, 1300, 1000]]}))


def test_logsf_covariance_toy():
    rows, cross = [], []
    rng = np.random.default_rng(5)
    for sim in ("fullsim", "fastsim"):
        w0 = rng.uniform(0.5, 1.5, 400)
        W = np.vstack([w0, w0 * rng.uniform(0.2, 3, 400)])
        for i, vk in enumerate(("tgt10cm", "tgt5cm")):
            rows.append(dict(mode="m", bin="b", point="p_10", sim=sim, hlt="parity", variant=vk, source_ctau_cm=10.0,
                             target_ctau_cm=[10.0, 5.0][i], n_raw=400, sumw=W[i].sum(), sumw2=(W[i] ** 2).sum()))
            for j, vj in enumerate(("tgt10cm", "tgt5cm")):
                cross.append(dict(mode="m", bin="b", kind="bin", point="p_10", sim=sim, hlt="parity", variant_i=vk,
                                  variant_j=vj, target_i_cm=[10.0, 5.0][i], target_j_cm=[10.0, 5.0][j],
                                  sumw_ij=float(W[i] @ W[j])))
    meta = pd.DataFrame([dict(mode="m", name="b", sxy="high", role="sr", group="g", group_display="G", order=0,
                              kin_label="", partner=None)])
    sf = stats.sf_table(pd.DataFrame(rows), meta)
    c = stats.logsf_covariance(pd.DataFrame(cross), sf)
    diag = c[c.target_i_cm == c.target_j_cm]
    for r in diag.itertuples():
        s = sf[np.isclose(sf.target_ctau_cm, r.target_i_cm)].iloc[0]
        assert r.cov_ln_sf == pytest.approx(s.ln_sf_err ** 2, rel=1e-12)
        assert r.corr_ln_sf == pytest.approx(1.0)
    off = c[c.target_i_cm != c.target_j_cm]
    assert ((off.corr_ln_sf > 0) & (off.corr_ln_sf < 1)).all()


def test_highlow_double_ratio_covariance():
    y, o = [], []
    for sim, (fh, fl, both) in (("fullsim", (2.0, 4.0, 0.02)), ("fastsim", (1.0, 4.0, 0.0))):
        y += [dict(mode="m", bin="H", point="p", sim=sim, hlt="parity", n_raw=100, sumw=fh, sumw2=fh * fh / 100),
              dict(mode="m", bin="L", point="p", sim=sim, hlt="parity", n_raw=100, sumw=fl, sumw2=fl * fl / 100)]
        if both:
            o.append(dict(point="p", sim=sim, hlt="parity", variant="noreweight", mode_a="m", sel_a="H", kind_a="bin",
                          mode_b="m", sel_b="L", kind_b="bin", n_a=100, n_b=100, n_ab=10, sumw_ab=0.2, sumw2_ab=both))
    meta = pd.DataFrame([dict(mode="m", name="H", sxy="high", role="sr", group="g", group_display="G", order=0,
                              kin_label="", partner="L", short="SR10"),
                         dict(mode="m", name="L", sxy="low", role="lowsxy_cr", group="gl", group_display="G", order=1,
                              kin_label="", partner="H", short="SR10 low")])
    t = stats.highlow(pd.DataFrame(y), pd.DataFrame(o), meta).iloc[0]
    assert t.double_ratio == pytest.approx((2 / 1) / (4 / 4))
    var_unc = 4 * 0.01
    assert t.ln_dr_err == pytest.approx(np.sqrt(var_unc - 2 * 0.02 / (2 * 4)))
    assert not t.disjoint and t.full_n_both == 10


def test_coverage_gaps_and_disjointness():
    from svclosure import bins as B

    assert B.coverage_gaps("(rjr_Ms[0] >= 1)", ["(rjr_Ms[0] >= 1) && (rjr_Ms[0] < 2)", "(rjr_Ms[0] >= 2)"]) == []
    g = B.coverage_gaps("(rjrIsr_PtIsr > 300)", ["(rjrIsr_RIsr >= 0.4) && (rjrIsr_RIsr < 0.6)", "(rjrIsr_RIsr >= 0.6)"])
    assert g == [("rjrIsr_RIsr", (-np.inf, False, 0.4, False))]
    assert B.provably_disjoint("(x >= 1) && (x < 2)", "(x >= 2)")
    assert not B.provably_disjoint("(x >= 1) && (x <= 2)", "(x >= 2)")


def test_reference_selections(all_bins):
    from svclosure import bins as B

    bins, _ = all_bins
    refs = B.reference_selections(bins)
    assert sum(r.kind == "group_envelope" for r in refs) == 12
    assert sum(r.kind == "sxy_inclusive" for r in refs) == 20
    gapped = {r.name for r in refs if r.gaps}
    assert gapped == {"envelope:MixDel_SR", "envelope:MixDel_AnchorCR"}  # R_ISR < 0.4 is not binned there


# ----------------------------------------------------------------------------- end-to-end on a synthetic file


def _synthetic_file(path: Path, bins, n=6000, seed=11, fastsim=False, nan_xa=0):
    import uproot

    rng = np.random.default_rng(seed)
    needed = sorted({br for b in bins for br in b.branches})
    data = {}
    for br in needed:
        if br in ("rjr_Ms", "rjr_Rs", "rjrPTS"):
            grid = {"rjr_Ms": [1999, 2000, 2500, 2600, 3000, 3500], "rjr_Rs": [0.1, 0.15, 0.3, 0.35, 0.5],
                    "rjrPTS": [100, 199.9, 200]}[br]
            vals = np.where(rng.random(n * 2) < 0.5, rng.choice(grid, n * 2), rng.uniform(0, 1.3 * max(grid), n * 2))
            data[br] = ak.unflatten(vals.astype(np.float32), np.full(n, 2))
        elif br.startswith("rjrIsr_RIsr"):
            data[br] = np.where(rng.random(n) < 0.5, rng.choice([0.3, 0.4, 0.6, 0.75, 0.9], n), rng.uniform(0.2, 1, n)).astype(np.float32)
        elif br.startswith("rjrIsr_PtIsr") or br == "selCMet":
            data[br] = np.where(rng.random(n) < 0.3, rng.choice([150, 300, 500, 900], n), rng.uniform(0, 1200, n)).astype(np.float32)
        elif br.startswith("n") or "nSVisObjects" in br:
            data[br] = rng.integers(0, 3, n).astype(np.int32)
        else:
            data[br] = rng.random(n) < 0.6
    xa = rng.exponential(10, n).astype(np.float32)
    xb = rng.exponential(10, n).astype(np.float32)
    xb[:3] = [-12.1, 0.0, 140.0]  # anomalies the real files contain
    xa[3:3 + nan_xa] = np.nan  # optional: non-finite proper decay lengths (fatal once selected and reweighted)
    flags = rng.integers(0, 4, n)  # 3 = no flag -> DecayReweight 0
    data.update(evtFillWgt=rng.uniform(1e-3, 2e-3, n).astype(np.float32), Xa_ctau=xa, Xb_ctau=xb,
                Evt_isZZ=flags == 0, Evt_isGZ=flags == 1, Evt_isGG=flags == 2)
    with uproot.recreate(path) as f:
        f["kuSkimTree"] = data
        cfg = {"sMCType": np.array([2 if fastsim else 0], np.int32), "nTotEvts": np.array([n], np.int32),
               "sCrossSection": np.array([1.0], np.float32)}
        t = f.mktree("kuSkimConfigTree", {k: v.dtype for k, v in cfg.items()})  # a TTree, as in the skims
        t.extend(cfg)
    return n


def test_engine_matches_rdataframe_with_lifetime_targets(study, all_bins, tmp_path):
    pytest.importorskip("ROOT")
    from svclosure import bins as B
    from svclosure.pipeline import _bin_payload, compare_backends
    from svclosure.yields import make_tasks, merge, run_task

    bins, _ = all_bins
    targets = [1.0, 5.0, 10.0, 15.0]
    refs = B.reference_selections(bins)
    for fastsim in (False, True):
        path = tmp_path / f"syn_{int(fastsim)}.root"
        n = _synthetic_file(path, bins, fastsim=fastsim)
        w = {a.mode: a.weights.__dict__ for a in study.analyses}
        common = dict(bins=_bin_payload(bins, refs), weights=w, is_fastsim=fastsim, tree="kuSkimTree", step_size=1700,
                      timeout=30, trigger_branches=[], variants=[None] + targets, source_ctau_cm=10.0, topk=10)
        res = [run_task(t) for t in make_tasks("p", "x", str(path), n, 2500, common)]
        assert not [r.error for r in res if r.error]
        m = merge(res)[("p", "x")]
        acc, comp = m["acc"], m["compiled"]
        rows = []
        vk = ["noreweight"] + [L.variant_key(t) for t in targets]
        for s, cb in enumerate(comp):
            if cb.kind != "bin":
                continue
            for h, hm in enumerate(("parity", "symmetric")):
                for v in range(len(vk)):
                    rows.append(dict(mode=cb.mode, bin=cb.name, hlt=hm, variant=vk[v], n_raw=int(acc.n[s, h]),
                                     sumw=float(acc.sumw[s, h, v]), sumw2=float(acc.sumw2[s, h, v])))
                # diagonal of the cross moments is sum w^2
                np.testing.assert_allclose(np.diag(acc.cross[s, h]), acc.sumw2[s, h], rtol=1e-12)
        a0 = study.analyses[0].weights
        spec = dict(path=str(path), tree="kuSkimTree", llpcombine_include=str(study.llpcombine_dir / "include"),
                    threads=0, bins=_bin_payload(bins), dump_weights=False,
                    weights=dict(lumi_cpp=a0.lumi_cpp(), sample_z=a0.sample_z, sample_g=a0.sample_g,
                                 target_z=a0.target_z, target_g=a0.target_g),
                    lifetime=dict(source_cm=10.0, targets=targets, variant_keys=vk[1:]))
        sp, op = tmp_path / "spec.json", tmp_path / "out.json"
        sp.write_text(json.dumps(spec))
        env = dict(os.environ, PYTHONPATH=str(SRC))
        r = subprocess.run([sys.executable, "-m", "svclosure.rdf_backend", str(sp), str(op)], env=env,
                           capture_output=True, text=True, timeout=900)
        assert r.returncode == 0, r.stderr[-3000:]
        c = compare_backends(pd.DataFrame(rows), json.loads(op.read_text()), rel_tol=1e-12)
        assert c.n_match.all()
        bad = c[~c.ok]
        assert bad.empty, bad[["bin", "hlt", "variant", "up_sumw", "rdf_sumw", "sumw_reldiff"]].head().to_string()
        assert (c.up_n > 0).sum() > 100  # the synthetic events populate the bins
        # within-channel bins never share events
        names = [cb.name for cb in comp]
        group = {(b.mode, b.name): b.group for b in bins}
        for h in range(2):
            for i, a in enumerate(comp):
                for j, b in enumerate(comp):
                    if i < j and a.kind == b.kind == "bin" and a.mode == b.mode and group[(a.mode, a.name)] == group[(b.mode, b.name)]:
                        assert acc.ov_n[h, i, j] == 0, (names[i], names[j])
