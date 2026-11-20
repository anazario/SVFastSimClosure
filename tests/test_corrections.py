"""Focused tests of the post-scan correctness fixes.

1. invalid-input policies (every anomaly class, non-finite weights never accumulated)
2. ln-SF covariance: diagonal identity, symmetry, bounds, gaussian_ok vs sf_class, no sample mixing
3. exact top-K weight concentration for channel unions and pools; pooled-constituent diagnostics
4. the direct-lifetime reference survives any --targets override
"""
import json

import numpy as np
import pandas as pd
import pytest

from svclosure import lifetime as L
from svclosure import stats
from svclosure.lifetime import Quality
from svclosure.weights import (ANOM_DECAY_MULTI, ANOM_DECAY_NONE, ANOM_NEGATIVE, ANOM_NONFINITE, ANOM_ZERO, WeightSpec,
                               lifetime_anomalies, policy_description, weight_matrix)

# ----------------------------------------------------------------------------- 1. policies

# one event per anomaly class (+ valid events); x > 0 selects, x <= 0 does not
CASES = [  # name, Xa, Xb, (ZZ, GZ, GG), selected
    ("valid", 7.0, 3.0, (1, 0, 0), True),
    ("nonfinite_nan", np.nan, 3.0, (0, 1, 0), True),
    ("nonfinite_inf", np.inf, 3.0, (0, 1, 0), True),
    ("negative", 5.0, -12.1, (0, 0, 1), True),
    ("zero", 0.0, 4.0, (1, 0, 0), True),
    ("flags_none", 2.0, 3.0, (0, 0, 0), True),
    ("flags_multi", 2.0, 3.0, (1, 0, 1), True),
    ("negative_unselected", 5.0, -17.2, (1, 0, 0), False),
]
CODE = {"valid": 0, "nonfinite_nan": ANOM_NONFINITE, "nonfinite_inf": ANOM_NONFINITE, "negative": ANOM_NEGATIVE,
        "zero": ANOM_ZERO, "flags_none": ANOM_DECAY_NONE, "flags_multi": ANOM_DECAY_MULTI,
        "negative_unselected": ANOM_NEGATIVE}


def _policy_file(path, names, n_valid=40, seed=3):
    import uproot

    rng = np.random.default_rng(seed)
    rows = [c for c in CASES if c[0] in names]
    xa = np.r_[rng.exponential(10, n_valid), [r[1] for r in rows]].astype(np.float32)
    xb = np.r_[rng.exponential(10, n_valid), [r[2] for r in rows]].astype(np.float32)
    fl = np.r_[np.tile([[1, 0, 0]], (n_valid, 1)), np.array([r[3] for r in rows]).reshape(-1, 3)].astype(bool)
    x = np.r_[rng.uniform(0.1, 1, n_valid), [1.0 if r[4] else -1.0 for r in rows]].astype(np.float32)
    n = len(x)
    data = dict(evtFillWgt=rng.uniform(1e-3, 2e-3, n).astype(np.float32), Xa_ctau=xa, Xb_ctau=xb,
                Evt_isZZ=fl[:, 0], Evt_isGZ=fl[:, 1], Evt_isGG=fl[:, 2], x=x)
    with uproot.recreate(path) as f:
        t = f.mktree("kuSkimTree", {k: v.dtype for k, v in data.items()})
        t.extend(data)
    return data, n_valid


def _run(path, policy, targets=(5.0, 10.0, 15.0), with_ctau=True):
    from svclosure.yields import run_task

    task = dict(point="p_10", sim="fullsim", path=str(path), entry_start=0, entry_stop=10 ** 6, tree="kuSkimTree",
                bins=[dict(mode="m", name="b", expression="(x > 0)", expression_no_hlt="(x > 0)")],
                weights={"m": WeightSpec(100.0).__dict__}, is_fastsim=False, step_size=10 ** 6, timeout=30,
                trigger_branches=[], variants=[None] + list(targets), source_ctau_cm=10.0, invalid_ctau_policy=policy,
                topk=10, lifetime_branches=with_ctau)
    return run_task(task)


def test_anomaly_classes_distinguished():
    names = [c[0] for c in CASES]
    a = {"Xa_ctau": np.array([c[1] for c in CASES], np.float32), "Xb_ctau": np.array([c[2] for c in CASES], np.float32),
         "Evt_isZZ": np.array([c[3][0] for c in CASES]), "Evt_isGZ": np.array([c[3][1] for c in CASES]),
         "Evt_isGG": np.array([c[3][2] for c in CASES])}
    assert lifetime_anomalies(a).tolist() == [CODE[n] for n in names]
    tab = {r["cls"]: r for r in policy_description()}
    assert tab["ctau_zero"]["lifetime_weight_valid"] and not tab["ctau_negative"]["lifetime_weight_valid"]
    assert not tab["ctau_nonfinite"]["lifetime_weight_valid"]
    assert {r["llpcombine"] for r in tab.values()} == {"keep", "fail_nonfinite_weight"}
    assert tab["ctau_zero"]["exclude"] == tab["ctau_zero"]["fail"] == "keep"
    for c in ("ctau_negative", "decay_flags_none", "decay_flags_multiple", "ctau_nonfinite"):
        assert tab[c]["exclude"] == "exclude" and tab[c]["fail"] == "fail"
    # a zero proper decay length is reweightable: w = tau0/tau exactly
    W = weight_matrix({"evtFillWgt": np.ones(1, np.float32), "Xa_ctau": np.zeros(1, np.float32),
                       "Xb_ctau": np.zeros(1, np.float32), "Evt_isZZ": [1], "Evt_isGZ": [0], "Evt_isGG": [0]},
                      WeightSpec(1.0), 10.0, [None, 5.0])
    assert np.isfinite(W).all() and W[1, 0] == pytest.approx(W[0, 0] * 4.0)


def test_llpcombine_policy_reproduces_reference_weights_for_finite_events(tmp_path):
    keep = ["valid", "negative", "zero", "flags_none", "flags_multi", "negative_unselected"]
    data, nv = _policy_file(tmp_path / "f.root", keep)
    r = _run(tmp_path / "f.root", "llpcombine")
    assert r.error is None
    W = weight_matrix(data, WeightSpec(100.0), 10.0, [None, 5.0, 10.0, 15.0])
    sel = data["x"] > 0
    for v in range(4):  # same per-row summation as the engine: bitwise the reference
        assert r.acc.sumw[0, 0, v] == float(W[v][sel].sum())
        assert r.acc.sumw2[0, 0, v] == float(np.square(W[v][sel]).sum())
    assert r.acc.n[0, 0] == sel.sum() and r.n_excluded == 0
    assert r.anomaly_totals["ctau_negative"] == 2 and r.anomaly_in_bins["ctau_negative"] == 1
    assert r.anomaly_totals["ctau_zero"] == 1 and r.anomaly_totals["decay_flags_none"] == 1
    assert r.anomaly_totals["decay_flags_multiple"] == 1 and r.anomaly_totals["decay_flags_not_one"] == 2
    assert len(r.events) == 5 and {e["entry"] for e in r.events} == set(range(nv, nv + 6)) - {nv}


@pytest.mark.parametrize("which", ["nonfinite_nan", "nonfinite_inf"])
def test_nonfinite_weight_of_selected_event_fails_with_report(tmp_path, which):
    _policy_file(tmp_path / "f.root", ["valid", which])
    r = _run(tmp_path / "f.root", "llpcombine")
    assert r.error and "NonFiniteWeightError" in r.error and r.acc is None  # nothing accumulated
    (e,) = r.failure_events
    assert e["reason"] == "nonfinite_weight" and e["entry"] == 41 and e["selections_parity"] == ["b"]
    assert e["targets_cm"] == [None, 5.0, 10.0, 15.0] and None not in e["nonfinite_targets"]
    assert set(e["nonfinite_targets"]) <= {5.0, 10.0, 15.0} and len(e["nonfinite_targets"]) >= 1
    assert e["evtFillWgt"] > 0 and e["flags"] == "010" and not np.isfinite(e["Xa_ctau"])
    # without lifetime targets the unreweighted weight is finite: no failure, the event is only recorded
    r0 = _run(tmp_path / "f.root", "llpcombine", targets=())
    assert r0.error is None and r0.anomaly_in_bins["ctau_nonfinite"] == 1


def test_unselected_nonfinite_event_is_harmless(tmp_path):
    import uproot

    data, _ = _policy_file(tmp_path / "f.root", ["valid"])
    data = {k: np.r_[v, v[:1]] for k, v in data.items()}
    data["Xa_ctau"][-1], data["x"][-1] = np.nan, -1.0
    with uproot.recreate(tmp_path / "g.root") as f:
        t = f.mktree("kuSkimTree", {k: v.dtype for k, v in data.items()})
        t.extend(data)
    r = _run(tmp_path / "g.root", "llpcombine")
    assert r.error is None and np.isfinite(r.acc.sumw).all()
    assert r.anomaly_totals["ctau_nonfinite"] == 1 and r.anomaly_in_bins["ctau_nonfinite"] == 0


def test_exclude_policy(tmp_path):
    names = [c[0] for c in CASES]
    data, nv = _policy_file(tmp_path / "f.root", names)
    r = _run(tmp_path / "f.root", "exclude")
    assert r.error is None
    code = lifetime_anomalies(data)
    kept = (data["x"] > 0) & ((code & (ANOM_NONFINITE | ANOM_NEGATIVE | ANOM_DECAY_NONE | ANOM_DECAY_MULTI)) == 0)
    assert r.acc.n[0, 0] == kept.sum() == nv + 2  # valid + zero survive
    assert r.n_excluded == 6
    W = weight_matrix(data, WeightSpec(100.0), 10.0, [None, 5.0, 10.0, 15.0])
    for v in range(4):
        assert r.acc.sumw[0, 0, v] == float(W[v][kept].sum())


def test_fail_policy(tmp_path):
    _policy_file(tmp_path / "f.root", ["valid", "zero", "negative_unselected"])
    assert _run(tmp_path / "f.root", "fail").error is None  # zero is valid, the negative one is not selected
    _policy_file(tmp_path / "g.root", ["valid", "flags_multi"])
    r = _run(tmp_path / "g.root", "fail")
    assert "InvalidInputError" in r.error and r.failure_events[0]["reason"] == "invalid_input"
    _policy_file(tmp_path / "h.root", ["valid"])
    r = _run(tmp_path / "h.root", "llpcombine", with_ctau=False)
    assert "InvalidInputError" in r.error and "missing" in r.error


# ----------------------------------------------------------------------------- 2. covariance


META = pd.DataFrame([dict(mode="m", name=b, sxy="high", role="sr", group="g", group_display="G", order=i, kin_label="",
                          partner=None) for i, b in enumerate(("b", "lo", "empty"))])


def _cov_toy(seed=5):
    """One generated sample (source 10 cm), targets 5/10/15, three bins: plenty / low-stats / FastSim empty."""
    rng = np.random.default_rng(seed)
    rows, cross = [], []
    tg = [5.0, 10.0, 15.0]
    for b, n in (("b", 400), ("lo", 7), ("empty", 0)):
        for sim in ("fullsim", "fastsim"):
            nn = 0 if (b == "empty" and sim == "fastsim") else (n or 8)
            w0 = rng.uniform(0.5, 1.5, nn)
            W = np.vstack([w0 * rng.uniform(0.2, 3, nn), w0, w0 * rng.uniform(0.3, 2, nn)])
            top = [np.pad(np.sort(W[i])[::-1][:10], (0, max(0, 10 - nn))) for i in range(3)]
            for i, t in enumerate(tg):
                rows.append(dict(mode="m", bin=b, point="p_10", sim=sim, hlt="parity", variant=L.variant_key(t),
                                 source_ctau_cm=10.0, target_ctau_cm=t, n_raw=nn, sumw=W[i].sum(),
                                 sumw2=(W[i] ** 2).sum(), top_w=top[i]))
                for j, u in enumerate(tg):
                    cross.append(dict(mode="m", bin=b, kind="bin", point="p_10", sim=sim, hlt="parity",
                                      variant_i=L.variant_key(t), variant_j=L.variant_key(u), target_i_cm=t,
                                      target_j_cm=u, sumw_ij=float(W[i] @ W[j])))
    y = stats.concentration(pd.DataFrame(rows))
    sf = stats.sf_table(y.drop(columns="top_w"), META, Quality())
    return sf, pd.DataFrame(cross)


def test_logsf_covariance_identities_and_classification():
    sf, cross = _cov_toy()
    c = stats.logsf_covariance(cross, sf)
    chk = stats.covariance_checks(c)
    assert chk["passed"], chk
    d = c[c.defined & (c.variant_i == c.variant_j)]
    assert len(d) == 6  # b and lo, three targets each
    np.testing.assert_allclose(d.cov_ln_sf, d.ln_sf_err_i ** 2, rtol=1e-12)
    np.testing.assert_allclose(d.corr_ln_sf, 1.0, rtol=1e-12)
    m = c.merge(c, left_on=["bin", "variant_i", "variant_j"], right_on=["bin", "variant_j", "variant_i"])
    np.testing.assert_allclose(m.cov_ln_sf_x, m.cov_ln_sf_y, rtol=1e-14)
    assert (c.corr_ln_sf.dropna().abs() <= 1 + 1e-12).all()
    # low-statistics: values kept but not Gaussian-usable; undefined: NaN, not defined
    lo = c[c.bin == "lo"]
    assert lo.defined.all() and np.isfinite(lo.cov_ln_sf).all() and not lo.gaussian_ok.any()
    assert set(sf[sf.bin == "lo"].sf_class) == {"low_stats"}
    em = c[c.bin == "empty"]
    assert not em.defined.any() and em.cov_ln_sf.isna().all() and not em.gaussian_ok.any()
    good = c[c.bin == "b"]
    assert good.gaussian_ok.all()
    assert (c.gaussian_ok == (c.defined & (c.sf_class_i == "reliable") & (c.sf_class_j == "reliable"))).all()


def test_gaussian_ok_requires_reliable_not_just_status_ok():
    """status ok (N_eff >= 10) but a reweighted point below min_neff_reweighted (25): not Gaussian-usable."""
    sf, cross = _cov_toy()
    q = Quality(min_neff=10, min_neff_reweighted=10 ** 6)  # every reweighted point -> sf_class low_stats
    rows = []
    for r in sf.itertuples():
        for sim, t in (("fullsim", "full"), ("fastsim", "fast")):
            rows.append(dict(mode=r.mode, bin=r.bin, point=r.point, sim=sim, hlt=r.hlt, variant=r.variant,
                             source_ctau_cm=r.source_ctau_cm, target_ctau_cm=r.target_ctau_cm,
                             n_raw=getattr(r, f"{t}_n_raw"), sumw=getattr(r, f"{t}_sumw"), sumw2=getattr(r, f"{t}_sumw2")))
    sf2 = stats.sf_table(pd.DataFrame(rows), META, q)
    c = stats.logsf_covariance(cross, sf2)
    b = c[c.bin == "b"]
    assert (b.status_i == "ok").all() and (b.status_j == "ok").all()  # what the old flag looked at
    direct = (b.target_i_cm == 10.0) & (b.target_j_cm == 10.0)
    assert b[direct].gaussian_ok.all() and not b[~direct].gaussian_ok.any()
    assert stats.covariance_checks(c)["gaussian_ok_with_nonreliable"] == 0


def test_covariance_never_mixes_samples():
    sf, cross = _cov_toy()
    with pytest.raises(ValueError, match="duplicate"):
        stats.logsf_covariance(pd.concat([cross, cross.iloc[:1]]), sf)
    with pytest.raises(ValueError, match="pooled"):
        stats.logsf_covariance(cross.assign(point="pooled"), sf)
    bad = sf.copy()
    bad.loc[bad.variant == "tgt5cm", "source_ctau_cm"] = 50.0
    with pytest.raises(ValueError, match="mix generated samples"):
        stats.logsf_covariance(cross, bad)
    bad = sf.copy()
    bad.loc[bad.variant == "tgt5cm", "target_ctau_cm"] = 7.0
    with pytest.raises(ValueError, match="inconsistent target"):
        stats.logsf_covariance(cross, bad)


# ----------------------------------------------------------------------------- 3. exact top-K


def test_merge_top_is_exact_for_disjoint_unions():
    rng = np.random.default_rng(9)
    for _ in range(200):
        parts = [rng.pareto(1.5, rng.integers(0, 30)) for _ in range(rng.integers(1, 6))]
        K = 10
        T = np.vstack([np.pad(np.sort(p)[::-1][:K], (0, K - min(K, len(p)))) for p in parts])
        allw = np.concatenate(parts + [np.zeros(K)])
        np.testing.assert_array_equal(stats.merge_top(T), np.sort(allw)[::-1][:K])


def _bin_rows(weights_by_bin, sim="fullsim", point="p_10", variant="tgt5cm", K=10):
    rows = []
    for b, w in weights_by_bin.items():
        w = np.asarray(w, float)
        rows.append(dict(mode="m", bin=b, point=point, sim=sim, hlt="parity", variant=variant, source_ctau_cm=10.0,
                         target_ctau_cm=5.0, n_raw=len(w), sumw=w.sum(), sumw2=(w ** 2).sum(),
                         top_w=np.pad(np.sort(w)[::-1][:K], (0, max(0, K - len(w))))))
    return stats.concentration(pd.DataFrame(rows))


CH_META = pd.DataFrame([dict(mode="m", name=b, sxy="high", role="sr", group="g", group_display="G", order=i,
                             kin_label="", partner=None) for i, b in enumerate(("A", "B", "C"))])


def test_channel_top5_exact_and_threshold_applied():
    """Union top-5 equals brute force; adding per-bin top-5 fractions or keeping only w_max is wrong.

    Note: if every bin passes max_top{1,5} the union always passes too (top_k(U) <= sum_b top_k(b)
    <= f * sum_b S_b), so the failing direction is 'a bin fails but the union passes', and the
    real defect was that the union's top-5 criterion was never evaluated.
    """
    rng = np.random.default_rng(4)
    A = np.r_[np.full(3, 1.0), np.full(200, 0.05)]  # three large weights each in A, B
    B = np.r_[np.full(3, 1.0), np.full(200, 0.05)]
    C = rng.uniform(0.045, 0.055, 2000)
    y = _bin_rows({"A": A, "B": B, "C": C})
    ch = stats.channel_yields(y, CH_META)
    g = ch[ch.bin == "g"].iloc[0]
    U = np.concatenate([A, B, C])
    s2 = (U ** 2).sum()
    assert g.sumw2 == pytest.approx(s2)
    assert g.top5_sumw2_frac == pytest.approx(np.sort(U ** 2)[::-1][:5].sum() / s2, rel=1e-14)
    assert g.top1_sumw2_frac == pytest.approx(1.0 / s2, rel=1e-14)
    approx = (y.top5_sumw2_frac * y.sumw2).sum() / s2  # summing constituent top-5s: overestimates
    assert approx > g.top5_sumw2_frac * 1.1
    # A and B are each concentrated on their own (~3/3.5 > 0.5); their union with C is not (~5/12 < 0.5)
    assert (y.set_index("bin").loc[["A", "B"]].top5_sumw2_frac > 0.5).all()
    assert g.top5_sumw2_frac < 0.5 < approx

    # a channel whose old classification (w_max only) said reliable but whose top-5 share fails
    D = np.r_[np.full(5, 1.0), np.full(400, 0.05)]  # top1 = 1/6 < 0.2, top5 = 5/6 > 0.5
    E = np.full(100, 0.05)
    rows = []
    for sim in ("fullsim", "fastsim"):
        rows.append(_bin_rows({"A": D, "B": E}, sim=sim))
    y2 = pd.concat(rows, ignore_index=True)
    q = Quality(min_neff=1, min_neff_reweighted=1)
    new = stats.sf_table(stats.channel_yields(y2, CH_META), stats.channel_meta(CH_META), q)
    old_in = stats.channel_yields(y2.drop(columns=["top_w", "top3_sumw2_frac", "top5_sumw2_frac"]), CH_META)
    old = stats.sf_table(old_in, stats.channel_meta(CH_META), q)
    assert old[old.bin == "g"].sf_class.iloc[0] == "reliable"  # the defect
    r = new[new.bin == "g"].iloc[0]
    assert r.status == "weight_concentrated" and r.sf_class == "low_stats"
    assert r.full_top1_sumw2_frac < q.max_top1_sumw2_frac < q.max_top5_sumw2_frac < r.full_top5_sumw2_frac


def test_pooled_top5_exact_and_constituent_fraction():
    rng = np.random.default_rng(2)
    rows, per = [], {}
    for i, p in enumerate(("p1_10", "p2_10", "p3_10")):
        for sim in ("fullsim", "fastsim"):
            w = rng.uniform(0.5, 1, [400, 400, 12][i])
            per[(p, sim)] = w
            rows.append(_bin_rows({"A": w}, sim=sim, point=p))
    y = pd.concat(rows, ignore_index=True)
    pool = stats.pooled(y, ["p1_10", "p2_10", "p3_10"], "pool")
    for sim in ("fullsim", "fastsim"):
        U = np.concatenate([per[(p, sim)] for p in ("p1_10", "p2_10", "p3_10")])
        r = pool[pool.sim == sim].iloc[0]
        assert r.top5_sumw2_frac == pytest.approx(np.sort(U ** 2)[::-1][:5].sum() / (U ** 2).sum(), rel=1e-14)
        assert r.w_max == U.max()
    q = Quality()
    psf = stats.sf_table(pool, CH_META, q)
    point_sf = stats.sf_table(y.drop(columns="top_w"), CH_META, q)
    t = stats.pool_constituents(psf, point_sf, {"pool": ["p1_10", "p2_10", "p3_10"]}).iloc[0]
    assert t.sf_class == "reliable" and t.pooled_yield_class == "reliable"  # the pooled yield is stable ...
    assert t.n_hyp_total == 3 and t.n_hyp_reliable == 2 and t.frac_hyp_reliable == pytest.approx(2 / 3)  # ... p3 is not
    with pytest.raises(ValueError, match="twice"):
        stats.pooled(pd.concat([y, y.iloc[:1]]), ["p1_10"], "dup")


# ----------------------------------------------------------------------------- 4. direct reference


def test_targets_override_keeps_direct_references():
    s = L.from_config({"targets": {10: [1, 5, 10, 15], 50: [20, 30, 40, 50, 70]},
                       "crosscheck_targets": {50: [10], 10: [50]}}, "10:5,15;50:20,30")
    assert s.targets_for(10.0) == [5.0, 10.0, 15.0, 50.0]
    assert s.targets_for(50.0) == [10.0, 20.0, 30.0, 50.0]
    assert s.targets_for(10.0, crosscheck=False) == [5.0, 10.0, 15.0]
    assert s.targets_for_nominal(10.0, requested_only=True) == [5.0, 15.0]
    assert s.auto_direct(10.0) and s.auto_direct(50.0)
    assert [s.origin(10, t) for t in s.targets_for(10)] == ["requested", "auto_direct", "requested", "crosscheck"]
    assert [s.role(10, t) for t in s.targets_for(10)] == ["nominal", "direct", "nominal", "stress_test"]
    assert [s.role(50, t) for t in s.targets_for(50)] == ["cross_check", "nominal", "nominal", "direct"]
    d = s.to_dict()
    assert d["auto_direct_references"] == ["10", "50"] and d["targets"] == {"10": [5.0, 15.0], "50": [20.0, 30.0]}
    assert d["evaluated_targets"]["10"] == [5.0, 10.0, 15.0, 50.0]
    back = L.from_dict(json.loads(json.dumps(d)))
    assert back.targets_for(10.0) == s.targets_for(10.0) and back.quality == s.quality
    # a list that already contains the source is not duplicated
    s2 = L.from_config({}, "10:5,10")
    assert s2.targets_for(10.0) == [5.0, 10.0] and not s2.auto_direct(10.0)


def test_direct_reference_rows_present_once_and_identical(tmp_path):
    from svclosure.outputs import lifetime_checks

    s = L.from_config({}, "10:5,15")
    tg = s.targets_for(10.0)
    data, _ = _policy_file(tmp_path / "f.root", ["valid", "zero", "negative"])
    r = _run(tmp_path / "f.root", "llpcombine", targets=tg)
    assert r.error is None
    vk = ["noreweight"] + [L.variant_key(t) for t in tg]
    assert len(set(vk)) == len(vk)
    rows = [dict(mode="m", bin="b", kind="bin", point="p_10", sim=sim, hlt="parity", variant=vk[v],
                 source_ctau_cm=10.0, target_ctau_cm=([10.0] + tg)[v], n_raw=int(r.acc.n[0, 0]),
                 sumw=float(r.acc.sumw[0, 0, v]), sumw2=float(r.acc.sumw2[0, 0, v]))
            for v in range(len(vk)) for sim in ("fullsim", "fastsim")]
    y = pd.DataFrame(rows)

    class YR:
        files = pd.DataFrame()
        anomalies = pd.DataFrame()

    chk = lifetime_checks(y, YR, pd.DataFrame())["direct_target_equals_noreweight"]
    assert chk == dict(chk, rows=2, bitwise_equal=2, expected_rows=2, duplicated_direct_rows=0, passed=True)
    bad = y.copy()
    bad.loc[bad.variant == "tgt10cm", "sumw"] *= 1 + 1e-15
    assert not lifetime_checks(bad, YR, pd.DataFrame())["direct_target_equals_noreweight"]["passed"]
    missing = y[y.variant != "tgt10cm"]
    assert not lifetime_checks(missing, YR, pd.DataFrame())["direct_target_equals_noreweight"]["passed"]
