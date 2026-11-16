import numpy as np
import pandas as pd
import pytest

from svclosure import stats
from svclosure.weights import WeightSpec, decay_weight, event_weights


@pytest.mark.parametrize("rates", [(0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 1.0, 0.0), (0.5, 0.5, 0.0, 1.0), (0.3, 0.6, 0.5, 0.2)])
def test_decay_weight_matches_llpcombine_header(study, rates):
    ROOT = pytest.importorskip("ROOT")
    assert ROOT.gInterpreter.Declare(f'#include "{study.llpcombine_dir}/include/BuildFitTools.h"')
    zo, go, zn, gn = rates
    spec = WeightSpec(1.0, zo, go, zn, gn)
    flags = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 0, 0), (1, 1, 0), (0, 1, 1)]
    ours = decay_weight(*(np.array(c) for c in zip(*flags)), spec)
    ref = [ROOT.ReweightTool.DecayReweight(bool(a), bool(b), bool(c), zo, zn, go, gn) for a, b, c in flags]
    # ROOT's binomial_pdf is not bit-identical to p*p off the 0.5 point; exact at the default
    assert np.allclose(ours, ref, rtol=1e-14, atol=0)
    if rates == (0.5, 0.5, 0.5, 0.5):
        assert list(ours) == ref


def test_event_weight_definition():
    arrays = {"evtFillWgt": np.array([1e-3, 2e-3], dtype=np.float32), "Evt_isZZ": np.array([0, 0]),
              "Evt_isGZ": np.array([1, 0]), "Evt_isGG": np.array([0, 0])}
    w = event_weights(arrays, WeightSpec(423.71))
    assert w[0] == pytest.approx(float(np.float32(1e-3)) * 423.71, rel=1e-15)
    assert w[1] == 0.0  # no decay flag -> DecayReweight returns 0


def _toy():
    rows = []
    for sim, sw in (("fullsim", 2.0), ("fastsim", 4.0)):
        for hlt in ("parity", "symmetric"):
            rows.append(dict(mode="m", bin="b", point="p", sim=sim, hlt=hlt, n_raw=100, sumw=sw, sumw2=sw * sw / 100))
    meta = pd.DataFrame([dict(mode="m", name="b", sxy="high", role="sr", group="g", group_display="G", order=0,
                              kin_label="", partner=None)])
    return pd.DataFrame(rows), meta


def test_sf_and_error():
    y, meta = _toy()
    t = stats.sf_table(y, meta)
    r = t[t.hlt == "parity"].iloc[0]
    assert r.sf == pytest.approx(0.5)
    assert r.sf_err == pytest.approx(0.5 * np.sqrt(0.01 + 0.01))
    assert r.full_n_eff == pytest.approx(100)


def test_plots_smoke(tmp_path, all_bins):
    from svclosure import plotting
    from svclosure.pipeline import bin_frame

    bins, _ = all_bins
    meta = bin_frame(bins)
    rng = np.random.default_rng(1)
    rows = []
    for p in ("gogoGZ_2300_1300_1000_10", "gogoGZ_2500_2400_2300_50"):
        for b in bins:
            for sim in ("fullsim", "fastsim"):
                for hlt in ("parity", "symmetric"):
                    s = rng.uniform(0.5, 2)
                    rows.append(dict(mode=b.mode, bin=b.name, point=p, sim=sim, hlt=hlt, n_raw=50, sumw=s, sumw2=s * s / 50))
    y = pd.DataFrame(rows)
    sf = stats.sf_table(y, meta)
    sfp = stats.sf_table(stats.pooled(y, list(y.point.unique())), meta)
    pts = pd.DataFrame([dict(key="gogoGZ_2300_1300_1000_10", mgl=2300, mn2=1300, mn1=1000, dm=300, ctau_cm="10",
                             ctau_cm_num=10.0, short_label="(2300, 1300, 1000), 10 cm", usable=True),
                        dict(key="gogoGZ_2500_2400_2300_50", mgl=2500, mn2=2400, mn1=2300, dm=100, ctau_cm="50",
                             ctau_cm_num=50.0, short_label="(2500, 2400, 2300), 50 cm", usable=True)])
    made = plotting.sf_plot(sfp, tmp_path, "sf", "toy", "(13.6 TeV)")
    made += plotting.heatmap(sf, pts, tmp_path, "hm", "(13.6 TeV)")
    made += plotting.sf_vs_dm(sf, pts, tmp_path, "dm", "(13.6 TeV)")
    assert all(p.exists() and p.stat().st_size > 1000 for p in made)
