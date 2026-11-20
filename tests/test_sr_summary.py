"""SR closure summary figures: row selection (SR only, six groups, nominal targets) and drawing smoke tests."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from svclosure import lifetime as L
from svclosure import lifetime_plots, plotting
from svclosure.plotting import SR_SUMMARY_GROUPS, sr_summary_rows

ROOT_DIR = Path(__file__).resolve().parents[1]
RUN = ROOT_DIR / "results" / "20261120-001429_lifetime"
REGIMES = SimpleNamespace(mode_of={"compressed": "compressed", "high_dm": "noncompressed"})
REGIME_OF = {"compressed": "compressed", "noncompressed": "high_dm"}
TARGETS = {10.0: [1.0, 5.0, 10.0, 15.0], 50.0: [20.0, 30.0, 40.0, 50.0, 70.0]}
CROSS = {10.0: [50.0], 50.0: [10.0]}


def _row(mode, b, role, src, sxy="high", hlt="parity", pool_regime=None, tgt=None, troles=None, **kw):
    reg = pool_regime or REGIME_OF[mode]
    r = dict(mode=mode, bin=b, role=role, sxy=sxy, hlt=hlt, pool=f"{reg}_ct{src:g}", source_ctau_cm=src,
             group=b.split(":")[0], group_display=b.split(":")[0], sf=0.9, sf_err=0.05, sf_class="reliable",
             status="ok", n_hyp_total=8, n_hyp_reliable=8, frac_hyp_reliable=1.0)
    if tgt is not None:
        r.update(target_ctau_cm=tgt, target_role=troles)
    r.update(kw)
    return r


def _table(lifetime: bool) -> pd.DataFrame:
    """The six SR groups plus every kind of row the figures must NOT use."""
    rows = []
    for k, (mode, b) in enumerate(SR_SUMMARY_GROUPS):
        other = "high_dm" if mode == "compressed" else "compressed"
        for src in (10.0, 50.0):
            pairs = ([(t, "direct" if t == src else "nominal") for t in TARGETS[src]]
                     + [(t, "cross_check" if src == 50 else "stress_test") for t in CROSS[src]]) if lifetime \
                else [(None, None)]
            for t, tr in pairs:
                sf = 1 + 0.01 * k + (0 if t is None else 0.001 * t) + 0.1 * (src == 50)
                rows.append(_row(mode, b, "channel_sr", src, tgt=t, troles=tr, sf=sf, sf_err=0.01 * (k + 1)))
                rows.append(_row(mode, b.split(":")[0], "channel", src, tgt=t, troles=tr, sf=99))  # SR + anchor
                rows.append(_row(mode, b, "channel_sr", src, sxy="low", tgt=t, troles=tr, sf=98))  # low-S_xy
                rows.append(_row(mode, b, "channel_sr", src, hlt="symmetric", tgt=t, troles=tr, sf=97))
                rows.append(_row(mode, b, "channel_sr", src, pool_regime=other, tgt=t, troles=tr, sf=96))  # wrong regime
    rows.append(_row("noncompressed", "Ch3CRAnchor:CR", "anchor", 10.0, sf=95, tgt=10.0 if lifetime else None,
                     troles="direct" if lifetime else None))
    return pd.DataFrame(rows)


@pytest.mark.parametrize("lifetime", [False, True])
def test_only_regime_matched_high_sxy_sr_rows(lifetime):
    t = _table(lifetime)
    d = sr_summary_rows(t, REGIMES, targets=TARGETS if lifetime else None)
    assert set(d.role) == {"channel_sr"} and set(d.sxy) == {"high"} and set(d.hlt) == {"parity"}
    assert (d.sf < 2).all()  # none of the CR / anchor / channel / low-S_xy / symmetric / wrong-pool rows
    assert list(dict.fromkeys(zip(d["mode"], d.bin))) == list(SR_SUMMARY_GROUPS)
    assert len(SR_SUMMARY_GROUPS) == 6 and len({b for _, b in SR_SUMMARY_GROUPS}) == 6
    assert all(p.startswith(REGIME_OF[m] + "_ct") for m, p in zip(d["mode"], d.pool))
    n_per_group = (sum(map(len, TARGETS.values())) if lifetime else 2)
    assert len(d) == 6 * n_per_group


def test_missing_or_extra_group_is_an_error():
    t = _table(False)
    with pytest.raises(ValueError, match="expected groups"):
        sr_summary_rows(t[t.bin != "MixDel_SR:SR"], REGIMES)
    extra = t[t.bin == "SVonly_SR:SR"].assign(bin="Other_SR:SR")
    with pytest.raises(ValueError, match="expected groups"):
        sr_summary_rows(pd.concat([t, extra]), REGIMES)


def test_lifetime_rows_nominal_targets_only_with_correct_sources():
    d = sr_summary_rows(_table(True), REGIMES, targets=TARGETS)
    assert set(d.target_role) == {"direct", "nominal"}
    for src, g in d.groupby("source_ctau_cm"):
        for _, gg in g.groupby(["mode", "bin"]):
            assert sorted(gg.target_ctau_cm) == TARGETS[src]
            assert (gg.target_role == "direct").sum() == 1
            assert gg.loc[gg.target_role == "direct", "target_ctau_cm"].iloc[0] == src
    # a nominal target that is not configured for its source is rejected (wrong-source guard)
    bad = _table(True)
    bad.loc[(bad.target_role == "nominal") & (bad.source_ctau_cm == 10) & (bad.target_ctau_cm == 5), "target_ctau_cm"] = 30
    with pytest.raises(ValueError, match="configured nominal targets"):
        sr_summary_rows(bad, REGIMES, targets=TARGETS)
    # a pool whose lifetime does not match the source is rejected
    bad = _table(True)
    bad.loc[bad.index[0], "source_ctau_cm"] = 50.0
    with pytest.raises(ValueError):
        sr_summary_rows(bad, REGIMES, targets=TARGETS)


def test_values_uncertainties_and_classes_are_copied_unchanged():
    t = _table(True)
    t.loc[(t.target_ctau_cm == 1.0) & (t.role == "channel_sr"), ["sf_class", "status"]] = ["low_stats", "weight_concentrated"]
    t.loc[(t.target_ctau_cm == 5.0) & (t.role == "channel_sr"), ["frac_hyp_reliable", "n_hyp_reliable"]] = [0.5, 4]
    t.loc[(t.target_ctau_cm == 15.0) & (t.role == "channel_sr"), ["frac_hyp_reliable", "n_hyp_reliable"]] = [0.75, 6]
    d = sr_summary_rows(t, REGIMES, targets=TARGETS)
    key = ["mode", "bin", "source_ctau_cm", "target_ctau_cm"]
    ref = t[(t.role == "channel_sr") & (t.sxy == "high") & (t.hlt == "parity") & (t.sf < 2)].set_index(key)
    m = d.set_index(key)
    for c in ("sf", "sf_err", "sf_class", "status", "frac_hyp_reliable", "n_hyp_reliable", "pool"):
        assert (m[c] == ref.loc[m.index, c]).all(), c
    cls = dict(zip(d.target_ctau_cm, d.display_class))
    assert cls[1.0] == "low_stats" and cls[5.0] == "reliable_partial" and cls[15.0] == "reliable"
    assert set(d.loc[d.sf_class == "reliable", "display_class"]) <= {"reliable", "reliable_partial"}


def test_sr_ylim_never_drops_reliable_points():
    sf, err = np.array([0.45, 3.5, 24.0, 0.9]), np.array([0.03, 0.2, 12.0, 0.3])
    cls = np.array(["reliable", "reliable", "low_stats", "low_stats"])
    for log in (False, True):
        lo, hi = plotting.sr_ylim(sf, err, cls, log=log)
        assert lo <= 0.42 and hi >= 3.7 and lo <= 0.6 and hi >= 1.2  # reliable bars + low-stats 0.9 +- 0.3
        assert hi < 24.0  # the wild low-stats value becomes a labelled arrow instead of stretching the axis


def test_figures_smoke(tmp_path):
    q = L.Quality()
    scan = SimpleNamespace(targets=TARGETS, quality=q)
    made = plotting.sf_sr_regions_direct(_table(False), REGIMES, tmp_path, "direct", "(13.6 TeV)")
    t = _table(True)
    t.loc[(t.target_ctau_cm == 1.0), ["sf", "sf_err", "sf_class"]] = [20.0, 9.0, "low_stats"]
    made += lifetime_plots.sf_sr_regions_vs_ctau(t, REGIMES, scan, tmp_path, "vs_ctau", "(13.6 TeV)")
    assert len(made) == 4 and all(p.stat().st_size > 1000 for p in made)


@pytest.mark.skipif(not (RUN / "lifetime_sf_channel_pooled_regime.csv").exists(), reason="reference run not present")
def test_reference_run_direct_points_match_direct_tables():
    import json

    scan = L.from_dict(json.loads((RUN / "run_config.json").read_text())["lifetime_scan"])
    a = sr_summary_rows(pd.read_csv(RUN / "sf_channel_pooled_regime.csv"), REGIMES)
    b = sr_summary_rows(pd.read_csv(RUN / "lifetime_sf_channel_pooled_regime.csv"), REGIMES, targets=scan.targets)
    assert len(a) == 12 and len(b) == 6 * 9
    assert set(b.target_role) == {"direct", "nominal"}
    direct = b[b.target_role == "direct"].set_index(["mode", "bin", "source_ctau_cm"]).sort_index()
    ref = a.set_index(["mode", "bin", "source_ctau_cm"]).sort_index()
    assert list(direct.index) == list(ref.index)
    for c in ("sf", "sf_err", "full_sumw", "fast_sumw"):
        np.testing.assert_allclose(direct[c], ref[c], rtol=1e-12)
    assert (direct.sf_class == ref.sf_class).all()
