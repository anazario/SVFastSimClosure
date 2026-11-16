import awkward as ak
import numpy as np
import pytest

from svclosure import expr


def ev(src, **cols):
    n = len(next(iter(cols.values())))
    return expr.evaluate(expr.parse(src), expr.EvalContext(arrays=cols, n_events=n))


def test_all_config_bins_parse(study):
    for a in study.analyses:
        for name in a.cfg.bins:
            node = expr.parse(a.cfg.combined_cuts(name))
            assert node.branches()


def test_precedence_matches_cpp():
    a = np.array([1, 1, 0, 0])
    b = np.array([1, 0, 1, 0])
    c = np.array([0, 0, 0, 1])
    # && binds tighter than ||
    assert ev("a && b || c", a=a, b=b, c=c).tolist() == [True, False, False, True]
    # ! binds tighter than ==
    assert ev("!a == 0", a=a).tolist() == [True, True, False, False]  # (!a) == 0


def test_float_branch_vs_double_literal_promotes_like_cpp():
    x = np.array([np.float32(0.15), np.float32(0.3)], dtype=np.float32)
    # float32(0.15) = 0.15000000596 > 0.15 (double) -> True in C++; float32(0.3) = 0.30000001192
    assert ev("x > 0.15", x=x).tolist() == [True, True]
    assert ev("x < 0.3", x=x).tolist() == [True, False]
    assert ev("x <= 0.15", x=x).tolist() == [False, False]


def test_index_out_of_range_is_counted_and_fails():
    arr = ak.Array([[2500.0], [], [1000.0, 3000.0]])
    ctx = expr.EvalContext(arrays={"m": arr}, n_events=3)
    assert expr.evaluate(expr.parse("m[0] >= 2000"), ctx).tolist() == [True, False, False]
    assert ctx.out_of_range == {"m[0]": 1}


def test_strip_triggers_regex():
    s = "((Trigger_A == 1)||(Trigger_B_PFHT60)) && (NotTrigger_X == 1) && (xTrigger_Y)"
    assert expr.strip_triggers(s) == "((true == 1)||(true)) && (NotTrigger_X == 1) && (xTrigger_Y)"


@pytest.mark.parametrize("src", ["a + b > 1", "f(a) > 1", "a < b < c", "a[x] > 1", "a ? b : c", "a & b"])
def test_unsupported_syntax_is_rejected(src):
    with pytest.raises(expr.UnsupportedExpression):
        expr.parse(src)


def test_variable_interval():
    node = expr.parse("(m >= 2000) && ((m < 2600) && (r >= 0.15)) && (m >= 2100 || r > 1)")
    iv = expr.variable_interval(node, "m")
    assert (iv.lo, iv.lo_closed, iv.hi, iv.hi_closed) == (2000, True, 2600, False)


def test_translation_agrees_with_rdataframe_on_synthetic_events(study, all_bins, tmp_path):
    """Every SV bin expression, original and trigger-stripped, evaluated by ROOT's C++ JIT vs by us."""
    ROOT = pytest.importorskip("ROOT")
    import uproot

    bins, _ = all_bins
    rng = np.random.default_rng(7)
    n = 20000
    needed = sorted({br for b in bins for br in b.branches})
    data = {}
    for br in needed:
        if br in ("rjr_Ms", "rjr_Rs", "rjrPTS"):
            grid = {"rjr_Ms": [1999, 2000, 2500, 2600, 2700, 2999, 3000, 3500], "rjr_Rs": [0.1, 0.15, 0.3, 0.35, 0.45, 0.5],
                    "rjrPTS": [100, 199.9, 200, 250]}[br]
            vals = np.where(rng.random(n * 2) < 0.5, rng.choice(grid, n * 2), rng.uniform(0, 1.3 * max(grid), n * 2))
            data[br] = ak.unflatten(vals.astype(np.float32), np.full(n, 2))
        elif br.startswith("rjrIsr_RIsr"):
            data[br] = np.where(rng.random(n) < 0.5, rng.choice([0.4, 0.6, 0.75, 0.9], n), rng.uniform(0.2, 1.0, n)).astype(np.float32)
        elif br.startswith("rjrIsr_PtIsr") or br == "selCMet":
            data[br] = np.where(rng.random(n) < 0.3, rng.choice([150, 300, 500, 850, 900], n), rng.uniform(0, 1200, n)).astype(np.float32)
        elif br.startswith("n") or "nSVisObjects" in br:
            data[br] = rng.integers(0, 3, n).astype(np.int32)
        else:  # flags / triggers
            data[br] = rng.random(n) < 0.6
    path = tmp_path / "synthetic.root"
    with uproot.recreate(path) as f:
        f["t"] = data
    df = ROOT.RDataFrame("t", str(path))
    handles = {}
    for b in bins:
        for kind, s in (("full", b.expression), ("stripped", b.expression_no_hlt)):
            handles[(b.mode, b.name, kind)] = df.Define("m", f"(int)({s})").Take["int"]("m")
    ctx = expr.EvalContext(arrays=data, n_events=n)
    nonempty = 0
    for b in bins:
        for kind, s in (("full", b.expression), ("stripped", b.expression_no_hlt)):
            ref = np.array(handles[(b.mode, b.name, kind)].GetValue(), dtype=bool)
            ours = expr.evaluate(expr.parse(s), ctx)
            assert np.array_equal(ref, ours), f"{b.mode}/{b.name}/{kind}: {np.count_nonzero(ref != ours)} differ"
            nonempty += int(ref.any())
    assert nonempty > len(bins)  # the synthetic sample actually populates the bins
