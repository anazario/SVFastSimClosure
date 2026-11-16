from collections import Counter

EXPECTED_HIGH = {
    "compressed": {
        "SVonly_SR": ["SVonly_CR00", "SVonly_SR10", "SVonly_SR20"],
        "MixPrompt_SR": ["MixPrompt_CR00", "MixPrompt_SR10", "MixPrompt_SR20"],
        "MixDel_SR": ["MixDel_CR00", "MixDel_SR10", "MixDel_SR20"],
    },
    "noncompressed": {
        "Ch2SRGeLep1": ["Ch2CRGeLep100", "Ch2SRGeLep110", "Ch2SRGeLep101", "Ch2SRGeLep111"],
        "Ch4SRGeHad1": ["Ch4CRGeHad100", "Ch4SRGeHad110", "Ch4SRGeHad101", "Ch4SRGeHad111", "Ch4SRGeHad121"],
        "Ch14SRgeq1SVHighDxygeq1PhoNotBHLate": ["Ch14CRgeq1SVHighDxygeq1PhoNotBHLate00",
                                                "Ch14SRgeq1SVHighDxygeq1PhoNotBHLate01"],
    },
}


def test_twenty_high_sxy_bins_in_six_groups(all_bins):
    bins, problems = all_bins
    assert problems == []
    high = [b for b in bins if b.sxy == "high"]
    assert len(high) == 20
    got = {}
    for b in high:
        got.setdefault(b.mode, {}).setdefault(b.group, []).append(b.name)
    assert got == EXPECTED_HIGH


def test_anchor_roles(all_bins):
    bins, _ = all_bins
    anchors = sorted(b.name for b in bins if b.role == "anchor")
    assert anchors == sorted(v[0] for m in EXPECTED_HIGH.values() for v in m.values())
    # compressed 'AnchorCR' bins are the ordinary LOW-S_xy control bins, not anchors
    for b in bins:
        if "AnchorCR" in b.name:
            assert b.sxy == "low" and b.role == "lowsxy_cr"


def test_low_sxy_partners_are_one_to_one(all_bins):
    bins, _ = all_bins
    by = {(b.mode, b.name): b for b in bins}
    assert Counter(b.sxy for b in bins) == {"high": 20, "low": 20}
    for b in bins:
        assert b.partner is not None
        other = by[(b.mode, b.partner)]
        assert other.partner == b.name and other.sxy != b.sxy


def test_trigger_stripping_only_touches_trigger_tokens(all_bins):
    bins, _ = all_bins
    for b in bins:
        assert "Trigger_" in b.expression
        assert "Trigger_" not in b.expression_no_hlt
        assert set(b.branches) - set(b.branches_no_hlt) == {x for x in b.branches if x.startswith("Trigger_")}


def test_labels_come_from_selection(all_bins):
    bins, _ = all_bins
    lab = {b.name: b.kin_label for b in bins}
    assert lab["Ch4SRGeHad111"] == r"$M_{S} \in [2.6,3.0)\,\mathrm{TeV}$; $R_{S} \geq 0.35$"
    assert lab["SVonly_SR10"] == r"$p_{T}^{ISR} > 500\,\mathrm{GeV}$; $R_{ISR} \in [0.6,0.75)$"
