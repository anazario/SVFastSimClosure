# SVFastSimClosure

Residual FullSim / FastSim signal-yield scale factors for every SV-containing bin of
the LLP analysis, using the bins, selections, weights and sample dictionary of
[LLPCombine](https://github.com/Jphsx/LLPCombine) as the single source of truth.

    SF(bin, hypothesis) = Y_FullSim / Y_FastSim,corr

The FastSim inputs (`*_Fast1_SxyScaleNominal_*`, SampleTool key `gogoGZFAST`) already
carry the nominal mass-splitting-dependent S_xy correction from skimming. It is
**not** applied again. This package only *measures*. It does not define an
intrinsic-scatter model or Combine nuisances.

## What is measured

* 40 SV bins resolved from `config_master/{LittleGuy_Compressed,BigGuy_NonCompressed}_FullRegions_*`:
  * 20 **high-S_xy** bins in 6 groups (`SVonly_SR`, `MixPrompt_SR`, `MixDel_SR`;
    `Ch2SRGeLep1`, `Ch4SRGeHad1`, `Ch14SRgeq1SVHighDxygeq1PhoNotBHLate`). Each group
    has one normalization-anchor bin (`*CR00` / `Ch*CR*00`).
  * 20 **low-S_xy** control bins (each the exact partner of a high-S_xy bin, with the
    S_xy flag swapped). These are for closure diagnostics. In the compressed configs,
    `*_AnchorCR*` are these *low*-S_xy bins, and the high-S_xy anchors are `*_CR00`.
* Per (mode, bin, hypothesis, simulation, HLT mode): raw count, Σw, Σw², stat. error, N_eff.
* HLT modes:
  * `parity`: as in the fit. Trigger_* terms are replaced by `true` only for
    samples with `sMCType == 2` (FastSim), exactly as `BuildFitInput::FilterRegions` does.
  * `symmetric`: a diagnostic. Trigger_* terms are dropped for both samples, which
    isolates reconstruction effects.
* Weights as LLPCombine: `evtrwt = evtFillWgt · L · decayWeight`, with
  L = 423.71 fb⁻¹ (summed data years). Decay reweighting is 0.5/0.5 → 0.5/0.5,
  which main.cpp turns on by default: weight 1 for `Evt_is{ZZ,GZ,GG}`, 0 otherwise.

## Selection backends

* Primary: uproot + awkward. `expr.py` parses the C++ cut strings with a
  restricted grammar (comparisons, `&& || !`, `x[i]`, literals). Anything else is
  rejected. Comparisons use C++ float→double promotion.
* Parity: `rdf_backend.py` replays LLPCombine's own path (RDataFrame JIT,
  `std::regex` trigger stripping, `ReweightTool::DecayReweight` from LLPCombine's
  header) in a **separate process**.
* Tests compile LLPCombine's real `ConfigParser.cpp` (`tools/dump_cuts.cpp`) and
  require identical cut strings for all 117 bins. They also compare the translator
  with ROOT's C++ JIT event by event on synthetic boundary-heavy events.

## Remote access (this Mac)

Files are read from `root://cmseos.fnal.gov//store/...` with the native XRootD
source (`handler=uproot.source.xrootd.XRootDSource`). This avoids the known
fsspec-xrootd close/hang problem when PyROOT is in the same process. You need a
Kerberos ticket in the file cache that conda's XRootD can read:
`KRB5CCNAME=FILE:/tmp/krb5cc_$UID` (set in `~/.zshrc`), then `kinit <user>@FNAL.GOV`.

## Usage

```bash
svclosure show-bins --sxy high                  # resolved selections (no file access)
svclosure manifest                              # sample + bin manifests, opens every file
svclosure validate --point gogoGZ_2300_1300_1000_10        # first-stage validation
svclosure validate --point gogoGZ_2300_1300_1000_10 \
    --compare-json ~/code/LLPCombine/json/<file>.json --compare-config ~/code/LLPCombine/<config>.yaml
svclosure run --workers 3                       # all matched points -> tables + figures
svclosure plot --run results/<run-dir>
```

Every run writes `results/<timestamp>_<tag>/`, containing:
* `provenance.json`: git commits and dirty state of all three repos, SHA-256 of each
  LLPCombine file read, package versions, weights and luminosity.
* `study_config.yaml`
* `manifest/`: bins and samples
* `yields.csv`, `sf.csv`, `sf_pooled.csv`
* `perf.json`: bytes requested, MB/s, timing per file
* `figures/`: PDF and PNG

Configuration lives in `configs/study.yaml`: paths, sample keys, I/O concurrency
(`workers`, `step_size`, `task_entries`) and the plot energy label.

## Tests

```bash
pytest -q
```
