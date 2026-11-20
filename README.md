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

## Lifetime reweighting

`svclosure lifetime-scan` evaluates the same generated FullSim/FastSim pairs at other proper
lifetimes with LLPCombine's event weight (`BuildFitInput::BuildReweights`):

    evtrwt = evtFillWgt * L * xawt * xbwt * decayWeight,
    xawt = ReweightTool::LifetimeReweight(Xa_ctau, tau0, tau)  (same for Xb),
    w(t; tau0 -> tau) = (tau0/tau) exp[t (1/tau0 - 1/tau)]

* Units: c*tau in **cm** everywhere (`Xa_ctau`/`Xb_ctau`, LLPCombine's `sampleLifetime`/`targetLifetime`,
  the signal-key token).  The generated lifetime is read from each file name independently (`ct0p1` =
  0.1 m = 10 cm) and must agree between FullSim, FastSim and the LLPCombine key.
* `point` keeps the LLPCombine process key of the *generated* sample; `source_ctau_cm` and
  `target_ctau_cm` are separate columns.  The scan refuses a `WeightSpec` that already carries lifetime
  settings, so the factor cannot be applied twice.  The FastSim S_xy correction is not touched.
* Selections are identical for all targets: masks are evaluated once per chunk, and all targets are
  accumulated together (sum w, sum w^2, sum w_i w_j between targets, top-k weights) in one read.
* The target equal to the generated lifetime is the direct reference.  It is always evaluated, even
  when `--targets` omits it (`target_origin = auto_direct`), and it must be bitwise identical to the
  unreweighted result (`lifetime_checks.json`; a mismatch fails the run).
* Input anomalies are counted for every event (`sample_lifetime_inputs.csv`, `lifetime_checks.json`).
  `lifetime_anomalies.csv` lists only examples, at most 2,000 per task.  The classes are:
  `ctau_nonfinite`, `ctau_negative`, `ctau_zero`, `decay_flags_none` and `decay_flags_multiple`.
  A zero ctau is a valid input, with weight `tau0/tau`.  `invalid_ctau_policy` takes `llpcombine`
  (the default: weight as LLPCombine does), `exclude` or `fail`.  The full table is in `weights.py`
  and in `run_config.json`.  Under every policy, a selected event with a non-finite weight is fatal.
* `run_status.json` records `complete`, `partial` or `failed`.  A failed run exits nonzero and is
  renamed `*_FAILED`; it keeps diagnostics only.  `svclosure plot --run` uses the configuration
  saved in that run.
* Python weights are bit-exact with the C++ header.  The ratio `tau0/tau` is computed in float and the
  exponent in double; `validate` compares every event's xawt/xbwt/evtrwt with RDataFrame.
* Quality criteria (`lifetime_scan.quality`): `min_neff`, `min_neff_reweighted`, `max_top1_sumw2_frac`,
  `max_top5_sumw2_frac`, and the per-target verdict thresholds.  A finite reweighted yield is not
  assumed to be reliable.
* Targets of one generated sample are correlated.  `lifetime_cov_yields.csv` (sum w_i w_j) and
  `lifetime_cov_logsf.csv` (Cov of ln SF) give the covariance; never treat targets as independent.
* `crosscheck_targets` (50 -> 10 cm cross-check, 10 -> 50 cm stress test) feed only
  `lifetime_source_crosscheck*.csv` and their figures, never the nominal summaries.

## Signal regimes and pooled summaries

Pools are **ratios of summed yields** (`estimator = ratio_of_sums`), not fitted common SFs.
The nominal figures pool each analysis mode over its own regime (`signal_regimes` in the YAML):
Compressed mode over compressed hypotheses, High-Delta-m mode over high-Delta-m hypotheses.
Each generated lifetime is pooled separately.  The regime rule is Delta m = m(N2) - m(N1) <= 200 GeV
(LLPCombine `macro/CalculateCutflowPiechart.py`).  It is cross-checked against the
`macro/LifetimeEff.py` convention and the `macro/llpcombine.py` benchmarks; any disagreement is
reported and excluded from the pools.  The legacy all-hypotheses pool (`sf_pooled.csv`) stays as a
diagnostic.

## Category checks

Every run accumulates event-level overlaps of all selection pairs and checks them against the
expected category logic (`overlap.py`):
* within a channel: disjoint is required (a violation makes the run fail),
* between channels of one mode, and between high/low partners: disjoint is expected (reported),
* between modes: overlap is allowed.

Bin structure problems also stop `run` and `lifetime-scan`.  Omissions are events inside a channel
envelope (the cuts shared by its bins) that enter no bin.  They are split into those explained by a
static kinematic gap (e.g. R_ISR < 0.4 in the delayed-photon channels) and unexplained ones.

## Usage

```bash
svclosure show-bins --sxy high                  # resolved selections (no file access)
svclosure manifest                              # sample + bin manifests, opens every file
svclosure validate --point gogoGZ_2300_1300_1000_10        # parity incl. lifetime targets of 10 cm
svclosure validate --point gogoGZ_2300_1300_1000_10 \
    --compare-json ~/code/LLPCombine/json/<file>.json --compare-config ~/code/LLPCombine/<config>.yaml
svclosure run --workers 4                       # generated lifetimes only -> results/<stamp>_scan
svclosure lifetime-scan --workers 4             # all targets            -> results/<stamp>_lifetime
svclosure lifetime-scan --targets "10:5,10;50:30,50" --no-crosscheck --tag lifetime_small
svclosure lifetime-scan --min-neff-reweighted 50 --max-top1 0.1 --tag lifetime_strict
svclosure plot --run results/<run-dir>          # remake figures from the CSVs
```

Each run writes `results/<timestamp>_<tag>/` with `OUTPUTS.md` describing every file.
It also writes `provenance.json` (git commit, dirty state, the list of modified/untracked files, and
the SHA-256 of the uncommitted diff of this repository; commit and status of LLPCombine and
LLPStandardPlots, plus the SHA-256 of every LLPCombine file read), `run_config.json` (the complete
scan configuration, quality criteria and reference selections), `study_config.yaml`, `manifest/`,
`perf.json`, the tables, and `figures/` (`figures/lifetime/` for the scan).

Figures follow these conventions:
* Filled markers are reliable.  Open markers with dashed error bars are low statistics or
  concentrated weights.
* Hollow triangles at the axis edge mean an empty FullSim or FastSim sample (undefined ratio).
* Values outside the automatic y range are drawn as arrows at the edge and counted in the note.
* The gray band is a +-10% reference tolerance, not a fit.
* The `*_noBinNames` versions are the primary ones.  Set `plot.full_bin_name_figures: true` for
  the debugging versions with analysis bin names.

### Migration from the first scan (results/20261116-021213_scan)

* `yields.csv`, `sf.csv` and `sf_pooled.csv` keep their names, rows and original columns in the
  original order.  New columns are appended (concentration diagnostics, anomaly counts, `sf_class`,
  `ln_sf`, `short`, mass hypothesis, sample paths, `svclosure_commit`/`_dirty`/`_diff_sha256`;
  `sf_pooled.csv` also gets `pool`, `estimator`, `n_hypotheses`).
* `yields.csv` and `sf.csv` are bitwise identical to the first scan.  `sf_pooled.csv` differs by
  <= 1 ulp (<= 4e-16 relative), because points are now summed in a fixed order instead of
  task-completion order.
* `sf_high_vs_low_pooled.csv` is now produced by the pipeline.  Its columns match the earlier ad-hoc
  file, but without rounding.  `ratio_err` includes the high/low covariance, which is zero because
  every pair is event-level disjoint, so it equals `ratio_err_uncorrelated`, the old definition.
* `svclosure plot` needs the new layout, so rerun `svclosure run` for old run directories.

### Reproducing a run from a clean committed revision

```bash
cd ~/code/SVFastSimClosure
git status --short            # must be empty: commit (or stash) first
git rev-parse HEAD            # recorded as repos.SVFastSimClosure.commit in provenance.json
conda activate hep            # or: conda env create -f environment.yml && conda activate svclosure && pip install -e .
export KRB5CCNAME=FILE:/tmp/krb5cc_$UID && kinit <user>@FNAL.GOV
PYTHONPATH=src python3 -m pytest -q
PYTHONPATH=src caffeinate -i python3 -m svclosure.cli run --workers 4
PYTHONPATH=src caffeinate -i python3 -m svclosure.cli lifetime-scan --workers 4
```

Check `provenance.json`: `repos.SVFastSimClosure.dirty` must be `false`, and the LLPCombine
`files_read` entries must all be `clean`.  LLPCombine itself may be dirty elsewhere; only the hashes
of the files read matter.  To reproduce an older run, check out its recorded commit.  If that run
was dirty, the recorded `uncommitted_diff_sha256` tells you whether a local diff matches it.

## Tests

```bash
PYTHONPATH=src python3 -m pytest -q
```

They include bit-exact checks of `lifetime_weight` against `ReweightTool::LifetimeReweight`, and an
offline end-to-end parity test: a synthetic ROOT file with boundary values, negative and zero c*tau,
and missing decay flags, processed by both the uproot engine and the RDataFrame replica of
`BuildReweights` for five weight variants.
