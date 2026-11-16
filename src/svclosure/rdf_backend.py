"""Independent RDataFrame backend that replays LLPCombine's own C++ path for one file.

Run as a *separate process* (``python -m svclosure.rdf_backend spec.json out.json``)
so PyROOT never shares a process with uproot's XRootD handles.

It reproduces, for one signal file:
  * the FastSim decision from kuSkimConfigTree.sMCType == 2,
  * the trigger stripping with C++ ``std::regex_replace`` and the identical pattern,
  * ``evtwt = evtFillWgt * std::to_string(Lumi)``, ``decayWeight`` from LLPCombine's
    own ``ReweightTool::DecayReweight`` (its header is #included read-only),
  * ``Filter(cuts)`` -> Count, Sum(evtrwt), Sum(evtrwt2),
and also evaluates the symmetric (both-stripped) diagnostic mode.

Lifetime targets (``spec["lifetime"] = {"source_cm": s, "targets": [...]}``) replay
``BuildFitInput::BuildReweights`` line for line: ``float tau_old = float(sampleLifetime)``,
``Define("xawt", [=](float ctau){ return ReweightTool::LifetimeReweight(ctau, tau_old, tau_new); },
{"Xa_ctau"})``, the same for ``xbwt``, then ``evtrwt = evtwt*xawt*xbwt*decayWeight`` (JIT string).
With ``spec["dump_weights"]`` the per-event xawt, xbwt and evtrwt of every target are written to
``<out>.npz`` for an element-wise comparison.
"""
from __future__ import annotations

import json
import sys
import time


def main(spec_path: str, out_path: str) -> int:
    import ROOT

    spec = json.load(open(spec_path))
    ROOT.gROOT.SetBatch(True)
    if spec.get("threads", 0):
        ROOT.EnableImplicitMT(int(spec["threads"]))
    ROOT.gInterpreter.Declare(
        r"""
        #include <regex>
        #include <string>
        std::string svc_strip_triggers(const std::string& s){
            static const std::regex trigger_re("\\bTrigger_\\w+\\b");
            return std::regex_replace(s, trigger_re, "true");
        }
        """
    )
    header = spec["llpcombine_include"] + "/BuildFitTools.h"
    used_header = bool(ROOT.gInterpreter.Declare(f'#include "{header}"'))
    if not used_header:
        raise RuntimeError(f"could not include {header}")
    w = spec["weights"]

    f = ROOT.TFile.Open(spec["path"])
    ct = f.Get("kuSkimConfigTree")
    ct.GetEntry(0)
    mctype = int(ct.sMCType) if ct.GetBranch("sMCType") else 0
    ntot = int(ct.nTotEvts)
    xsec = float(ct.sCrossSection)
    f.Close()
    is_fastsim = mctype == 2

    df = ROOT.RDataFrame(spec.get("tree", "kuSkimTree"), spec["path"])
    lumi_str = w["lumi_cpp"]
    node = df.Define("evtwt", "evtFillWgt * " + lumi_str).Define("evtwt2", "evtwt*evtwt")
    zo, go, zn, gn = w["sample_z"], w["sample_g"], w["target_z"], w["target_g"]
    node = node.Define(
        "decayWeight",
        f"ReweightTool::DecayReweight(Evt_isZZ, Evt_isGZ, Evt_isGG, {zo!r}, {zn!r}, {go!r}, {gn!r})",
    )
    node = node.Define("evtrwt", "evtwt*decayWeight").Define("evtrwt2", "evtrwt*evtrwt")
    variants = [("noreweight", "evtrwt", None)]
    lt = spec.get("lifetime")
    if lt:
        ROOT.gInterpreter.Declare(
            r"""
            ROOT::RDF::RNode svc_lifetime(ROOT::RDF::RNode df, const std::string& sfx, double sampleLifetime,
                                          double targetLifetime){
                float tau_old = float(sampleLifetime);   // BuildFitInput::BuildReweights
                float tau_new = float(targetLifetime);
                auto a = df.Define("xawt" + sfx, [=](float ctau){ return ReweightTool::LifetimeReweight(ctau, tau_old, tau_new); },
                                   {"Xa_ctau"});
                auto b = a.Define("xbwt" + sfx, [=](float ctau){ return ReweightTool::LifetimeReweight(ctau, tau_old, tau_new); },
                                  {"Xb_ctau"});
                auto c = b.Define("evtrwt" + sfx, "evtwt*xawt" + sfx + "*xbwt" + sfx + "*decayWeight");
                return c.Define("evtrwt2" + sfx, "evtrwt" + sfx + "*evtrwt" + sfx);
            }
            """
        )
        node = ROOT.RDF.AsRNode(node)
        for i, t in enumerate(lt["targets"]):
            sfx = f"_t{i}"
            node = ROOT.svc_lifetime(node, sfx, float(lt["source_cm"]), float(t))
            variants.append((lt["variant_keys"][i], "evtrwt" + sfx, float(t)))
    dumps = {}
    if spec.get("dump_weights") and lt:
        dumps["entry"] = node.Take["ULong64_t"]("rdfentry_")
        dumps["evtrwt_noreweight"] = node.Take["double"]("evtrwt")
        for i, (vk, col, t) in enumerate(variants[1:]):
            dumps[f"xawt_{vk}"] = node.Take["float"](f"xawt_t{i}")
            dumps[f"xbwt_{vk}"] = node.Take["float"](f"xbwt_t{i}")
            dumps[f"evtrwt_{vk}"] = node.Take["double"](col)

    results = {}
    t0 = time.perf_counter()
    for b in spec["bins"]:
        full = b["expression"]
        stripped = str(ROOT.svc_strip_triggers(full))
        cuts = {"parity": stripped if is_fastsim else full, "symmetric": stripped}
        for hlt, c in cuts.items():
            filt = node.Filter(c, f"{b['name']}_{hlt}")
            for vk, col, _ in variants:
                results[(b["mode"], b["name"], hlt, vk)] = (filt.Count(), filt.Sum(col), filt.Sum(col.replace("evtrwt", "evtrwt2", 1)),
                                                            stripped)
    out = {
        "path": spec["path"],
        "sMCType": mctype,
        "nTotEvts": ntot,
        "sCrossSection": xsec,
        "is_fastsim": is_fastsim,
        "header_included": used_header,
        "root_version": ROOT.gROOT.GetVersion(),
        "yields": [],
    }
    for (mode, name, hlt, vk), (cnt, s, s2, stripped) in results.items():
        out["yields"].append(
            dict(mode=mode, bin=name, hlt=hlt, variant=vk, n_raw=int(cnt.GetValue()), sumw=float(s.GetValue()),
                 sumw2=float(s2.GetValue()), cpp_stripped=stripped)
        )
    if dumps:
        import numpy as np

        np.savez_compressed(out_path.replace(".json", "") + ".npz",
                            **{k: np.asarray(v.GetValue()) for k, v in dumps.items()})
        out["weights_dump"] = out_path.replace(".json", "") + ".npz"
    out["lifetime"] = lt
    out["seconds"] = time.perf_counter() - t0
    json.dump(out, open(out_path, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
