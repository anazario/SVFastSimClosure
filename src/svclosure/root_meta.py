"""Read kuSkimConfigTree metadata with PyROOT, in its own process.

Why: in the skims_v52 signal files every kuSkimConfigTree branch claims 200
entries while TTree::GetEntries() is 100 and only 101 basket entries exist on
disk.  uproot refuses such branches (even for entry_stop=1); ROOT tolerates
them, and LLPCombine reads them with ROOT (SetBranchAddress + GetEntry(0)).
Reading them the same way keeps metadata parity with LLPCombine.

Usage: python -m svclosure.root_meta out.json path1 [path2 ...]
"""
from __future__ import annotations

import json
import sys

FIELDS = ("nTotEvts", "sCrossSection", "sMCType", "sumEvtWgt", "nEvents", "nSelectedEvents")


def main(out: str, paths: list[str]) -> int:
    import ROOT

    ROOT.gROOT.SetBatch(True)
    ROOT.gErrorIgnoreLevel = ROOT.kError
    res = {}
    for p in paths:
        try:
            f = ROOT.TFile.Open(p)
            if not f or f.IsZombie():
                raise OSError("could not open")
            ct = f.Get("kuSkimConfigTree")
            if not ct:
                raise KeyError("no kuSkimConfigTree")
            have = [x for x in FIELDS if ct.GetBranch(x)]
            n = int(ct.GetEntries())
            rows = []
            for i in range(n):
                ct.GetEntry(i)
                rows.append({x: getattr(ct, x) for x in have})
            branch_entries = {x: int(ct.GetBranch(x).GetEntries()) for x in have}
            f.Close()
            entry0 = {k: (float(v) if isinstance(v, float) else int(v)) for k, v in rows[0].items()} if rows else {}
            sums = {k: float(sum(r[k] for r in rows)) for k in ("nTotEvts", "sumEvtWgt", "nEvents") if k in have}
            distinct = {k: len({r[k] for r in rows}) for k in have}
            res[p] = dict(ok=True, entries=n, branch_entries=branch_entries, entry0=entry0, sums=sums,
                          n_distinct=distinct)
        except Exception as exc:
            res[p] = dict(ok=False, error=f"{type(exc).__name__}: {exc}")
    json.dump(res, open(out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2:]))
