"""Notebook 07: our methods given exactly the transitions FLUS's forecasts made (FLUS's amount), to
separate how much FLUS changes from where it places change. Usage: python run07_flus_amount.py"""

import json

import tools100 as T
import decisions100 as M

D = T.D
FLUS = {"observed": "flus_2025.tif", "projected": "flus_2025_projected.tif"}
RUNS = [("blend", s) for s in M.SEEDS] + [("document", 42), ("backend", 42)]

if __name__ == "__main__":
    for kind, f in FLUS.items():
        demand = T.tool_transitions(T.BENCH / "output_flus" / f)
        for method, seed in RUNS:
            tag = D.tag("test", seed, method)
            atlas, _ = D.variant_inputs(tag, "test", "no clouds", "observed")
            name = f"{kind}_{tag}"
            cache = T.CODE / "flus_amount" / name / "score.json"
            if not cache.exists():
                path = T.same_amount(name, atlas, demand, folder="flus_amount")
                cache.write_text(json.dumps(T.score_esri(path), default=float))
            s = json.loads(cache.read_text())
            print(f"FLUS's {kind} amount | {tag}: FoM {s['FoM']:.2f}, new {s['new-change FoM']:.2f}, "
                  f"random {s['random FoM']:.2f}, changed {s['predicted change (cells)'] / 1e6:.2f} M", flush=True)
