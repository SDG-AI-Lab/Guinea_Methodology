"""Runs every variant of notebook 12 in one go (about 25 minutes each; results cached, so the
notebook then only reads them). From landuse_experiments/:  .venv/Scripts/python scripts/run_predictor_tests.py"""
import sys
import time

import esri_methods as E
import predictor_tests as P

VARIANTS = {"pred_full": {}}
VARIANTS |= {f"pred_drop_{i:02d}": {"drop": (name,)} for i, name in enumerate(E.PREDICTORS, 1)}
VARIANTS |= {"pred_add_mining": {"extra": P.mining_layers}, "pred_add_fire": {"extra": P.fire_layers}}
if "--seeds" in sys.argv:                       # the candidates again with seeds 1 and 2
    VARIANTS = {f"{tag}_seed{s}": {**kw, "seed": s} for tag, kw in VARIANTS.items() if tag.startswith("pred_add")
                for s in (1, 2)}

for tag, options in VARIANTS.items():
    for phase in ("validation", "test"):
        started = time.time()
        result = P.improved_variant(tag, phase, log=lambda *a: None, **options)
        fom = result["scores"]["Expander-only"]["figure_of_merit_pct"]
        print(f"{tag:22s} {phase:10s} FoM {fom:6.2f}  ({(time.time() - started) / 60:.1f} min)", flush=True)
