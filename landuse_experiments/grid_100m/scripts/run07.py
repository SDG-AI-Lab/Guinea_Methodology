"""Runs and scores everything notebook 07 shows at 100 m, one run at a time (each finished piece is
cached, so the script can be restarted). Usage: python run07.py"""

import json
import logging
import time

import tools100 as T

logging.basicConfig(level=logging.ERROR)
log = lambda m: print(m, flush=True)
SCORES = T.CODE / "scores"
FLUS_SEEDS = (42, 1, 2)


def scored(key, who, comparison, make, kind="tool"):
    """Score of one forecast, cached in output_code/scores/<key>.json with its run time and peak memory.
    make() returns the forecast's path (tool codes, or Esri codes for kind="esri")."""
    cache = SCORES / f"{key}.json"
    if not cache.exists():
        SCORES.mkdir(parents=True, exist_ok=True)
        with T.D.P100.Peak() as peak:
            path = make()
        s = T.score_tool(path) if kind == "tool" else T.score_esri(path)
        cache.write_text(json.dumps({"key": key, "who": who, "comparison": comparison, **s,
                                     "minutes": peak.seconds / 60, "peak_GB": peak.peak / 1e9}, default=float))
    s = json.loads(cache.read_text())
    log(f"{key}: FoM {s['FoM']:.2f}, new {s['new-change FoM']:.2f}, random {s['random FoM']:.2f} "
        f"({s['minutes']:.0f} min, {s['peak_GB']:.1f} GB; {(time.time() - t0) / 60:.0f} min so far)")
    return s


if __name__ == "__main__":
    t0 = time.time()
    plus = T.BENCH / "output_plus"
    scored("plus_observed", "GUI", "as it runs, observed amount", lambda: plus / "plus_2025Simulation_1.tif")
    scored("plus_projected", "GUI", "as it runs, projected amount", lambda: plus / "plus_2025_projectedSimulation_1.tif")
    scored("plus_same", "GUI", "same amount", lambda: T.same_amount("plus_leas", T.plus_atlas()), "esri")

    flus = T.BENCH / "output_flus"                     # FLUS 2.3, run by hand
    scored("flus_observed", "GUI", "as it runs, observed amount", lambda: flus / "flus_2025.tif")
    scored("flus_projected", "GUI", "as it runs, projected amount", lambda: flus / "flus_2025_projected.tif")
    scored("flus_same", "GUI", "same amount", lambda: T.same_amount(
        "flus_gui", T.class_band_atlas(flus / "Probability-of-occurrence.tif", out=T.CODE / "flus_gui")), "esri")

    for seed in FLUS_SEEDS:
        scored(f"flus_same_s{seed}", "code", "same amount",
               lambda: T.same_amount(f"flus_network_s{seed}", T.class_band_atlas(T.flus_network(seed, log))), "esri")

    scored("clue_s_observed", "code", "as it runs, observed amount", lambda: T.clue_s("observed", log) / "clue_s_2025.tif")
    scored("clue_s_same", "code", "same amount",
           lambda: T.same_amount("clue_s", T.class_band_atlas(T.CODE / "clue_s" / "clue_s_probability.tif")), "esri")

    for method, seed in (("lr", 0), ("ann", 0), ("lr", 1), ("lr", 2)):
        k = f"molusce_{method}" + (f"_s{seed}" if seed else "")
        scored(k, "code", "as it runs, own amount", lambda: T.molusce(method, seed) / f"molusce_{method}_2025.tif")
        scored(f"{k}_same", "code", "same amount",
               lambda: T.same_amount(k, T.molusce_atlas(T.molusce(method, seed), method)), "esri")

    # last: with the projected amount CLUE-S converges slowly (up to its 20,000-iteration limit)
    scored("clue_s_projected", "code", "as it runs, projected amount", lambda: T.clue_s("projected", log) / "clue_s_2025.tif")
    log("done")
