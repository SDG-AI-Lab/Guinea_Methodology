"""The platform's land-use run (main, commit 6710b3d) at 100 m on this laptop, with its own code:
a read-only snapshot of landuse_tool in grid_100m/data/platform_main (git archive, never edited).

The stages are those of backend/landuse_api/tasks.py:

  run_training    change_analysis.run_change_analysis on the training years (first -> last);
                  every transition with a count above zero; training.train_transition_models
                  (one Random Forest per transition, use_growth_modes=False, the API default),
                  predictors of the first training year
  run_prediction  prediction.generate_suitability_atlas with the base year's predictors and map;
                  demand = run_change_analysis on 2017 ... base year x (target - base);
                  prediction.run_allocation_simulation (deterministic)

Each stage's time and peak memory (this process and its worker processes) are recorded.
Inputs are written once by predictors_lu100 (write_esri, write_predictors) on E:.

Usage:  python platform100.py --train 2023 2024 --base 2024 --target 2025 --seed 42 --tag test_s42
"""

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

for _key in ("PROJ_LIB", "PROJ_DATA", "GDAL_DATA"):                # see nb_helpers
    os.environ.pop(_key, None)

import numpy as np
import psutil

HERE = Path(__file__).resolve().parents[1]
MAIN = HERE / "data" / "platform_main"
sys.path.insert(0, str(HERE / "scripts"))
sys.path.insert(0, str(MAIN))
import predictors_lu100 as P                                        # noqa: E402

RUNS = P.LANDUSE / "runs"
RUNS_D = HERE / "data" / "runs"                                     # runs started after E: needed a repair (2026-10-29)
FIRST_YEAR = 2017                                                   # TARGET_RASTERS_CONFIG year_hint


def tool():
    """main's landuse_tool. The 250 m helpers (nb_helpers) put an older copy, Land Use-Deforestation/
    landuse_tool, first on the path; a run that imported it would not be main's, so this refuses it."""
    if sys.path[0] != str(MAIN):
        sys.path.insert(0, str(MAIN))
    import landuse_tool
    if Path(landuse_tool.__file__).resolve().parents[1] != MAIN:
        raise RuntimeError(f"landuse_tool was imported from {landuse_tool.__file__}, not main's snapshot")
    return landuse_tool


class Peak:
    """Peak resident memory of this process and its children while a stage runs."""

    def __init__(self, every=0.2):
        self.every, self.proc = every, psutil.Process()

    def _rss(self):
        total = self.proc.memory_info().rss
        for child in self.proc.children(recursive=True):
            try:
                total += child.memory_info().rss
            except psutil.Error:
                pass
        return total

    def _watch(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, self._rss())
            time.sleep(self.every)

    def __enter__(self):
        self.peak, self._stop, self.t0 = self._rss(), threading.Event(), time.time()
        self._thread = threading.Thread(target=self._watch, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self.seconds = time.time() - self.t0
        self.peak = max(self.peak, self._rss())


def platform_stack_memory(year):
    """Peak memory of holding the predictor stack as main's assemble_predictor_stack does: twelve
    full-grid float32 arrays in a dict, slope and aspect from the grid DEM, distances by
    distance_transform_edt over the grid (process_vector_layer)."""
    import grid100 as G
    import platform_utils as PU
    from scipy.ndimage import distance_transform_edt
    stack = {}
    with Peak() as peak:
        for name in P.NAMES[:6]:
            stack[name] = G.to_grid(P.predictor(name, year), fill=np.nan, dtype="float32")
        prof = dict(G.profile(), nodata=None)
        stack["SLOPE"] = PU.calculate_slope(stack["DEM_GEE"], prof)
        stack["ASPECT"] = PU.calculate_aspect(stack["DEM_GEE"], prof)
        inside = G.inside_mask()
        for name, key in (("ROADS_distance", "roads"), ("WATERWAYS_distance", "waterways"), ("POI_heatmap", "poi")):
            rasterized = G.read_layer("osm", [key], as_grid=True, fill=0)[key].astype("float32")
            dist = distance_transform_edt(rasterized == 0).astype(np.float32)
            if name == "POI_heatmap":
                dist = np.exp(-dist / (dist.max() * 0.1))
            stack[name] = np.where(inside, dist, np.nan)
            del rasterized, dist
        stack["PROTECTED_AREAS_zone"] = G.to_grid(P.predictor("PROTECTED_AREAS_zone", year), fill=np.nan, dtype="float32")
    return {"stage": "Predictor stack in memory (assemble_predictor_stack)", "seconds": peak.seconds,
            "peak_GB": peak.peak / 1e9, "layers": len(stack)}


def run_dir(tag):
    """Runs live on E:. From 2026-10-29 to 2026-11-01 new runs were written to D: while E: waited for a
    repair (chkdsk clean 2026-10-30); a finished run still on D: is read there, and every other run is on
    E: (D: filled up on 2026-11-01)."""
    return RUNS_D / tag if (RUNS_D / tag / "result.json").exists() else RUNS / tag


def run(train, base, target, seed, tag, growth_modes=False, log=print):
    """growth_modes: the API's use_growth_modes (default False): Expander and Patcher models per
    transition, combined by dilation in the allocation. A finished run (result.json) is not redone."""
    tool()
    from landuse_tool import change_analysis, prediction, training
    out = run_dir(tag)
    if (out / "result.json").exists():
        return json.loads((out / "result.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    stages = []

    def record(name, peak, **extra):
        stages.append({"stage": name, "seconds": peak.seconds, "peak_GB": peak.peak / 1e9, **extra})
        log(f"[{tag}] {name}: {peak.seconds:.0f} s, peak {peak.peak / 1e9:.2f} GB {extra or ''}")

    with Peak() as peak:
        years = sorted(set(range(FIRST_YEAR, base + 1)) | set(train) | {target})
        esri = {y: P.write_esri(y) for y in years}
        pred_train, pred_base = P.write_predictors(train[0]), P.write_predictors(base)
    record("Write inputs (this folder, not the platform)", peak)
    import gc
    P._coarse.cache_clear(); P.D.static.cache_clear(); gc.collect()   # the platform's stages start clean

    # run_training
    target_files = [{"file": esri[y], "year": y} for y in train]
    with Peak() as peak:
        _, counts, _, _ = change_analysis.run_change_analysis(target_files)
        transitions = [(int(i), int(j)) for i in counts.index for j in counts.columns
                       if i != j and counts.loc[i, j] > 0]
    record("Training transitions (change analysis)", peak, transitions=len(transitions))
    np.random.seed(seed)                                            # create_transition_dataset samples with np.random
    with Peak() as peak:
        model_paths, _ = training.train_transition_models(transitions, target_files, pred_train,
                                                          use_growth_modes=growth_modes, temp_dir=str(out / "models"))
    record("Training (train_transition_models)", peak, models=len(model_paths))

    # run_prediction
    with Peak() as peak:
        suitability = prediction.generate_suitability_atlas(pred_base, esri[base], model_paths, str(out))   # job_output_dir, as run_prediction
    record("Suitability atlas (generate_suitability_atlas)", peak, maps=len(suitability))
    if len(suitability) != len(model_paths):                        # main skips a map it cannot read
        raise RuntimeError(f"{len(suitability)} of {len(model_paths)} suitability maps were written; see the log")
    with Peak() as peak:
        history = [{"file": esri[y], "year": y} for y in range(FIRST_YEAR, base + 1)]
        _, demand, _, mode = change_analysis.run_change_analysis(history)
        demand = demand * (target - base)
    record("Demand (change analysis 2017 to base year)", peak, mode=mode)
    demand.to_csv(out / "demand.csv")
    report = {}
    with Peak() as peak:
        predicted = prediction.run_allocation_simulation(esri[base], demand, suitability, str(out),
                                                         stochastic=False, allocation_report=report)
    record("Allocation (run_allocation_simulation)", peak)

    result = {"tag": tag, "train": train, "base": base, "target": target, "seed": seed, "growth_modes": growth_modes,
              "transitions": [list(t) for t in transitions], "models": [list(k) for k in model_paths],
              "predicted": predicted, "stages": stages, "allocation": report,
              "platform_commit": (HERE / "data" / "platform_main" / "COMMIT").read_text().strip()}
    (out / "result.json").write_text(json.dumps(result, indent=1, default=float))
    return result


if __name__ == "__main__":
    import logging
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, nargs="+", required=True)
    ap.add_argument("--base", type=int, required=True)
    ap.add_argument("--target", type=int, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--stack-memory", action="store_true", help="only measure the in-memory predictor stack")
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    if a.stack_memory:
        s = platform_stack_memory(a.base)
        (RUNS / a.tag).mkdir(parents=True, exist_ok=True)
        (RUNS / a.tag / "stack_memory.json").write_text(json.dumps(s, indent=1))
        print(s, flush=True)
        sys.exit(0)
    run(a.train, a.base, a.target, a.seed, a.tag, log=lambda m: print(m, flush=True))
