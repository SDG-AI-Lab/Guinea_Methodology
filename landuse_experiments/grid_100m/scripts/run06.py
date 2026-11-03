"""Runs and scores notebook 06's runs, one process per run (memory is released between runs); each
finished piece is cached, so the script can be restarted.
  * the backend (main) on the fresh window 2020 -> 2021 -> 2022, three seeds: the window no choice used,
    for a fair estimate of the improvement (Cawley & Talbot 2010);
  * the blend's weight check: the document's map weighted 0.25 and 0.75 instead of 0.5, three seeds on
    validation, seed 42 on test and backtest.
Usage: python run06.py   (all, each in its own process);  python run06.py method phase seed"""

import json
import logging
import subprocess
import sys
import time

import decisions100 as M
from run05 import scored

D = M.D
logging.basicConfig(level=logging.ERROR)
t0 = time.time()
log = lambda m: print(f"{m} ({(time.time() - t0) / 60:.0f} min)", flush=True)
JOBS = ([("backend", "fresh", s) for s in M.SEEDS]
        + [(m, "validation", s) for m in ("blendw25", "blendw75") for s in M.SEEDS]
        + [(m, ph, 42) for m in ("blendw25", "blendw75") for ph in ("test", "backtest")])


def run(method, phase, seed):
    if method == "backend":
        D.run_backend(phase, seed, log=log)
    else:
        M.RUN[method](phase, seed, log=log)
    tag = D.tag(phase, seed, method)
    for demand in ("observed", "projected"):
        s = scored(tag, phase, demand)
        log(f"{tag} | {demand}: FoM {s['FoM']:.2f}, new {s['new-change FoM']:.2f}, random {s['random FoM']:.2f}")


if __name__ == "__main__":
    if len(sys.argv) == 4:
        run(sys.argv[1], sys.argv[2], int(sys.argv[3]))
    else:
        for method, phase, seed in JOBS:
            subprocess.run([sys.executable, __file__, method, phase, str(seed)], check=True)
        log("done")
