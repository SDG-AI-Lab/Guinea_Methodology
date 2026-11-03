"""Runs and scores notebook 05's runs (decisions100.JOBS), one at a time; each finished piece is cached,
so the script can be restarted. Usage: python run05.py [method phase seed]   (one run only)
                                        python run05.py --winner method     (test and backtest, three seeds)"""

import json
import logging
import sys
import time

import decisions100 as M

D = M.D
logging.basicConfig(level=logging.ERROR)
log = lambda m: print(m, flush=True)


def scored(run_tag, phase, demand):
    path = D.allocate(run_tag, phase, "as it runs", demand)
    cache = path.parent / "score.json"
    if not cache.exists():
        cache.write_text(json.dumps(D.score(phase, path), default=float))
    return json.loads(cache.read_text())


if __name__ == "__main__":
    t0 = time.time()
    if sys.argv[1:2] == ["--winner"]:
        jobs = [(sys.argv[2], ph, s) for ph in ("test", "backtest") for s in M.SEEDS]
    elif len(sys.argv) > 1:
        jobs = [(sys.argv[1], sys.argv[2], int(sys.argv[3]))]
    else:
        jobs = M.JOBS
    for method, phase, seed in jobs:
        M.RUN[method](phase, seed, log=log)
        run_tag = D.tag(phase, seed, method)
        for demand in ("observed", "projected"):
            s = scored(run_tag, phase, demand)
            log(f"{run_tag} | {demand}: FoM {s['FoM']:.2f}, new {s['new-change FoM']:.2f}, "
                f"random {s['random FoM']:.2f} ({(time.time() - t0) / 60:.0f} min)")
    log("done")
