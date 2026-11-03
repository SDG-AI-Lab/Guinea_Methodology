"""Runs and scores everything notebook 04 shows (each finished piece is cached, so the script can be
restarted). Usage: python run04.py"""

import json
import logging
import time

import deployed100 as D

logging.basicConfig(level=logging.ERROR)
log = lambda m: print(m, flush=True)

JOBS, FIXES = D.JOBS, D.FIXES


def scored(run_tag, phase, variant, demand):
    path = D.allocate(run_tag, phase, variant, demand)
    cache = path.parent / "score.json"
    if not cache.exists():
        cache.write_text(json.dumps(D.score(phase, path), default=float))
    return json.loads(cache.read_text())


if __name__ == "__main__":
    t0 = time.time()
    for method, phase, seed in JOBS:
        if method == "document":
            D.run_document(phase, seed, log=log)
        else:
            D.run_backend(phase, seed, growth=(method == "growth"), log=log)
        run_tag = D.tag(phase, seed, method)
        variants = D.variants(method, seed)
        for variant in variants:
            for demand in ("projected", "observed"):
                s = scored(run_tag, phase, variant, demand)
                log(f"{run_tag} | {variant} | {demand}: FoM {s['FoM']:.2f}, new {s['new-change FoM']:.2f}, "
                    f"random {s['random FoM']:.2f} ({(time.time() - t0) / 60:.0f} min)")
    log("done")
