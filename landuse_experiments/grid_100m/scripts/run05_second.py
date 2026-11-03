"""Notebook 05, second round: contagion and FLUS's network added to the blend (decisions100.SECOND),
chosen on the validation window with the rule of the first round, then the chosen one on test and
backtest. Each finished piece is cached, so the script can be restarted. Usage: python run05_second.py;
python run05_second.py heldout scores the other candidates on test and backtest (diagnostic, not to choose)."""

import json
import logging
import sys
import time

import decisions100 as M
from run05 import scored

D = M.D
logging.basicConfig(level=logging.ERROR)
t0 = time.time()
log = lambda m: print(f"{m} ({(time.time() - t0) / 60:.0f} min)", flush=True)
DECISION = M.P100.RUNS / "run05_second_decision.json"


def run(method, phase, seed):
    M.RUN[method](phase, seed, log=log)
    for demand in ("observed", "projected"):
        s = scored(D.tag(phase, seed, method), phase, demand)
        log(f"{D.tag(phase, seed, method)} | {demand}: FoM {s['FoM']:.2f}, new {s['new-change FoM']:.2f}, "
            f"random {s['random FoM']:.2f}")


def choose():
    """The candidate with the highest mean on validation (observed amount) among those that beat the
    blend by more than the larger of the two seed spreads, if any."""
    S = M.scores()
    v = S[(S.phase == "validation") & (S.demand == "observed")].groupby("method")["FoM"]
    mean, spread = v.mean(), v.max() - v.min()
    rows = {m: {"mean": mean[m], "spread": spread[m], "gain": mean[m] - mean["blend"],
                "passes": bool(mean[m] - mean["blend"] > max(spread[m], spread["blend"]))}
            for m in ("blend", "contagion", "flusnet", *M.SECOND) if m in mean}
    passing = [m for m in M.SECOND if rows.get(m, {}).get("passes")]
    winner = max(passing, key=lambda m: rows[m]["mean"]) if passing else None
    DECISION.write_text(json.dumps({"winner": winner, "validation": rows}, indent=1, default=float))
    return winner, rows


if __name__ == "__main__":
    if sys.argv[1:] == ["heldout"]:                                 # diagnostic only, not used to choose
        for method in ("contagion", "flusnet", "blendc", "blend3"):
            for phase in ("test", "backtest"):
                for seed in M.SEEDS:
                    run(method, phase, seed)
        log("done")
        sys.exit()
    for method in ("contagion", "flusnet", *M.SECOND):
        for seed in M.SEEDS:
            run(method, "validation", seed)
    winner, rows = choose()
    for m, r in rows.items():
        log(f"validation, observed: {m} {r['mean']:.2f} (spread {r['spread']:.2f}), gain {r['gain']:+.2f}, passes {r['passes']}")
    log(f"winner: {winner}")
    if winner:
        for phase in ("test", "backtest"):
            for seed in M.SEEDS:
                run(winner, phase, seed)
    log("done")
