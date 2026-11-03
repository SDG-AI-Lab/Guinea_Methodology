"""Notebook 05, last check: the blend against the blend with contagion on a window no choice has
used, 2020 -> 2021 -> 2022 (decisions100 / deployed100 PHASES["fresh"]), three seeds each. Both
passed the validation rule, and the test and backtest windows picked the blend with contagion; a
window not used in any choice gives a fair estimate (Cawley & Talbot 2010). The rule, fixed before
the run: the blend with contagion is adopted if its mean beats the blend's by more than the larger
of the two seed spreads, with the observed amount. Usage: python run05_fresh.py [seed ...]
(one process per seed keeps memory down; without seeds it runs any missing ones, then decides)."""

import json
import sys

import decisions100 as M
from run05_second import run, log

DECISION = M.P100.RUNS / "run05_fresh_decision.json"
CANDIDATES = ("blend", "blendc")

if __name__ == "__main__":
    for seed in [int(a) for a in sys.argv[1:]] or M.SEEDS:
        for method in CANDIDATES:
            run(method, "fresh", seed)
    if sys.argv[1:]:
        sys.exit()
    S = M.scores()
    rows = {}
    for demand in ("observed", "projected"):
        v = S[(S.phase == "fresh") & (S.demand == demand)].groupby("method")["FoM"]
        mean, spread = v.mean(), v.max() - v.min()
        gain = mean["blendc"] - mean["blend"]
        rows[demand] = {"blend": mean["blend"], "blend spread": spread["blend"], "blendc": mean["blendc"],
                        "blendc spread": spread["blendc"], "gain": gain,
                        "passes": bool(gain > max(spread["blend"], spread["blendc"]))}
        log(f"fresh, {demand}: blend {mean['blend']:.2f}, blendc {mean['blendc']:.2f}, gain {gain:+.2f}, "
            f"passes {rows[demand]['passes']}")
    adopted = "blendc" if rows["observed"]["passes"] else "blend"
    DECISION.write_text(json.dumps({"adopted": adopted, "fresh": rows}, indent=1, default=float))
    log(f"adopted: {adopted}")
    log("done")
