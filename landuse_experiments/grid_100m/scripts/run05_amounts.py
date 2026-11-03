"""Notebook 05: the blend's maps placed with each amount candidate (decisions100.AMOUNTS).
Usage: python run05_amounts.py validation | test backtest"""

import logging
import sys
import time

import decisions100 as M

logging.basicConfig(level=logging.ERROR)

if __name__ == "__main__":
    t0 = time.time()
    for phase in sys.argv[1:]:
        for seed in M.SEEDS:
            for kind in M.AMOUNTS:
                s = M.allocate_amount(phase, seed, kind)
                print(f"{phase} s{seed} | {kind}: FoM {s['FoM']:.2f}, new {s['new-change FoM']:.2f}, random {s['random FoM']:.2f}, "
                      f"changed {s['predicted change (cells)'] / 1e6:.2f} M ({(time.time() - t0) / 60:.0f} min)", flush=True)
