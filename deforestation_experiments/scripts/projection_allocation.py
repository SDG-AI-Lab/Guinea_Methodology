"""Notebook 07: how the platform's baseline projection places the projected loss, against Verra's allocation.

The platform's projection (scenario.allocate_future_loss) marks forest as cleared in order of risk,
highest first, until the projected hectares are used up: whole cells of the half-cell forest mask, each
counted at the mean cell area. VT0007 v1.0 (UDef-A) spreads the loss instead: each risk class gets
the loss rate it had in the calibration window, times each cell's forest, scaled to the total.

Both are given the observed total of the test window, so only where the loss goes differs, and both
are scored with Verra's prediction MedAE over the ~100,000 ha blocks. Maps: the improved model's and
its rank average with riskmapjnr's local loss rate (first seed, as notebooks 08-10).

  highest risk first, half-cell mask   the platform's rule
  highest risk first, shares           the same rule, each cell's forest in hectares from its shares
  VT0007 class rates                   Verra's allocation, as in every other notebook
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import improve as I                                              # noqa: E402
import model as MM                                               # noqa: E402
import tool_comparison as T                                      # noqa: E402
import verra as V                                                # noqa: E402

RULES = {"highest_half": "Highest risk first, half-cell mask (platform)",
         "highest_shares": "Highest risk first, forest from the shares",
         "vt0007": "VT0007 class rates"}
MAPS = {"improved": "Improved model", "rank": "Rank average with riskmapjnr"}


def highest_first(risk, forest_ha, demand):
    """Cells in order of risk, highest first, each losing all its forest, until `demand` ha (the last in part)."""
    ok = np.flatnonzero(np.isfinite(risk) & (forest_ha > 0))
    order = ok[np.argsort(-risk[ok], kind="stable")]
    cum = np.cumsum(forest_ha[order])
    k = int(np.searchsorted(cum, demand))
    out = np.zeros(risk.size)
    out[order[:k]] = forest_ha[order[:k]]
    if k < order.size:
        out[order[k]] = demand - (cum[k - 1] if k else 0.0)
    return out


def run_split(split, log=print):
    t1, t2, t3 = split
    area = MM.hansen()["area"]
    f1, f2 = MM.forest_pct(t1), MM.forest_pct(t2)
    observed = MM.loss_pct(t2, t3) / 100 * area
    _, (rmj_fit, _), _ = T.riskmapjnr(split, log=lambda *a: None, same_timing=True)
    fit, test = I.load_map("improved_fit", split), I.load_map("improved_test", split)
    rmj_test = I.load_map("rmj_test", split)
    maps = {"improved": (fit, test),
            "rank": ((T._rank(fit, f1 > 0) + T._rank(rmj_fit, f1 > 0)) / 2,
                     (T._rank(test, f2 > 0) + T._rank(rmj_test, f2 > 0)) / 2)}
    half = f2 >= 50
    out = {"split": MM.M.split_label(split), "maps": {}}
    for key, (risk_fit, risk_test) in maps.items():
        r = out["maps"][key] = {
            "vt0007": V.vt0007_tests(split, risk_fit, risk_test)["prediction_medae_ha"],
            "highest_shares": V.medae(highest_first(risk_test, f2 / 100 * area, observed.sum()), observed),
            "highest_half": V.medae(highest_first(np.where(half, risk_test, np.nan), np.where(half, area, 0.0),
                                                  observed.sum()), observed)}
        log(f"{out['split']}: {key} {r}", flush=True)
    return out


def split_name(split):
    return "projection_allocation_{}_{}_{}".format(*split)


if __name__ == "__main__":
    for split in MM.SPLITS:
        MM.cached(split_name(split), lambda: run_split(split), rerun="--rerun" in sys.argv)
        print("SAVED", split_name(split), flush=True)
