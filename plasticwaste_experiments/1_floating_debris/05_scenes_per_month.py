"""Sentinel-2 scenes behind each month's composite over the three coastal regions, with the
deployed scene filter, for the twelve months to October 2026, and how much of the water
keeps at least one clear observation after the deployed cloud mask."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from common import ROOT  # noqa: E402

import ee  # noqa: E402

from plasticwaste_tool import config as pwcfg  # noqa: E402

OUT = ROOT / "1_floating_debris" / "outputs"
sys.path.insert(0, str(Path(__file__).resolve().parent))
_spec = __import__("importlib").import_module("04_patches_on_tidal_flats")


def main():
    ee.Initialize(project=_spec.project())
    cfg = pwcfg.S2_CONFIG
    limit = pwcfg.DETECTION_CONFIG.get("sar_fallback_cloud_pct", cfg["cloud_threshold_pct"])
    rows = []
    months = pd.period_range("2025-11", "2026-10", freq="M")
    for key in ("conakry", "konkoure", "fatala"):
        b = pwcfg.AOI_CONFIG[key]["bbox"]
        aoi = ee.Geometry.Rectangle(list(b))
        for m in months:
            s, e = str(m.start_time.date()), str((m + 1).start_time.date())
            col = (ee.ImageCollection(cfg["collection"]).filterDate(s, e).filterBounds(aoi))
            kept = col.filter(ee.Filter.lt(cfg["cloud_property"], limit))
            n = col.size().getInfo()
            k = kept.size().getInfo()
            info = {"all": n, "kept": k, "clear_median": 0, "cloud": None}
            if k:
                clear = kept.map(lambda i: i.select("SCL").remap(cfg["scl_clear_classes"], [1] * len(cfg["scl_clear_classes"]), 0).rename("c"))
                info["clear_median"] = clear.sum().unmask(0).reduceRegion(ee.Reducer.median(), aoi, 200, bestEffort=True).get("c").getInfo()
                info["cloud"] = kept.aggregate_mean(cfg["cloud_property"]).getInfo()
            rows.append({"region": key, "month": str(m), **info})
            print(rows[-1], flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "05_scenes_per_month.csv", index=False)
    print(d.pivot(index="month", columns="region", values="clear_median"))
    print(f"limit {limit}")


if __name__ == "__main__":
    main()
