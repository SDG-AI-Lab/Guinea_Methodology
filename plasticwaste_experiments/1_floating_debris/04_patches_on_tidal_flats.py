"""
How many published patches lie on mapped tidal flats?

Takes every published patch of the reference month (live platform, January 2024,
regional runs and whole-coast run) and asks an independent map, the Murray et al.
(2019) global intertidal classification (30 m, 2014-2016), and JRC Global Surface
Water occurrence, what each patch sits on. A pixel on water in fewer than 90 % of
Landsat observations is not permanent water. Light Earth Engine calls only.
"""
import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from common import PROJECT, ROOT  # noqa: E402

import ee  # noqa: E402

OUT = ROOT / "1_floating_debris" / "outputs"


def project():
    for line in (PROJECT / ".env").read_text(encoding="utf-8").splitlines():
        if line.startswith("GEE_PROJECT_ID="):
            return line.split("=", 1)[1].strip().strip('"')
    return os.environ.get("GEE_PROJECT_ID")


def main():
    ee.Initialize(project=project())
    live = Path(sys.argv[1])
    patches = [p for p in json.loads(live.read_text()) if p["acquisition_date"].startswith("2024-01")]
    feats = [ee.Feature(ee.Geometry(p["geometry"]), {"id": p["id"], "run": p["run"], "area": p["area_m2"]})
             for p in patches]
    fc = ee.FeatureCollection(feats)
    tidal = ee.ImageCollection("UQ/murray/Intertidal/v1_1/global_intertidal").select("classification").mosaic().unmask(0).rename("tidal")
    tidal500 = tidal.focal_max(radius=150, units="meters").rename("tidal150")
    occ = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").unmask(0).rename("occ")
    img = tidal.addBands(tidal500).addBands(occ)
    res = img.reduceRegions(collection=fc, reducer=ee.Reducer.max().combine(ee.Reducer.min(), sharedInputs=True),
                            scale=30).getInfo()
    d = pd.DataFrame([f["properties"] for f in res["features"]])
    d.to_csv(OUT / "04_patch_surfaces.csv", index=False)
    names = {152: "Conakry", 154: "Fatala", 155: "Whole coast"}
    d["region"] = d.run.map(names)
    rows = []
    for k, g in list(d.groupby("region")) + [("All", d)]:
        rows.append({"region": k, "patches": len(g),
                     "on_tidal_flat_map": int((g.tidal_max > 0).sum()),
                     "within_150m_of_tidal_flat": int((g.tidal150_max > 0).sum()),
                     "occurrence_below_90": int((g.occ_min < 90).sum()),
                     "either": int(((g.tidal150_max > 0) | (g.occ_min < 90)).sum())})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "04_patch_surfaces_summary.csv", index=False)
    print(t.to_string())
    print(d.occ_min.describe())


if __name__ == "__main__":
    main()
