"""Copied from 09_national_footprint.py and renamed so it can be imported.

The numbered original stays in place for the notebooks; this copy is
what the package and the backend import. Runtime importlib loads of
sibling scripts have been rewritten as relative imports.
Regenerate with scratchpad/build_package.py -- do not edit by hand
until the originals are retired.
"""

"""
09 — National scan footprint and random labelling sample.

Two things that must be reproducible and were previously computed ad hoc:

  budget   How large is a "nation-wide" scan, and what does it cost?
  sample   Draw an unbiased labelling sample from that exact footprint.

WHY THE FOOTPRINT IS NOT THE WHOLE COUNTRY
------------------------------------------
Guinea is 244,006 km2 = 3,904,090 cells at 250 m. Every cell needs an HTTP
image fetch plus a CNN forward pass, and fetching — not compute — is the
binding cost. An exhaustive scan is ~5.6 days of continuous polite requests
against a free tile service, repeated on every refresh. That is not a design.

Informal dumping sits at the settlement fringe, so the scan is masked to
**GHSL Degree of Urbanisation (SMOD) class >= 21** (urban centre, dense
cluster, semi-dense, suburban) plus a 1 km fringe. SMOD is used rather than a
land-cover class because it is the EU/UN standard settlement classification
behind SDG indicator reporting — a defensible thing to cite in a methodology
rather than a self-defined buffer.

That mask is 89,336 cells, 2.29% of the country, and still nation-wide in the
sense the SRS requires: every settlement in Guinea is covered, by an objective
published layer rather than a hand-picked list of towns.

Two findings from measuring it, both non-obvious:
  * "All settled land" (SMOD >= 11) is useless as a mask — very-low-density
    rural covers 99.5% of Guinea, indistinguishable from an exhaustive scan.
  * A 500 m fringe is indistinguishable from no fringe, because SMOD's 1 km
    pixels cannot resolve it. The meaningful choices are 0, 1 km and 2 km.

1 km is chosen: the empirical literature puts illegal dumping overwhelmingly
within 250–500 m of populated land with ~1 km capturing essentially all
validated sites, and 2 km costs 64% more for little gain.

WHY THE LABELLING SAMPLE MUST BE RANDOM
---------------------------------------
The project's first label set was 63% positive because it was collected by
looking for dumpsites. A probe fitted on that learns a prior wildly unlike
deployment (~2% of windows), so its scores are miscalibrated and any threshold
chosen on it does not transfer. Labels drawn uniformly at random from the scan
footprint fix the class prior, make the negatives representative of the ground
the model must reject, and make precision/recall unbiased for the population
actually being scanned.

Rows are shuffled across regions on purpose: label in file order and any prefix
of the file is still an unbiased, geographically spread sample.

Usage
-----
    python 09_national_footprint.py budget
    python 09_national_footprint.py budget --cell-m 250 --fringe 500 1000 2000
    python 09_national_footprint.py sample --n 600 --out manual_qa/regions
"""

import argparse
import json
import logging
import math
import os
import re
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

SMOD_ASSET = "JRC/GHSL/P2023A/GHS_SMOD/2030"
SMOD_URBAN_MIN = 21          # urban centre / dense cluster / semi-dense / suburban
GAUL0 = "FAO/GAUL/2015/level0"
GAUL1 = "FAO/GAUL/2015/level1"
COUNTRY = "Guinea"

FETCH_PER_S = 8.0            # polite sustained rate against a tile service
GPU_PER_S = 25.0             # ResNet50-FPN @800px, one modern GPU
CPU_PER_S = 2.5              # realistic Azure default with no GPU worker
GOOGLE_FREE = 10_000         # Static Maps free calls per month
GOOGLE_RATE = 2.0 / 1000     # USD per call thereafter


def init_ee():
    """Earth Engine via the project's existing service account."""
    import ee
    key = os.environ.get("GEE_SA_KEY_FILE") or os.environ.get("GEE_SA_KEY")
    if not key or not Path(key).exists():
        key = ROOT / "landuse_tool" / "credentials" / \
            "tribal-mapper-275801-7b57ce84f3bd.json"
    key = Path(key)
    if not key.exists():
        raise SystemExit(f"GEE service-account key not found: {key}")
    sa = json.loads(key.read_text(encoding="utf-8"))["client_email"]
    project = os.environ.get("GEE_PROJECT_ID", "tribal-mapper-275801")
    ee.Initialize(ee.ServiceAccountCredentials(sa, str(key)), project=project)
    return ee


def _country(ee):
    return ee.FeatureCollection(GAUL0).filter(
        ee.Filter.eq("ADM0_NAME", COUNTRY)).geometry()


def budget(cell_m, fringes):
    ee = init_ee()
    geom = _country(ee)
    area = geom.area(maxError=1000).divide(1e6).getInfo()
    cell_km2 = (cell_m / 1000.0) ** 2
    total = area / cell_km2

    smod = ee.Image(SMOD_ASSET).select("smod_code")
    px = ee.Image.pixelArea()

    def km2(mask):
        return mask.multiply(px).reduceRegion(
            reducer=ee.Reducer.sum(), geometry=geom, scale=1000,
            maxPixels=1e10, bestEffort=True).values().get(0).getInfo() / 1e6

    def fmt(h):
        return f"{h:,.1f}h" if h < 48 else f"{h / 24:,.1f}d"

    def row(name, a):
        n = a / cell_km2
        calls = n * (250.0 / cell_m) ** 2 if cell_m != 250 else n
        cost = max(calls - GOOGLE_FREE, 0) * GOOGLE_RATE
        print(f"{name:30s} {a:>9,.0f} {n:>11,.0f} {100 * n / total:>7.2f}% "
              f"{fmt(n / FETCH_PER_S / 3600):>9} {fmt(n / GPU_PER_S / 3600):>8} "
              f"{fmt(n / CPU_PER_S / 3600):>9} {cost:>9,.0f}")
        return n

    print(f"\n{COUNTRY}: {area:,.0f} km2, {total:,.0f} cells at {cell_m:.0f} m\n")
    print(f"{'scan footprint':30s} {'km2':>9} {'cells':>11} {'%ctry':>8} "
          f"{'fetch':>9} {'GPU':>8} {'CPU':>9} {'USD*':>9}")
    print("-" * 100)
    row("whole country", area)
    urban = smod.gte(SMOD_URBAN_MIN)
    row(f"SMOD>={SMOD_URBAN_MIN} (urban+suburban)", km2(urban))
    for f in fringes:
        m = urban.focal_max(radius=f, units="meters", kernelType="circle")
        row(f"  + {f:.0f}m fringe", km2(m))
    row("all settled (SMOD>=11)", km2(smod.gte(11).And(smod.neq(10))))
    print("-" * 100)
    print("* USD assumes Google Static Maps at $2/1000 after 10,000 free/month;")
    print("  zero if Esri is used. CPU column assumes no GPU worker.")


def sample(n_total, out_dir, cell_m, fringe, seed):
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import box

    ee = init_ee()
    adm1 = ee.FeatureCollection(GAUL1).filter(
        ee.Filter.eq("ADM0_NAME", COUNTRY))
    regions = sorted(adm1.aggregate_array("ADM1_NAME").getInfo())
    logger.info("%d regions: %s", len(regions), ", ".join(regions))

    mask = (ee.Image(SMOD_ASSET).select("smod_code").gte(SMOD_URBAN_MIN)
            .focal_max(radius=fringe, units="meters", kernelType="circle")
            .selfMask().rename("m"))

    per = max(n_total // len(regions), 1)
    rows = []
    for rname in regions:
        g = adm1.filter(ee.Filter.eq("ADM1_NAME", rname)).geometry()
        # The mask covers ~2% of the country, so sampling the region and
        # discarding misses throws away most draws. Scatter many random points,
        # keep those inside the mask, take the first `per`.
        pts = ee.FeatureCollection.randomPoints(region=g, points=per * 120,
                                                seed=seed)
        hit = (mask.unmask(0)
               .sampleRegions(collection=pts, scale=cell_m, geometries=True)
               .filter(ee.Filter.eq("m", 1)))
        feats = hit.limit(per).getInfo()["features"]
        slug = re.sub(r"[^a-z0-9]+", "", rname.lower())[:4]
        for i, f in enumerate(feats):
            lon, lat = f["geometry"]["coordinates"]
            rows.append({"tile_id": f"nat_{slug}_{i:04d}", "region": rname,
                         "lat": lat, "lon": lon})
        logger.info("  %-22s %4d cells", rname, len(feats))

    df = pd.DataFrame(rows)
    geoms = []
    for r in df.itertuples():
        hlat = (cell_m / 2) / 110_574.0
        hlon = (cell_m / 2) / max(111_320.0 * math.cos(math.radians(r.lat)), 1.0)
        geoms.append(box(r.lon - hlon, r.lat - hlat, r.lon + hlon, r.lat + hlat))

    gdf = gpd.GeoDataFrame(df.drop(columns=["lat", "lon"]),
                           geometry=geoms, crs="EPSG:4326")
    for c in ("label", "confidence", "notes", "annotator", "lat", "long"):
        gdf[c] = None
    gdf = gdf.sample(frac=1.0, random_state=0).reset_index(drop=True)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gp = out_dir / "national_random_sample.geojson"
    if gp.exists():
        raise SystemExit(
            f"{gp} already exists and may contain labels. Move it aside first "
            "— overwriting would destroy labelling work.")
    gdf.to_file(gp, driver="GeoJSON")
    logger.info("%d cells -> %s", len(gdf), gp)
    print("\nLabel in FILE ORDER: rows are shuffled across regions, so any "
          "prefix\nis still an unbiased, geographically spread sample.")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("budget", help="scan size, time and cost")
    b.add_argument("--cell-m", type=float, default=250.0)
    b.add_argument("--fringe", type=float, nargs="+", default=[500, 1000, 2000])

    s = sub.add_parser("sample", help="draw a random labelling sample")
    s.add_argument("--n", type=int, default=600)
    s.add_argument("--out", default=str(HERE / "manual_qa" / "regions"))
    s.add_argument("--cell-m", type=float, default=250.0)
    s.add_argument("--fringe", type=float, default=1000.0)
    s.add_argument("--seed", type=int, default=42)

    a = ap.parse_args()
    if a.cmd == "budget":
        budget(a.cell_m, a.fringe)
    else:
        sample(a.n, a.out, a.cell_m, a.fringe, a.seed)


if __name__ == "__main__":
    main()
