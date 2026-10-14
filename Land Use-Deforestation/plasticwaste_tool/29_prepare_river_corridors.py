"""
Build river-corridor AOIs for Guinea, so the FDI can be run upstream.

WHY CORRIDORS RATHER THAN BOUNDING BOXES
----------------------------------------
The FDI threshold is mean + 3 sigma over the water population of whatever area
is analysed. A rectangular bounding box drawn over a river is mostly land, so
the water population is small and the threshold is estimated from too few
pixels to be stable. That is the Fatala failure (7.1.9): the run returned zero
detections not because the river was clean but because the anomaly threshold
was meaningless, and the pipeline correctly refused to call it assessed.

A corridor -- a buffer around the river centreline -- confines the analysis to
where water actually is, which raises the water pixel count for a given area
analysed and keeps imagery cost proportional to the thing being measured.

WHAT A CORRIDOR CANNOT FIX, AND WHY THE GUARD HAD TO CHANGE TOO
---------------------------------------------------------------
Corridors alone do NOT make the existing `min_water_fraction` guard work on
rivers, and it is worth being explicit about why, because the arithmetic is
unforgiving.

HydroRIVERS v1.0 centrelines are derived from HydroSHEDS at 15 arc-seconds,
roughly 500 m at this latitude, so a centreline can sit a few hundred metres
from the real channel. The buffer therefore has to be wide enough to contain
the channel despite that error -- 250 m either side is the minimum that is
safe. But a Guinean river at the 25 m3/s screening threshold is only about
36 m wide, so within a 500 m corridor the water fraction is around 7%, which
is BELOW the 10% fraction guard even though the corridor is doing its job.

The conclusion is that water FRACTION is simply the wrong statistic for a
river. What makes a mean + 3 sigma threshold trustworthy is the NUMBER of
water pixels behind it: the relative standard error of a standard deviation is
about 1/sqrt(2n), so n = 100 gives 7% and n = 1000 gives 2.2%. Fraction only
ever mattered as a proxy for count on a compact coastal box.

So this script produces the corridors, and `DETECTION_CONFIG` gains an absolute
`min_water_px` floor that applies alongside the fraction guard. On coastal AOIs
both are satisfied and nothing changes; on river AOIs the count guard is the
one that carries the decision.

A WHOLE RIVER SYSTEM IS NOT ONE FETCH -- IT HAS TO BE TILED
-------------------------------------------------------------
This was found the hard way: the first live run registered each system's
DISSOLVED corridor by its bounding box (`AOI_CONFIG[...]["bbox"]`, the only
geometry `api_connector.get_aoi_geometry` knows how to build), and one system
hung for two hours and never finished.

The reason is that a river winds, and a bounding box around a winding line is
mostly empty land. Measured across the 16 assessable Guinea systems, the
bounding box of the dissolved corridor was 10-300x the corridor's own area,
and the worst single case fetched a bounding box of 61,772 km^2 to analyse a
203 km^2 corridor. Summed, the 16 systems implied 252,497 km^2 of Sentinel-2
compositing -- larger than Guinea itself -- to look at 2,864 km^2 of actual
river.

Clipping the EXPORTED raster to the corridor polygon does not fix this: a
raster is rectangular, so Earth Engine still has to build and composite the
full bounding-box rectangle before anything is clipped or discarded. The
bounding box has to be made small in the first place.

The fix exploits the fact that a corridor is built from many short reaches
(median 2.5 km, 99th percentile 12.3 km here) rather than being one long line:
reaches are grouped into TILES by snapping each reach's centroid, in a
projected CRS, to a grid whose cell size is chosen so a filled cell's area is
about `--max-tile-km2` (default 2500 km^2, comparable to the existing coastal
AOIs: Conakry 2,428 km^2, Mangrove 6,074 km^2). Empty grid cells contain no
reach and are never created, so a winding river produces a chain of small
tiles following its path rather than one rectangle enclosing all of it.

Each tile is its own AOI and gets its OWN threshold, exactly like the five
coastal regions. This is a known simplification: a truly pooled per-SYSTEM
threshold (combining every tile's water population before thresholding once)
would be more statistically efficient for a system split into many small
tiles, but it requires restructuring `run_from_arrays` to accept a
pre-computed threshold rather than deriving one internally, which has not been
done. Recorded here as the natural next step rather than left silent.

WHICH RIVERS ARE KEPT
---------------------
Only rivers wide enough to be resolved by Sentinel-2. Width is estimated from
discharge by the global hydraulic-geometry relation of Andreadis et al. (2013),
w = 7.2 * Q^0.5, itself in the tradition of Leopold and Maddock (1953):

    Q =  10 m3/s  ->  w ~ 23 m  ~ 2.3 pixels
    Q =  25 m3/s  ->  w ~ 36 m  ~ 3.6 pixels
    Q =  50 m3/s  ->  w ~ 51 m  ~ 5.1 pixels
    Q = 100 m3/s  ->  w ~ 72 m  ~ 7.2 pixels

The default keeps Q >= 25 m3/s, about three pixels, below which mixed
land-water edge pixels dominate the channel and the "water population" is not
really water. This is a screening rule and not a claim about any single reach.

Sources, both free and redistributable:
  HydroRIVERS v1.0 (Lehner & Grill 2013), https://www.hydrosheds.org/
  geoBoundaries GIN ADM1, already in data/

Usage:
    python 29_prepare_river_corridors.py                 # defaults
    python 29_prepare_river_corridors.py --min-discharge 50 --buffer-m 300
"""

import argparse
import json
import math
from pathlib import Path

import geopandas as gpd
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
ROOT = HERE.parent

AFRICA_SHP = (DATA / "HydroRIVERS_africa" / "HydroRIVERS_v10_af_shp"
              / "HydroRIVERS_v10_af.shp")
BOUNDARY = ROOT / "data" / "geoBoundaries-GIN-ADM1.geojson"
OUT_DIR = DATA / "rivers"

# Sentinel-2 optical resolution. The pixel the FDI is computed on.
S2_PIXEL_M = 10.0

# Andreadis et al. (2013) global hydraulic geometry: w = a * Q^b.
WIDTH_A, WIDTH_B = 7.2, 0.5

# Default screening threshold: about three Sentinel-2 pixels of channel width.
DEFAULT_MIN_DISCHARGE_CMS = 25.0

# Half-width of the corridor. Must exceed HydroRIVERS centreline positional
# error (HydroSHEDS 15 arc-sec, ~500 m) or the buffer can miss the channel it
# is supposed to contain. Widening it costs area, not accuracy: the water mask
# inside the corridor is what the threshold is computed from.
DEFAULT_BUFFER_M = 250.0

# Grid cell target area for tiling a system's reaches into fetchable AOIs.
# Chosen to sit inside the range the pipeline already fetches successfully for
# the five coastal regions (1,484 - 6,074 km^2), so a river tile costs no more
# than an existing coastal run.
DEFAULT_MAX_TILE_KM2 = 2500.0

# UTM 28N. Covers Guinea; used wherever a projected CRS is needed for a
# distance or an area rather than a geographic one.
UTM_CRS = "EPSG:32628"


def estimated_width_m(discharge_cms):
    return WIDTH_A * (discharge_cms ** WIDTH_B)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-discharge", type=float,
                    default=DEFAULT_MIN_DISCHARGE_CMS,
                    help="m3/s; reaches below this are dropped as sub-pixel")
    ap.add_argument("--buffer-m", type=float, default=DEFAULT_BUFFER_M,
                    help="corridor half-width in metres")
    ap.add_argument("--min-system-km", type=float, default=10.0,
                    help="drop river systems shorter than this in total")
    ap.add_argument("--max-tile-km2", type=float,
                    default=DEFAULT_MAX_TILE_KM2,
                    help="target max bounding-box area per fetchable AOI")
    args = ap.parse_args()

    if not AFRICA_SHP.exists():
        raise SystemExit(f"HydroRIVERS not found at {AFRICA_SHP}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 1. national boundary ----
    adm = gpd.read_file(BOUNDARY)
    if adm.crs is None:
        adm = adm.set_crs("EPSG:4326")
    adm = adm.to_crs("EPSG:4326")
    national = adm.dissolve()
    bbox = tuple(national.total_bounds)
    print(f"Guinea boundary: {len(adm)} ADM1 units, bbox "
          f"{[round(v, 3) for v in bbox]}")

    # ---- 2. read only what falls in the bbox, then clip to the polygon ----
    # The continental file is 1.5 M reaches and 237 MB; a bbox read avoids
    # loading the rest of Africa into memory to throw it away.
    print("reading HydroRIVERS within bbox ...")
    riv = gpd.read_file(AFRICA_SHP, bbox=bbox)
    print(f"  {len(riv)} reaches in bbox")
    riv = gpd.clip(riv, national.geometry.iloc[0])
    riv = riv[~riv.geometry.is_empty & riv.geometry.notna()]
    print(f"  {len(riv)} reaches after clipping to the national boundary")

    # ---- 3. keep only reaches Sentinel-2 can resolve ----
    riv["width_m_est"] = riv["DIS_AV_CMS"].astype(float).map(estimated_width_m)
    riv["width_px_est"] = riv["width_m_est"] / S2_PIXEL_M
    kept = riv[riv["DIS_AV_CMS"].astype(float) >= args.min_discharge].copy()
    print(f"  {len(kept)} reaches at >= {args.min_discharge:g} m3/s "
          f"(est. width >= {estimated_width_m(args.min_discharge):.0f} m, "
          f"{estimated_width_m(args.min_discharge) / S2_PIXEL_M:.1f} px), "
          f"{kept['LENGTH_KM'].sum():.0f} km")
    if kept.empty:
        raise SystemExit("Nothing retained; lower --min-discharge.")

    # ---- 4. group reaches into river systems ----
    # MAIN_RIV is HydroRIVERS' own identifier for the river a reach belongs to.
    # Grouping matters because a single reach rarely carries enough water
    # pixels for a stable threshold, whereas a whole system does, and because
    # reaches of one river should share a threshold if they are to be compared
    # with each other at all.
    sysinfo = (kept.groupby("MAIN_RIV")
               .agg(length_km=("LENGTH_KM", "sum"),
                    max_discharge_cms=("DIS_AV_CMS", "max"),
                    n_reaches=("HYRIV_ID", "count"))
               .reset_index())
    sysinfo = sysinfo[sysinfo["length_km"] >= args.min_system_km]
    kept = kept[kept["MAIN_RIV"].isin(sysinfo["MAIN_RIV"])].copy()
    print(f"  {len(sysinfo)} river systems >= {args.min_system_km:g} km "
          f"({len(kept)} reaches)")

    # ---- 5. corridors ----
    # Buffer in a projected CRS so the distance is metres. EPSG:32628 is
    # UTM 28N, which covers Guinea.
    utm = kept.to_crs("EPSG:32628")
    utm["geometry"] = utm.geometry.buffer(args.buffer_m)
    corridors = (utm.dissolve(by="MAIN_RIV").reset_index()
                 .to_crs("EPSG:4326"))
    corridors = corridors.merge(sysinfo, on="MAIN_RIV", how="left")

    # Whole-system corridor: kept for context and map display, but NOT
    # registered as a fetchable AOI -- its bounding box is what caused the
    # 252,497 km^2 problem the module docstring describes.
    areas_km2 = corridors.to_crs(UTM_CRS).area / 1e6
    corridors["corridor_km2"] = areas_km2.values
    corridors["est_channel_width_m"] = (
        corridors["max_discharge_cms"].map(estimated_width_m))
    corridors["est_water_fraction"] = (
        corridors["est_channel_width_m"] / (2 * args.buffer_m)).clip(upper=1.0)
    corridors["est_water_px"] = (
        corridors["length_km"] * 1000.0 / S2_PIXEL_M
        * corridors["est_channel_width_m"] / S2_PIXEL_M)

    keep_cols = ["MAIN_RIV", "n_reaches", "length_km", "max_discharge_cms",
                 "corridor_km2", "est_channel_width_m", "est_water_fraction",
                 "est_water_px", "geometry"]
    corridors = corridors[keep_cols]
    corridors = corridors.sort_values("max_discharge_cms", ascending=False)

    lines_out = OUT_DIR / "guinea_rivers.geojson"
    corr_out = OUT_DIR / "guinea_river_corridors.geojson"
    kept[["HYRIV_ID", "MAIN_RIV", "LENGTH_KM", "DIS_AV_CMS", "ORD_CLAS",
          "width_m_est", "geometry"]].to_file(lines_out, driver="GeoJSON")
    corridors.to_file(corr_out, driver="GeoJSON")

    # ---- 6. tile each system into fetchable AOIs ----
    # Grid side in metres, chosen so a FILLED cell's area is ~max_tile_km2.
    # Reaches are grouped by snapping their UTM centroid to this grid; empty
    # cells are never created, so a winding river yields a chain of tiles
    # tracing its path rather than one rectangle enclosing all of it.
    side_m = (args.max_tile_km2 * 1e6) ** 0.5
    kept_utm = kept.to_crs(UTM_CRS).copy()
    cx = kept_utm.geometry.centroid.x
    cy = kept_utm.geometry.centroid.y
    kept_utm["_ix"] = (cx // side_m).astype(int)
    kept_utm["_iy"] = (cy // side_m).astype(int)
    kept_utm["_tile_key"] = (
        kept_utm["MAIN_RIV"].astype(int).astype(str) + "_"
        + kept_utm["_ix"].astype(str) + "_" + kept_utm["_iy"].astype(str))

    tile_rows = []
    tiles_geom = []
    for tile_key, grp in kept_utm.groupby("_tile_key"):
        main_riv = int(grp["MAIN_RIV"].iloc[0])
        poly = grp.geometry.buffer(args.buffer_m).unary_union
        b_utm = poly.bounds
        area_km2 = poly.area / 1e6
        tile_rows.append({
            "tile_key": tile_key, "main_riv": main_riv,
            "n_reaches": int(len(grp)),
            "length_km": float(grp["LENGTH_KM"].sum()),
            "max_discharge_cms": float(grp["DIS_AV_CMS"].max()),
            "tile_km2": area_km2,
        })
        tiles_geom.append(poly)

    tiles_utm = gpd.GeoDataFrame(tile_rows, geometry=tiles_geom, crs=UTM_CRS)
    tiles = tiles_utm.to_crs("EPSG:4326")
    tiles["est_channel_width_m"] = tiles["max_discharge_cms"].map(estimated_width_m)
    tiles["est_water_px"] = (
        tiles["length_km"] * 1000.0 / S2_PIXEL_M
        * tiles["est_channel_width_m"] / S2_PIXEL_M)

    # Order tiles along each system (upstream to downstream-ish, by mean
    # DIST_DN_KM of their reaches) purely for readable, stable numbering --
    # not used for anything geometric.
    dist_dn = kept.groupby(
        kept_utm["_tile_key"])["DIST_DN_KM"].mean() \
        if "DIST_DN_KM" in kept.columns else None
    tiles_out = OUT_DIR / "guinea_river_corridor_tiles.geojson"
    tiles.to_file(tiles_out, driver="GeoJSON")

    # ---- 7. the AOIs the FDI pipeline will actually consume ----
    aois = {}
    systems = {}
    per_system_counter = {}
    tiles_sorted = tiles.sort_values(
        ["main_riv"] + (["tile_key"]))
    for r in tiles_sorted.itertuples():
        n = per_system_counter.get(r.main_riv, 0) + 1
        per_system_counter[r.main_riv] = n
        key = f"river_{r.main_riv}_t{n:02d}"
        b = r.geometry.bounds
        bbox_km2 = ((b[2] - b[0]) * 111.32
                    * abs(__import__("math").cos(
                        __import__("math").radians((b[1] + b[3]) / 2)))
                    * (b[3] - b[1]) * 110.57)
        aois[key] = {
            "bbox": [round(v, 5) for v in b],
            "kind": "river_corridor",
            "main_riv": int(r.main_riv),
            "tile_key": r.tile_key,
            "length_km": round(float(r.length_km), 1),
            "corridor_km2": round(float(r.tile_km2), 1),
            "bbox_km2": round(float(bbox_km2), 1),
            "max_discharge_cms": round(float(r.max_discharge_cms), 1),
            "est_channel_width_m": round(float(r.est_channel_width_m), 1),
            "est_water_px": int(r.est_water_px),
        }
        systems.setdefault(str(int(r.main_riv)), []).append(key)

    system_summary = {}
    for r in corridors.itertuples():
        mk = str(int(r.MAIN_RIV))
        system_summary[mk] = {
            "main_riv": int(r.MAIN_RIV),
            "length_km": round(float(r.length_km), 1),
            "max_discharge_cms": round(float(r.max_discharge_cms), 1),
            "corridor_km2": round(float(r.corridor_km2), 1),
            "tiles": systems.get(mk, []),
            "n_tiles": len(systems.get(mk, [])),
        }

    total_bbox_km2 = sum(a["bbox_km2"] for a in aois.values())
    meta = {
        "source": "HydroRIVERS v1.0 (Lehner & Grill 2013), clipped to "
                  "geoBoundaries GIN ADM1",
        "min_discharge_cms": args.min_discharge,
        "buffer_m": args.buffer_m,
        "min_system_km": args.min_system_km,
        "max_tile_km2": args.max_tile_km2,
        "width_relation": f"w = {WIDTH_A} * Q^{WIDTH_B} "
                          f"(Andreadis et al. 2013)",
        "n_systems": int(len(corridors)),
        "n_tiles": len(aois),
        "n_reaches": int(len(kept)),
        "total_length_km": round(float(kept["LENGTH_KM"].sum()), 1),
        "total_corridor_km2": round(float(corridors["corridor_km2"].sum()), 1),
        "total_fetch_bbox_km2": round(total_bbox_km2, 1),
        "note": "Each entry in `aois` is a TILE, not a whole river system -- "
                "see the module docstring ('A whole river system is not one "
                "fetch'). `systems` groups tile keys by MAIN_RIV for "
                "reporting. est_water_fraction is REPORTED, not enforced; "
                "rivers are gated on absolute water pixel count "
                "(DETECTION_CONFIG['min_water_px']).",
        "systems": system_summary,
        "aois": aois,
    }
    (OUT_DIR / "river_aois.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\nwrote {lines_out.name}, {corr_out.name}, {tiles_out.name}, "
          f"river_aois.json in {OUT_DIR}")
    print(f"\n{len(corridors)} systems -> {len(aois)} tiles "
          f"(target {args.max_tile_km2:.0f} km2/tile)")
    print(f"total corridor area : {corridors['corridor_km2'].sum():,.0f} km2")
    print(f"total FETCH area    : {total_bbox_km2:,.0f} km2  "
          f"(bounding box of every tile, summed)")
    print(f"\n{'system':>12s} {'km':>7s} {'Q m3/s':>8s} {'tiles':>6s} "
          f"{'max tile km2':>13s}")
    for r in corridors.itertuples():
        sysk = str(int(r.MAIN_RIV))
        tks = systems.get(sysk, [])
        max_tile = max((aois[k]["bbox_km2"] for k in tks), default=0)
        print(f"{int(r.MAIN_RIV):>12d} {r.length_km:7.0f} "
              f"{r.max_discharge_cms:8.1f} {len(tks):6d} {max_tile:13.0f}")

    below = sum(1 for a in aois.values() if a["est_water_px"] < 1000)
    print(f"\n{below} of {len(aois)} tiles fall below the 1000-water-pixel "
          f"floor on the geometric estimate and will likely report "
          f"not-assessed; run `30_run_river_fdi.py --preflight` for the "
          f"real count.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
