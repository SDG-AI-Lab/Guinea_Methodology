"""
Two-stage national dumpsite scan, and the polygons it produces.

This is the deployment path. Everything else in this package measures the
model; this runs it.

WHY TWO STAGES
--------------
An 80 m window is what makes the model a waste detector rather than a
settlement detector (see the package README), but scanning every 80 m window
over the national footprint means millions of Google Static Maps calls. Stage 1
scores 250 m cells on FREE Esri imagery and stage 2 runs only inside the cells
it keeps. Measured on the frozen screen: it keeps 31.6% of cells and retains
85.5% of true sites, which puts a national scan of the settled footprint
(SMOD>=21 plus a 1 km fringe) at roughly $530-790 depending on how clustered
the kept cells are. Stage 1 is free, so it can afford to be generous.

The cost of the screen is real and must be stated: at a stage-1 threshold of
0.30 it discards 14.5% of true sites before stage 2 ever sees them. That is a
deliberate trade, not an oversight. End-to-end recall is this 85.5% multiplied
by stage 2's recall at whichever tier is displayed -- it is NOT the tier's
recall alone, and any published recall figure must say so.

STAGE-2 SOURCE: GOOGLE OR ESRI
------------------------------
Stage 2 can confirm on licensed Google z19 (the configuration the methodology
measured first) or on FREE Esri z17 with the head fitted on Esri
(probe_esri.joblib). The Esri route bills nothing, so the platform can run it
itself; its blind-grid precision and recall are measured separately
(dumpsite_detection_workflow.ipynb, Part 3) and must be the figures published
with an Esri scan -- the Google figures do not transfer.

CONFIDENCE TIERS, NOT A SINGLE CUT-OFF
--------------------------------------
Every polygon carries its raw score and a tier label. Global Forest Watch
publishes GLAD/RADD alerts as low confidence until corroborated and lets users
show only high-confidence ones; Global Plastic Watch runs three sensitivity
regimes with a confidence score per merged polygon. Neither picks one cut-off,
because the right one depends on the viewer's purpose. So the threshold here is
a display filter: nothing is discarded, and the tiers can be retuned later
without re-scanning.

WHAT THE OUTPUT IS, AND IS NOT
------------------------------
`status` is always `unverified`. These are photo-interpreted candidates from a
model whose own ground truth was photo-interpretation, never field survey. The
honest description is "AI-detected dumpsite candidates, not field-verified",
and `imagery_date` may be well over a year old -- a new detection means newly
detected, not newly appeared.
"""

import json
import logging
import math
from pathlib import Path

import torch  # noqa: F401  -- must precede the GEE/GDAL stack on Windows
import numpy as np
import pandas as pd

from . import inference as gi
from . import features as bm

logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
MODELS = HERE / "models"

SCREEN_TILE_M = 250.0
SCREEN_ZOOM = 17            # Esri's ceiling outside Conakry; free
SCREEN_THRESHOLD = 0.30     # generous: stage 1 is free, misses here are final
CONFIRM_TILE_M = 80.0
CONFIRM_ZOOM = 19           # Google; ~272 real pixels in an 80 m window

SCHEMA = ["geometry", "confidence", "tier", "area_ha", "n_windows", "status",
          "detection_date", "imagery_date", "model_version",
          "material_flags", "toxicity_flag"]

# Fallback only. The authoritative tiers are published in probe_meta.json by
# `freeze`, so they can be retuned without editing code or re-scanning.
DEFAULT_TIERS = {"high": 0.90, "medium": 0.70, "all": 0.50}


# Stage-2 imagery source -> the head fitted on that source. A head scored
# against another provider's pixels returns near-zero everywhere with no error
# (Google head on Esri: blind-grid ROC-AUC 0.483), so the pairing is enforced.
CONFIRM_HEADS = {"google": "probe.joblib", "esri": "probe_esri.joblib"}


def confirm_head_for(source):
    """The frozen stage-2 head that matches an imagery source."""
    if source not in CONFIRM_HEADS:
        raise ValueError("confirm_source must be one of %s, got %r"
                         % (sorted(CONFIRM_HEADS), source))
    return CONFIRM_HEADS[source]


def read_tiers(confirm_head="probe.joblib"):
    """Tier thresholds the frozen head was published with, without loading it.

    Callers that only need to filter an existing scan should use this rather
    than `load_head`, which deserialises the model for no reason.
    """
    meta_p = MODELS / (Path(confirm_head).stem + "_meta.json")
    if meta_p.exists():
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        return meta.get("tiers", DEFAULT_TIERS), meta.get("default_tier", "high")
    return DEFAULT_TIERS, "high"


def load_head(name):
    """Load a frozen probe head and its metadata."""
    import joblib
    p = MODELS / name
    meta_p = p.with_name(p.stem + "_meta.json")
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Freeze it first:\n"
            f"  python 08_point_model.py freeze --tile-m 80 --source google")
    meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
    return joblib.load(p), meta


# Every grid is anchored here, not at the AOI corner, so a tile_id is a pure
# function of position on Earth: the same ground always gets the same id and
# the same footprint, in any AOI and on any run. That is what makes the tile
# cache worth having -- a cached tile stays valid when the scan area changes.
ORIGIN_LON = -180.0
ORIGIN_LAT = -90.0


def _row_dlon(j, dlat, tile_m):
    """Longitude step for global grid row j, at that row's own latitude."""
    mid = ORIGIN_LAT + (j + 0.5) * dlat
    return tile_m / max(111_320.0 * math.cos(math.radians(mid)), 1.0)


def _window(j, i, dlat, tile_m):
    """One full (unclipped) window of the global grid, by row/column index."""
    dlon = _row_dlon(j, dlat, tile_m)
    lat = ORIGIN_LAT + j * dlat
    lon = ORIGIN_LON + i * dlon
    return {"tile_id": "r%07dc%07d" % (j, i), "minx": lon, "miny": lat,
            "maxx": lon + dlon, "maxy": lat + dlat}


def _windows(bounds, tile_m):
    """Global-grid windows of tile_m covering a lat/lon bbox."""
    minx, miny, maxx, maxy = bounds
    dlat = tile_m / 110_574.0
    j0 = int(math.floor((miny - ORIGIN_LAT) / dlat))
    j1 = int(math.ceil((maxy - ORIGIN_LAT) / dlat))
    rows = []
    for j in range(j0, j1):
        dlon = _row_dlon(j, dlat, tile_m)
        i0 = int(math.floor((minx - ORIGIN_LON) / dlon))
        i1 = int(math.ceil((maxx - ORIGIN_LON) / dlon))
        for i in range(i0, i1):
            rows.append(_window(j, i, dlat, tile_m))
    return pd.DataFrame(rows)


def _windows_for_cells(cells, tile_m):
    """Global-grid windows of tile_m intersecting any of the given cells.

    Deduplicated by tile_id, so ground shared between two adjacent kept cells
    is fetched, billed and scored exactly once.
    """
    dlat = tile_m / 110_574.0
    seen = {}
    for cell in cells.itertuples():
        j0 = int(math.floor((cell.miny - ORIGIN_LAT) / dlat))
        j1 = int(math.ceil((cell.maxy - ORIGIN_LAT) / dlat))
        for j in range(j0, j1):
            dlon = _row_dlon(j, dlat, tile_m)
            i0 = int(math.floor((cell.minx - ORIGIN_LON) / dlon))
            i1 = int(math.ceil((cell.maxx - ORIGIN_LON) / dlon))
            for i in range(i0, i1):
                w = _window(j, i, dlat, tile_m)
                seen.setdefault(w["tile_id"], w)
    return pd.DataFrame(list(seen.values()))


def _score(paths, model):
    """Score fetched tiles; tiles that failed to fetch score NaN, never 0."""
    keep = [i for i, p in enumerate(paths) if p]
    out = np.full(len(paths), np.nan)
    if not keep:
        return out
    X = bm.imagenet_features([paths[i] for i in keep])
    out[keep] = model.predict_proba(X)[:, 1]
    return out


def _tier_of(score, tiers):
    """Highest tier whose threshold the score meets, else None."""
    best = None
    for name, thr in sorted(tiers.items(), key=lambda kv: -kv[1]):
        if score >= thr:
            return name
        best = name
    return None


def scan(bounds, cache_dir, screen_head="probe_screen.joblib",
         confirm_head=None, budget=None, imagery_date=None,
         confirm_source="google"):
    """Run the two-stage cascade over a bbox and return candidate polygons.

    `confirm_source` picks the stage-2 imagery: "google" (licensed, billed per
    80 m window, Google z19) or "esri" (free, Esri z17, scored with the head
    fitted on Esri). `confirm_head` defaults to the head matching the source
    and is refused if its metadata names a different source.

    `budget` caps BILLABLE Google requests in stage 2; cached tiles are free.
    Always set it for a Google scan -- an unbounded run over a national
    footprint is a four-figure bill. It is ignored for "esri", which bills
    nothing.
    """
    import geopandas as gpd
    from shapely.geometry import box
    from shapely.ops import unary_union

    cache_dir = Path(cache_dir)
    confirm_head = confirm_head or confirm_head_for(confirm_source)
    screen, screen_meta = load_head(screen_head)
    confirm, confirm_meta = load_head(confirm_head)
    head_source = confirm_meta.get("source", "google")
    if head_source != confirm_source:
        raise ValueError(
            "stage-2 head %s was fitted on %s imagery but confirm_source is %r; "
            "a mismatched head scores at chance without any error"
            % (confirm_head, head_source, confirm_source))
    tiers = confirm_meta.get("tiers", DEFAULT_TIERS)
    floor = min(tiers.values())

    # ---- stage 1: free Esri screen at 250 m -------------------------------
    cells = _windows(bounds, SCREEN_TILE_M)
    logger.info("stage 1: %d cells of %.0fm from Esri z%d (free)",
                len(cells), SCREEN_TILE_M, SCREEN_ZOOM)
    paths, _ = gi.fetch_tiles_esri(cells, cache_dir / "screen", size_px=800,
                                   zoom=SCREEN_ZOOM)
    cells["score"] = _score(paths, screen)
    unscored = int(cells.score.isna().sum())
    if unscored:
        logger.warning("stage 1: %d/%d cells could not be fetched and were NOT "
                       "examined -- a coverage gap, not an absence of sites",
                       unscored, len(cells))
    kept = cells[cells.score >= SCREEN_THRESHOLD].copy()
    logger.info("stage 1 kept %d/%d cells (%.0f%%) at threshold %.2f",
                len(kept), len(cells), 100 * len(kept) / max(len(cells), 1),
                SCREEN_THRESHOLD)
    if kept.empty:
        return gpd.GeoDataFrame(columns=SCHEMA, geometry="geometry",
                                crs="EPSG:4326")

    # ---- stage 2: confirm at 80 m, only inside kept cells -----------------
    win = _windows_for_cells(kept, CONFIRM_TILE_M)
    if confirm_source == "esri":
        zoom = int(confirm_meta.get("zoom", SCREEN_ZOOM))
        logger.info("stage 2: %d windows of %.0fm from Esri z%d (free)",
                    len(win), CONFIRM_TILE_M, zoom)
        # Own cache folder: an Esri window must never be read as a Google one.
        wpaths, _ = gi.fetch_tiles_esri(win, cache_dir / "confirm_esri",
                                        size_px=800, zoom=zoom)
    else:
        logger.info("stage 2: %d windows of %.0fm from Google z%d (billable)",
                    len(win), CONFIRM_TILE_M, CONFIRM_ZOOM)
        wpaths = gi.fetch_tiles_google(win, cache_dir / "confirm", size_px=800,
                                       zoom=CONFIRM_ZOOM, budget=budget)
    win["score"] = _score(wpaths, confirm)

    unscored_w = int(win.score.isna().sum())
    if unscored_w:
        logger.warning("stage 2: %d/%d windows could not be fetched and were NOT "
                       "examined (budget exhausted, or no imagery)"
                       " -- a coverage gap, not an absence of sites",
                       unscored_w, len(win))
    hits = win[win.score >= floor].copy()
    logger.info("stage 2 flagged %d windows at or above the lowest tier (%.2f)",
                len(hits), floor)
    if hits.empty:
        return gpd.GeoDataFrame(columns=SCHEMA, geometry="geometry",
                                crs="EPSG:4326")

    # ---- merge touching windows into sites, within one score band ---------
    # Windows are merged only with touching windows of the SAME tier. Merging
    # across tiers let one strong window label a whole neighbourhood: on the
    # Conakry demo AOI (Esri stage 2) 298 of 422 windows reached the lowest
    # tier and dissolved into a single 133 ha polygon tagged "high" because one
    # window inside it scored 0.994. A "high" polygon now covers only ground
    # that itself scored high; a cluster spanning bands becomes adjacent
    # polygons, one per band.
    hits["geometry"] = [box(r.minx, r.miny, r.maxx, r.maxy)
                        for r in hits.itertuples()]
    hits["band"] = [_tier_of(s, tiers) for s in hits.score]
    g = gpd.GeoDataFrame(hits, geometry="geometry", crs="EPSG:4326")
    parts = []
    for band, grp in g.groupby("band", sort=False):
        merged = unary_union(grp.geometry.buffer(1e-6))
        parts += [(poly, grp) for poly in getattr(merged, "geoms", [merged])]

    rows = []
    today = pd.Timestamp.utcnow().date().isoformat()
    version = "imagenet-probe-%sm-%s" % (int(CONFIRM_TILE_M), head_source)
    for poly, grp in parts:
        inside = grp[grp.geometry.intersects(poly)]
        # A site's confidence is its STRONGEST window: a dump partly covered by
        # a weak window should not be penalised for the window's edge. Every
        # window in the polygon shares its tier, so this cannot promote it.
        conf = float(inside.score.max())
        # metres-squared -> hectares, at this polygon's own latitude
        lat = poly.centroid.y
        m2 = (poly.area * 110_574.0
              * 111_320.0 * math.cos(math.radians(lat)))
        rows.append({
            "geometry": poly, "confidence": round(conf, 4),
            "tier": _tier_of(conf, tiers),
            "area_ha": round(m2 / 10_000.0, 3),
            "n_windows": int(len(inside)),
            "status": "unverified",
            "detection_date": today,
            "imagery_date": imagery_date,
            "model_version": version,
            "material_flags": None, "toxicity_flag": None,
        })

    out = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
    logger.info("%d candidate sites (%s)", len(out),
                ", ".join(f"{k}={int((out.tier == k).sum())}" for k in tiers))
    return out[SCHEMA]
