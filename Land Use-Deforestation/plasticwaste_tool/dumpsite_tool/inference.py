"""Copied from 05_guinea_inference.py and renamed so it can be imported.

The numbered original stays in place for the notebooks; this copy is
what the package and the backend import. Runtime importlib loads of
sibling scripts have been rewritten as relative imports.
Regenerate with scratchpad/build_package.py -- do not edit by hand
until the originals are retired.
"""

"""
05 — Run the dumpsite classifier over a Guinea AOI and export a ranked
candidate list.

This is the script that breaks the chicken-and-egg you are stuck in:

    "I can't train for Guinea without labels"
    "I can't get labels without knowing where to look"

The resolution is that **the model's output IS the annotation queue**. A model
trained entirely on foreign data is not accurate enough to report, but it is
far better than random at deciding *which 2,000 tiles a human should look at
first*. You verify those tiles, and the verifications become your Guinea
training labels.

So the deliverable here is NOT an accuracy figure. It is a prioritised work
queue. Two things fall out of it:

  1. Guinea training labels (what you verify becomes ground truth)
  2. **precision@N**, which IS measurable from verification alone — no
     exhaustive labelling needed. See `precision_at_n()` below.

Imagery sources, in order of preference:

  nicfi   Planet NICFI basemaps via GEE, 4.77 m, FREE, monthly.
          Requires NICFI sign-up. Coarse — only reliable for LARGE sites
          (roughly >30 m across). Good enough to start, and it costs nothing.
  s2      Sentinel-2 via GEE, 10 m. Too coarse for individual dumpsites;
          included only for pipeline testing since your GEE auth already works.
  gstatic Google Maps Static API, ~30 cm. What Guatemala used, and the right
          production source. Costs money and has caching terms — read them.

Usage:
    python 05_guinea_inference.py --aoi conakry --checkpoint runs/best.pt \
        --source nicfi --tile-m 300 --top-n 2000 --out queue/
"""

import argparse
import json
import logging
import io as _io
import os
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Import torch FIRST, before anything Earth-Engine-related (ee, and the
# grpc/protobuf stack it drags in) gets a chance to load. Both ship their own
# native Windows DLLs, and whichever loads into the process first wins --
# score_tiles() used to `import torch` lazily at the very end of a run, well
# after main() had already initialized `ee`, and that ordering made torch's
# c10.dll fail its OWN init routine (not "missing", "already loaded and
# conflicting") on every run that got that far. Importing it here, before
# `ee` is ever touched, avoids the conflict instead of just working around it.
import torch  # noqa: F401

# Run standalone (as this script's own docstring shows: `python
# 05_guinea_inference.py --aoi conakry ...`) or via subprocess.run() from a
# notebook, neither of which puts the project root on sys.path -- only the
# script's own directory (plasticwaste_tool/) gets added automatically, but
# `from plasticwaste_tool import api_connector` below needs the PARENT of
# that directory. Add it explicitly so this doesn't depend on the caller's cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Standard ImageNet stats -- matches both this project's own
# 02_train_baseline.py AND the AerialWaste architecture's documented
# preprocessing (utils/imutils.py normalize_image() defaults), so this is one
# shared constant, not a coincidence between two unrelated choices.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Esri serves a flat "Map data not yet available" placeholder instead of an
# error where it lacks coverage, so blank tiles must be detected by content.
# Measured on this project's data: real tiles have pixel std ~22-40 and mean
# gradient magnitude ~2-7; every one of 118 placeholder tiles had std exactly
# 5.26 and gradient ~0.37. The thresholds sit well clear of both clusters.
PLACEHOLDER_STD_MAX = 12.0
PLACEHOLDER_DETAIL_MAX = 1.5
MIN_ESRI_ZOOM = 15          # below this a 40m tile is meaningless anyway
MAX_ZOOM_DOWNGRADE = 4      # how many zoom levels to walk down before giving up


def _is_placeholder(pil_img):
    """True if an image looks like Esri's blank no-coverage placeholder."""
    import numpy as _np
    a = _np.asarray(pil_img.convert("RGB"), dtype=_np.float32)
    g = a.mean(axis=2)
    gy, gx = _np.gradient(g)
    detail = float(_np.hypot(gx, gy).mean())
    return float(a.std()) < PLACEHOLDER_STD_MAX and detail < PLACEHOLDER_DETAIL_MAX


# ---------------------------------------------------------------------------
# Tiling
# ---------------------------------------------------------------------------
def make_tile_grid(bounds, tile_m=300.0):
    """
    Build a tile grid over a lat/lon bounding box.

    Guatemala used ~1000 px at ~30 cm GSD = ~300 m tiles. Keeping the same
    ground footprint means results stay comparable to the published reference
    even if the pixel resolution differs.

    Longitude degree length shrinks with latitude, so it is computed per-row
    rather than assumed constant. At Conakry (~9.5 N) the error from ignoring
    this is ~1.4%; it grows fast further from the equator.
    """
    minx, miny, maxx, maxy = bounds
    deg_lat = tile_m / 110_574.0

    tiles = []
    lat = miny
    row = 0
    while lat < maxy:
        mid_lat = lat + deg_lat / 2.0
        # metres per degree longitude at this latitude
        m_per_deg_lon = 111_320.0 * math.cos(math.radians(mid_lat))
        deg_lon = tile_m / max(m_per_deg_lon, 1.0)

        lon = minx
        col = 0
        while lon < maxx:
            tiles.append({
                "tile_id": f"r{row:04d}c{col:04d}",
                "minx": lon, "miny": lat,
                "maxx": min(lon + deg_lon, maxx),
                "maxy": min(lat + deg_lat, maxy),
                "center_lon": lon + deg_lon / 2.0,
                "center_lat": mid_lat,
            })
            lon += deg_lon
            col += 1
        lat += deg_lat
        row += 1

    logger.info("Tile grid: %d tiles of %.0f m over bounds %s",
                len(tiles), tile_m, bounds)
    return pd.DataFrame(tiles)


def make_poi_grid(points_df, tile_m=40.0, id_prefix="poi"):
    """
    Build one tile polygon centered on each point of interest (POI) --
    e.g. Google-listed waste collection/management sites found outside the
    systematic browsing grid. Same footprint convention and output schema
    (tile_id, minx/miny/maxx/maxy, center_lon/center_lat) as make_tile_grid,
    so POI-derived and systematic-grid tiles can be labeled in the same CSV
    and fetched through the same fetch_tiles_esri() path -- a POI is just a
    1-tile grid, not a different kind of object.

    points_df needs `lat`/`lon` columns; an optional `poi_id` column gives
    readable tile_ids (e.g. "poi_wastecollector3"), otherwise a zero-padded
    row index is used. tile_m defaults to 40 to match browsing_grid.geojson
    -- keep it matched so fine-tuning sees a consistent image footprint
    across both label sources. An optional per-row `tile_m` column overrides
    the default for that one point (e.g. a site described as "a wide area,
    not only one point" needs a bigger footprint than a single 40m tile).
    """
    import re

    has_row_tile_m = "tile_m" in points_df.columns

    tiles = []
    for i, row in enumerate(points_df.itertuples()):
        lat, lon = float(row.lat), float(row.lon)
        row_tile_m = tile_m
        if has_row_tile_m:
            v = getattr(row, "tile_m", None)
            if v is not None and not (isinstance(v, float) and math.isnan(v)) and str(v).strip() != "":
                row_tile_m = float(v)
        deg_lat = row_tile_m / 110_574.0
        m_per_deg_lon = 111_320.0 * math.cos(math.radians(lat))
        deg_lon = row_tile_m / max(m_per_deg_lon, 1.0)

        raw_id = getattr(row, "poi_id", None) or f"{i:04d}"
        safe_id = re.sub(r"[^A-Za-z0-9_-]+", "_", str(raw_id)).strip("_")
        tiles.append({
            "tile_id": f"{id_prefix}_{safe_id}",
            "minx": lon - deg_lon / 2.0, "miny": lat - deg_lat / 2.0,
            "maxx": lon + deg_lon / 2.0, "maxy": lat + deg_lat / 2.0,
            "center_lon": lon, "center_lat": lat, "tile_m": row_tile_m,
        })

    df = pd.DataFrame(tiles)
    dup = df.tile_id[df.tile_id.duplicated()]
    if len(dup):
        logger.warning("%d duplicate POI tile_id(s) after sanitizing -- give "
                       "distinct poi_id values: %s", len(dup), dup.tolist()[:10])
    logger.info("POI grid: %d tiles, default %.0f m (%d with a per-row override)",
                len(df), tile_m, int((df.tile_m != tile_m).sum()))
    return df


def estimate_cost(n_tiles, source):
    """
    Print what a full pass costs, because this is the number that decides
    whether national coverage is affordable.
    """
    logger.info("=" * 58)
    logger.info("COVERAGE ESTIMATE")
    logger.info("  tiles in this AOI      : %s", f"{n_tiles:,d}")
    if source == "gstatic":
        # Google Static Maps is billed per 1000 requests; rate varies by
        # contract, so this is an order-of-magnitude figure only.
        logger.info("  approx cost @ $2/1000  : $%.2f", n_tiles * 0.002)
        logger.info("  -> screening with a susceptibility prior is what makes")
        logger.info("     national coverage affordable. Guinea is 245,857 km2")
        logger.info("     = ~2.7M tiles at this size.")
    elif source == "esri":
        logger.info("  cost                   : free tile endpoint (no API key)")
        logger.info("  -> but see fetch_tiles_esri()'s docstring: this endpoint's")
        logger.info("     ToS is written for interactive map display, not bulk")
        logger.info("     classification -- check licensing before this volume")
        logger.info("     becomes a production/scheduled platform job.")
    else:
        logger.info("  cost                   : free (GEE quota only)")
    logger.info("=" * 58)


# ---------------------------------------------------------------------------
# Imagery fetch
# ---------------------------------------------------------------------------
def fetch_tiles_gee(tiles_df, out_dir, source="nicfi", year=2024, month=1,
                    size_px=256, max_tiles=None):
    """
    Download tile thumbnails from GEE. Resume-aware: existing files are reused,
    matching the caching convention used elsewhere in the package.
    """
    import ee
    import requests

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if source == "nicfi":
        col = ee.ImageCollection("projects/planet-nicfi/assets/basemaps/africa")
        img = (col.filterDate(f"{year}-{month:02d}-01",
                              f"{year}-{month:02d}-28")
                  .first())
        vis = {"bands": ["R", "G", "B"], "min": 64, "max": 5454, "gamma": 1.8}
    elif source == "s2":
        col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                 .filterDate(f"{year}-{month:02d}-01", f"{year}-{month:02d}-28")
                 .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20)))
        img = col.median()
        vis = {"bands": ["B4", "B3", "B2"], "min": 0, "max": 3000}
    else:
        raise ValueError(f"source '{source}' is not a GEE source")

    df = tiles_df if max_tiles is None else tiles_df.head(max_tiles)
    paths, failures = [], 0

    for i, row in enumerate(df.itertuples(), 1):
        p = out_dir / f"{row.tile_id}.png"
        if p.exists():
            paths.append(str(p))
            continue
        try:
            region = ee.Geometry.Rectangle(
                [row.minx, row.miny, row.maxx, row.maxy])
            url = img.getThumbURL({**vis, "region": region,
                                   "dimensions": size_px, "format": "png"})
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            p.write_bytes(r.content)
            paths.append(str(p))
        except Exception as e:
            logger.warning("tile %s failed: %s", row.tile_id, e)
            paths.append(None)
            failures += 1

        if i % 100 == 0:
            logger.info("  fetched %d/%d (%d failures)", i, len(df), failures)

    if failures:
        logger.warning("%d/%d tiles failed to download. Tiles with no imagery "
                       "are excluded from the queue, NOT scored as negative.",
                       failures, len(df))
    return paths


def fetch_tiles_esri(tiles_df, out_dir, size_px=256, max_tiles=None, zoom=19, errors=None):
    """
    Returns `(paths, zoom_used)`: the image path per tile (None when it was
    not fetched), and the zoom each fetched tile was actually taken at, keyed
    by tile id (feature 006, toolchain.md §3: the capture date is looked up
    for that zoom). A tile already on disk was fetched earlier and has no
    entry. Callers that need only the paths take `[0]`. `errors`, when a dict
    is given, receives the exception of each tile that failed, keyed by tile
    id, so a caller can tell a refused request from missing imagery.

    Fetch VHR basemap tiles from Esri World Imagery (free ArcGIS Online tile
    service, no API key, no Earth Engine involved -- Esri's imagery isn't in
    GEE's public data catalog, so this bypasses `ee` entirely via the same
    `contextily` tile-fetch mechanism already used for this project's basemap
    plots). Resume-aware like `fetch_tiles_gee`.

    WHY THIS EXISTS: `nicfi` (4.77m) and `s2` (10m) both produced unusable
    output on a real 5,000-tile Guinea run -- not just "low confidence" but a
    model output dominated by a spurious brightness/texture shortcut
    (probability anti-correlated with image texture, corr ~ -0.3; a solid-
    black cloud-obscured tile scored 0.989, near the maximum). Diagnosis: the
    training data (INS drone photos, 1280x1280px, plausibly ~1-5cm/px given
    typical low-altitude survey flights) implies a ~13-65m frame footprint,
    while this module's tiles were 300m wide -- at Sentinel-2's 10m native
    resolution that leaves just 30 REAL pixels per tile side, upsampled 9x to
    fill a 256px thumbnail; the network was being shown interpolated blur, at
    completely the wrong scene scale, not real texture at any relevant scale.
    Esri World Imagery is commonly ~0.3-1m GSD in well-imaged urban/coastal
    areas (confirmed by direct inspection over Conakry), closing both gaps at
    once when paired with a smaller `tile-m` matching the training scale.

    Scientific basis: matching sensor/resolution class reduces but does not
    eliminate cross-domain performance loss (Tuia, Persello & Bruzzone 2016,
    "Domain Adaptation for the Classification of Remote Sensing Data", IEEE
    Geoscience and Remote Sensing Magazine) -- fine-tuning on real Guinea
    labels is still required regardless of imagery source. The Guatemala
    reference result this whole module benchmarks against (F1 0.68) was
    itself achieved on ~30cm-class imagery (Google Static Maps), not
    Sentinel-2/NICFI, so imagery resolution isn't incidental to that number.

    CAVEATS, do not skip these:
      - Esri's mosaic has non-uniform date/resolution/coverage by region --
        confirmed good over central Conakry by direct inspection; NOT
        verified across the rest of the AOI or wider Guinea. Spot-check a
        representative sample of your actual target area before trusting it.
      - Esri World Imagery's terms of service are written around interactive
        map display, not systematic bulk classification workloads. Fine for
        this notebook's prototyping/testing use; before relying on it for
        production platform deployment at volume, check licensing for your
        actual usage pattern -- an ArcGIS Image Service subscription may be
        the correct production path, not this free tile endpoint.
    """
    import contextily as cx
    from PIL import Image

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = tiles_df if max_tiles is None else tiles_df.head(max_tiles)
    paths, failures, downgraded, blank = [], 0, 0, 0
    zoom_used = {}

    for i, row in enumerate(df.itertuples(), 1):
        p = out_dir / f"{row.tile_id}.png"
        if p.exists():
            paths.append(str(p))
            continue
        got = None
        try:
            # Walk DOWN from the requested zoom until real imagery appears.
            # Esri serves a uniform "Map data not yet available" placeholder
            # rather than an error where it has no coverage at a given zoom,
            # so a naive fetch stores a blank image and everything downstream
            # treats it as a valid tile. That is not hypothetical: a first
            # Boke run at the default zoom 19 wrote 118 placeholders (pixel
            # std 5.26 on every single one vs ~32 for real tiles), and the
            # probe trained on them scored exactly chance -- a whole region's
            # labels wasted on blank images. Coverage there tops out at z17.
            for z in range(int(zoom), max(int(zoom) - MAX_ZOOM_DOWNGRADE, MIN_ESRI_ZOOM) - 1, -1):
                img, _ = cx.bounds2img(
                    row.minx, row.miny, row.maxx, row.maxy,
                    zoom=z, source=cx.providers.Esri.WorldImagery, ll=True)
                rgb = Image.fromarray(img).convert("RGB").resize((size_px, size_px))
                if not _is_placeholder(rgb):
                    got = (rgb, z)
                    break
        except Exception as e:
            logger.warning("tile %s failed: %s", row.tile_id, e)
            if errors is not None:
                errors[row.tile_id] = e
            paths.append(None)
            failures += 1
            continue

        if got is None:
            # No real imagery at any zoom tried. Return None so the tile is
            # EXCLUDED, never scored -- same contract as a failed download.
            blank += 1
            paths.append(None)
            continue

        rgb, z = got
        rgb.save(p)
        zoom_used[row.tile_id] = z
        if z != int(zoom):
            downgraded += 1
        paths.append(str(p))

        if i % 100 == 0:
            logger.info("  fetched %d/%d (%d failures, %d zoom-downgraded, "
                        "%d no-coverage)", i, len(df), failures, downgraded, blank)

    if downgraded:
        zs = sorted(set(zoom_used.values()))
        # Whether a downgrade actually hurts depends on the tile FOOTPRINT, not
        # on the zoom alone: at z17 a 40m tile is ~34 real pixels (unusable, and
        # neighbouring tiles return duplicate imagery), while a 250m tile is
        # ~214 (a genuine scene). Compute it rather than assuming.
        lat = float(((df.miny + df.maxy) / 2.0).median())
        tile_m = float(((df.maxx - df.minx)
                        * 111_320.0 * math.cos(math.radians(lat))).median())
        m_per_px = 156543.03392 * math.cos(math.radians(lat)) / (2 ** min(zs))
        real_px = tile_m / max(m_per_px, 1e-9)
        msg = ("%d/%d tiles had no imagery at zoom %d and were fetched lower "
               "(zooms used: %s). At z%d a %.0fm tile is ~%d real pixels, "
               "rendered at %dpx.")
        args = (downgraded, len(df), int(zoom), zs, min(zs), tile_m,
                int(real_px), size_px)
        if real_px < 100:
            logger.warning(
                msg + " That is too coarse: adjacent tiles will return "
                "overlapping or identical imagery, so labels on them are not "
                "independent observations. Use a larger tile footprint "
                "(~%.0fm would give ~200 real pixels here).",
                *args, 200 * m_per_px)
        else:
            logger.info(msg + " Detail is reduced but the tile remains a "
                        "genuine scene.", *args)
    if blank:
        logger.warning("%d/%d tiles have NO Esri coverage at any zoom down to "
                       "%d and were EXCLUDED (returned as None), not scored.",
                       blank, len(df), MIN_ESRI_ZOOM)

    if failures:
        logger.warning("%d/%d tiles failed to download. Tiles with no imagery "
                       "are excluded from the queue, NOT scored as negative.",
                       failures, len(df))
    return paths, zoom_used


ESRI_IDENTIFY_URL = ("https://services.arcgisonline.com/arcgis/rest/services/"
                     "World_Imagery/MapServer/identify")


def esri_capture_date(lat, lon, zoom):
    """
    The capture date of the Esri World Imagery band shown at `zoom` over a
    point (feature 006, toolchain.md §4; doc 70 S48).

    World Imagery is a mosaic: each zoom band over a place comes from a
    different image, so the date is the band's whose MinMapLevel <= zoom <=
    MaxMapLevel -- the zoom the window was actually fetched at, not the one
    asked for. Returns
      {status: "provided", date: "YYYY-MM-DD", source_info}   a dated band;
      {status: "not_provided"}                                no band, or its date is Null;
      {status: "lookup_failed", error}                        the request or its answer failed.
    Never raises: a missing date is stated, never guessed, and never stops
    the check that asked for it.
    """
    import datetime as _dt

    import requests

    half = 0.001
    params = {
        "f": "json",
        "geometry": f"{lon},{lat}",
        "geometryType": "esriGeometryPoint",
        "sr": "4326",
        "layers": "all",
        "tolerance": "1",
        "returnGeometry": "false",
        "mapExtent": f"{lon - half},{lat - half},{lon + half},{lat + half}",
        "imageDisplay": "400,400,96",
    }
    try:
        # 60 s: the timeout the toolchain already uses for imagery requests
        # (fetch_tiles_gee).
        r = requests.get(ESRI_IDENTIFY_URL, params=params, timeout=60)
        r.raise_for_status()
        body = r.json()
        if "error" in body:
            return {"status": "lookup_failed", "error": str(body["error"])}
        for result in body.get("results") or []:
            a = result.get("attributes") or {}
            try:
                low, high = int(a.get("MinMapLevel")), int(a.get("MaxMapLevel"))
            except (TypeError, ValueError):
                continue
            if not low <= int(zoom) <= high:
                continue
            raw = str(a.get("DATE (YYYYMMDD)") or a.get("SRC_DATE") or "").strip()
            if not raw or raw.lower() == "null":
                return {"status": "not_provided"}
            date = _dt.datetime.strptime(raw[:8], "%Y%m%d").date()
            source = " ".join(str(a[k]).strip() for k in ("SOURCE", "DESCRIPTION")
                              if a.get(k) and str(a[k]).lower() != "null")
            resolution = a.get("RESOLUTION (M)") or a.get("SRC_RES")
            if resolution and str(resolution).lower() != "null":
                source = f"{source}, {resolution} m" if source else f"{resolution} m"
            return {"status": "provided", "date": date.isoformat(), "source_info": source}
        return {"status": "not_provided"}
    except Exception as e:  # noqa: BLE001 - never raises, by contract
        return {"status": "lookup_failed", "error": f"{type(e).__name__}: {e}"}


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def _score_tiles_probe(tile_paths, probe_path, batch_size=8):
    """
    Score with a linear probe on frozen features (see score_tiles' docstring
    for why this exists). The probe's own metadata dictates backbone, input
    size and normalization, so inference cannot silently drift from whatever
    02e_train_probe_head.py actually fitted -- a mismatch there would be
    invisible in the output, just quietly wrong probabilities.
    """
    import json
    import joblib
    import numpy as np
    import torch
    from PIL import Image
    from torchvision import transforms

    probe_path = Path(probe_path)
    meta_path = probe_path.parent / "probe_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"{meta_path} not found. arch='probe' needs the probe_meta.json "
            f"that 02e_train_probe_head.py writes next to probe.joblib.")
    meta = json.loads(meta_path.read_text())
    probe = joblib.load(probe_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = int(meta["img_size"])
    mean = tuple(meta["normalization"]["mean"])
    std = tuple(meta["normalization"]["std"])

    logger.info("Probe: backbone=%s, %d-d features, blocked-CV ROC-AUC %.3f "
                "%s, operating threshold %.3f",
                meta["backbone"], meta["feature_dim"],
                meta["roc_auc_blocked_cv"], meta["roc_auc_ci95"],
                meta["operating_threshold"])

    if meta["backbone"] == "aerialwaste":
        import importlib.util
        from plasticwaste_tool.aerialwaste_model.architecture.resnet50_fpn import Net
        from . import feature_probe as fp

        net = Net(num_classes=1, pretrained=False)
        ckpt = torch.load(meta["base_checkpoint"], map_location="cpu",
                          weights_only=False)
        net.load_state_dict(
            ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt
            else ckpt, strict=True)
        net.to(device).eval()

        def featurize(batch):
            feats, _, _ = fp.extract(net, batch)
            return feats.cpu().numpy()
    else:
        from torchvision.models import resnet50
        try:
            from torchvision.models import ResNet50_Weights
            net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        except ImportError:
            net = resnet50(pretrained=True)
        net.fc = torch.nn.Identity()
        net.to(device).eval()

        def featurize(batch):
            return net(batch).cpu().numpy()

    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])

    valid = [(i, p) for i, p in enumerate(tile_paths) if p]
    probs = [None] * len(tile_paths)

    with torch.no_grad():
        for s in range(0, len(valid), batch_size):
            chunk = valid[s:s + batch_size]
            batch = torch.stack([tf(Image.open(p).convert("RGB"))
                                 for _, p in chunk]).to(device)
            feats = featurize(batch)
            p = probe.predict_proba(feats)[:, 1]
            for (idx, _), v in zip(chunk, p):
                probs[idx] = float(v)
            if (s // batch_size) % 20 == 0:
                logger.info("  scored %d/%d", min(s + batch_size, len(valid)),
                            len(valid))

    scored = [p for p in probs if p is not None]
    if scored:
        logger.info("Probe scores: min %.4f, median %.4f, max %.4f "
                    "(threshold %.3f -> %d/%d above)",
                    float(np.min(scored)), float(np.median(scored)),
                    float(np.max(scored)), meta["operating_threshold"],
                    int(sum(p >= meta["operating_threshold"] for p in scored)),
                    len(scored))
    return probs


def score_tiles(tile_paths, checkpoint_path, scripts_dir, img_size=224,
                batch_size=32, arch="ins_resnet50fpn"):
    """
    Return P(dumpsite) for each tile. None for tiles with no imagery.

    arch="ins_resnet50fpn" (default, unchanged): the from-scratch model
    trained by 02_train_baseline.py on the INS drone dataset.

    arch="aerialwaste": the SRS-specified model -- ResNet50+FPN pretrained on
    AerialWaste (Torres & Fraternali, Scientific Reports 2023), optionally
    fine-tuned by 02b_finetune_aerialwaste.py. Accepts either that base
    checkpoint (a plain state_dict) or a fine-tuned one (wrapped in
    {"model_state": ...}, this project's usual convention) -- detected
    automatically. Defaults to 800px input (img_size is ignored for this arch
    unless explicitly overridden), matching the architecture's own documented
    training image size.

    arch="probe": a linear probe on FROZEN pretrained features, trained by
    02e_train_probe_head.py; `checkpoint_path` is that script's probe.joblib
    (its probe_meta.json must sit beside it, and names which backbone to use).
    This is the arch to use in practice, because the "aerialwaste" path above
    scored ROC-AUC 0.287 on this project's Guinea labels -- BELOW coin flip,
    i.e. it ranks real dumpsites below non-dumpsites. Cause (measured, see
    02c/02d): with num_classes=1 that head funnels the whole image through 4
    ReLU-gated scalars, and on 73.4% of Guinea tiles all four gates close, so
    the output is exactly classifier.bias regardless of content. Keeping the
    same frozen backbone but replacing the collapsed head lifts blocked-CV
    ROC-AUC to 0.834 [95% CI 0.681-0.920].
    """
    import importlib.util
    import torch
    from PIL import Image
    from torchvision import transforms

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if arch == "probe":
        return _score_tiles_probe(tile_paths, checkpoint_path, batch_size)

    if arch == "aerialwaste":
        from plasticwaste_tool.aerialwaste_model.architecture.resnet50_fpn import Net
        net = Net(num_classes=1, pretrained=False).to(device)
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
        net.load_state_dict(state_dict)
        net.eval()
        logger.info("Loaded AerialWaste-arch checkpoint from %s (epoch %s)",
                    checkpoint_path, ckpt.get("epoch", "n/a (base checkpoint)")
                    if isinstance(ckpt, dict) else "n/a (base checkpoint)")
        mean, std = IMAGENET_MEAN, IMAGENET_STD
        img_size = img_size if img_size != 224 else 800  # 224 is the other arch's default, not a real override here
    else:
        spec = importlib.util.spec_from_file_location(
            "train_mod", Path(scripts_dir) / "02_train_baseline.py")
        tm = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tm)
        net = tm.ResNet50FPNClassifier(pretrained=False).to(device)
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        net.load_state_dict(ckpt["model_state"])
        net.eval()
        logger.info("Loaded checkpoint from epoch %s", ckpt.get("epoch"))
        mean, std = tm.IMAGENET_MEAN, tm.IMAGENET_STD

    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])

    valid = [(i, p) for i, p in enumerate(tile_paths) if p]
    probs = [None] * len(tile_paths)

    with torch.no_grad():
        for s in range(0, len(valid), batch_size):
            chunk = valid[s:s + batch_size]
            batch = torch.stack([tf(Image.open(p).convert("RGB"))
                                 for _, p in chunk]).to(device)
            out = torch.sigmoid(net(batch)).cpu().numpy()
            for (idx, _), pr in zip(chunk, out):
                probs[idx] = float(pr)
            if s % (batch_size * 20) == 0:
                logger.info("  scored %d/%d", s, len(valid))
    return probs


# ---------------------------------------------------------------------------
# The measurable metric
# ---------------------------------------------------------------------------
def precision_at_n(verified_labels, n=None):
    """
    Precision@N — the one accuracy figure you can compute BEFORE having a
    fully labelled dataset.

    `verified_labels` is a list of 1/0 for the top-N candidates after a human
    has checked them, in rank order.

    Why this matters: precision needs only the tiles you actually looked at.
    Recall needs to know about dumpsites you did NOT flag, which requires
    exhaustively labelling a region. So precision@N is available after one
    afternoon of photo-interpretation; recall is a Phase-2 deliverable.

    Report it as: "Of the top 200 ranked candidates, 47% contained a visible
    dumpsite on inspection (n=200)." That is a real, defensible,
    field-relevant number — and it is exactly what tells the UNDP CO how much
    survey effort the tool saves.
    """
    y = np.asarray(verified_labels).astype(int)
    if n is not None:
        y = y[:n]
    if len(y) == 0:
        return {"n": 0, "precision_at_n": float("nan")}

    out = {"n": int(len(y)), "n_true_positive": int(y.sum()),
           "precision_at_n": float(y.mean())}
    for k in (10, 50, 100, 200, 500):
        if len(y) >= k:
            out[f"precision_at_{k}"] = float(y[:k].mean())
    # Does the ranking actually rank? Compare the top and bottom halves.
    if len(y) >= 20:
        half = len(y) // 2
        out["precision_top_half"] = float(y[:half].mean())
        out["precision_bottom_half"] = float(y[half:].mean())
        out["ranking_works"] = bool(
            out["precision_top_half"] > out["precision_bottom_half"])
    return out


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
def export_queue(tiles_df, probs, out_dir, top_n=2000, strata=(0.6, 0.25, 0.15)):
    """
    Write the annotation queue as GeoJSON + CSV.

    Stratified rather than pure top-N. The low-confidence stratum feels like a
    waste of annotator time and is not: without it you only ever measure
    precision on tiles the model already likes, you never discover what it
    misses, and your training set inherits the model's own blind spots. It is
    also what breaks the circularity when the susceptibility prior is later
    validated against these labels.
    """
    import geopandas as gpd
    from shapely.geometry import box

    df = tiles_df.copy()
    df["probability"] = probs
    df = df[df.probability.notna()].copy()
    df = df.sort_values("probability", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)

    # Scale strata to the ACTUAL scored pool, not the requested top_n: with
    # top_n's default of 2000 and strata=(0.6,.25,.15), n_high alone is 1200 --
    # if fewer tiles than that were actually scored (e.g. a --max-tiles test
    # run), `df.head(n_high)` silently swallows the entire pool, leaving `rest`
    # empty and mid/low at 0 with no error. Cap against len(df) and warn.
    effective_n = min(top_n, len(df))
    if len(df) < top_n:
        logger.warning("Only %d tiles scored, fewer than --top-n %d -- "
                       "stratifying against the smaller pool instead of "
                       "silently dumping everything into the 'high' stratum.",
                       len(df), top_n)
    n_high, n_mid, n_low = (int(effective_n * s) for s in strata)
    high = df.head(n_high)
    rest = df.iloc[n_high:]
    mid = rest[rest.probability >= 0.2].sample(
        min(n_mid, max((rest.probability >= 0.2).sum(), 0)), random_state=42) \
        if len(rest) else rest
    low = rest[rest.probability < 0.2].sample(
        min(n_low, max((rest.probability < 0.2).sum(), 0)), random_state=42) \
        if len(rest) else rest

    high = high.assign(stratum="high")
    mid = mid.assign(stratum="mid")
    low = low.assign(stratum="low")
    queue = pd.concat([high, mid, low], ignore_index=True)

    # Blank columns the annotator fills in. `verified` stays empty until a
    # human has looked; an unfilled row must never be treated as a negative.
    queue["verified"] = ""
    queue["annotator"] = ""
    queue["notes"] = ""

    gdf = gpd.GeoDataFrame(
        queue,
        geometry=[box(r.minx, r.miny, r.maxx, r.maxy)
                  for r in queue.itertuples()],
        crs="EPSG:4326")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out_dir / "annotation_queue.geojson", driver="GeoJSON")
    queue.drop(columns=[c for c in ("geometry",) if c in queue]).to_csv(
        out_dir / "annotation_queue.csv", index=False)

    summary = {
        "tiles_scored": int(len(df)),
        "queue_size": int(len(queue)),
        "strata": {"high": int(len(high)), "mid": int(len(mid)),
                   "low": int(len(low))},
        "prob_range_high": [float(high.probability.min()),
                            float(high.probability.max())] if len(high) else None,
        "note": ("Probabilities come from a model trained on non-Guinea data. "
                 "They are a RANKING, not a calibrated probability, and no "
                 "accuracy figure for Guinea can be reported until these "
                 "tiles are verified."),
    }
    (out_dir / "queue_summary.json").write_text(json.dumps(summary, indent=2))

    logger.info("Queue written: %d tiles (high %d / mid %d / low %d)",
                len(queue), len(high), len(mid), len(low))
    return gdf, summary


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Google Maps Static API fetcher
# ---------------------------------------------------------------------------
# Esri is the default source for model input: it is free, needs no key, and a
# controlled ablation showed its z17 imagery performs identically to z19
# (dAUC -0.011, P=0.70), so resolution is not a reason to pay.
#
# What Esri CANNOT do is serve a small footprint with enough pixels in it.
# Outside Conakry it caps at z17 (~1.17 m/px), so an 80m window is only ~68
# real pixels upsampled to the model's 800px input. Footprint turned out to
# matter a great deal -- 250m cells produced a settlement detector, 80m
# point-centred crops produced a waste detector -- so whether more pixels
# inside that small window helps is a live question, and this is how to test it.
#
# Google licenses the same Maxar/Vantor imagery at deeper zoom levels. The Maps
# Static API is the licensed product for this (unlike the mt*.google.com tile
# endpoint, which is internal and unlicensed regardless of any API key).
# Static Maps is an "Essentials" SKU: 10,000 calls free per month, then ~$2/1000.
#
# NOTE ON USE: fetching imagery to extract features is a different question from
# retaining an imagery corpus. This pipeline discards pixels after feature
# extraction. Before relying on Google imagery for a published deliverable, have
# the current Maps Platform terms checked by whoever handles licensing.

GOOGLE_STATIC_URL = "https://maps.googleapis.com/maps/api/staticmap"
GOOGLE_MAX_SIZE = 640          # API limit for a single request (scale doubles it)


def _google_api_key(explicit=None):
    """Key from argument, environment, or the labelling tool's local file.

    The file is read as BYTES and decoded defensively. A key pasted on Windows
    is routinely saved as UTF-16 (PowerShell's `>` redirection does this), which
    a plain utf-8 read rejects with "can't decode byte 0xff" -- that byte is the
    byte-order mark, not a corrupt key. Surrounding quotes and stray nulls are
    stripped too, since pasting `"AIza..."` is an easy and silent mistake.
    """
    if explicit:
        return explicit.strip()
    k = os.environ.get("GOOGLE_MAPS_API_KEY", "").strip()
    if k:
        return k
    # This module is copied into dumpsite_tool/, where __file__.parent is the
    # PACKAGE, not the tool root. Anchor on the tool root so both copies
    # find the one key file instead of reporting it missing.
    here = Path(__file__).resolve().parent
    root = here.parent if here.name == "dumpsite_tool" else here
    f = root / "manual_qa" / "regions" / "._google_api_key.txt"
    if not f.exists():
        return ""
    raw = f.read_bytes()
    text = None
    for enc in ("utf-8-sig", "utf-16", "utf-16-le", "utf-16-be", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    if text is None:
        logger.error("could not decode %s in any common encoding", f.name)
        return ""
    return text.replace(chr(0), "").strip().strip('"').strip("'").strip()


def fetch_tiles_google(tiles_df, out_dir, size_px=800, zoom=19, api_key=None,
                       max_tiles=None, budget=None):
    """Fetch each tile's footprint from the Maps Static API.

    Unlike a tile service, Static Maps returns one image centred on a point, so
    the request is sized to cover the tile's own width: pixels_needed =
    width_m / (m per pixel at this zoom and latitude). That keeps the GROUND
    FOOTPRINT equal to the grid cell, which is what makes results comparable
    with the Esri path rather than merely similar.

    `budget` caps the number of billable requests actually issued, so a run
    cannot silently consume a month's free quota; cached tiles never count
    against it. Resume-aware: an existing file is reused, as in fetch_tiles_esri.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # The key is resolved now but NOT required yet. A run with budget=0, or one
    # whose tiles are all cached, issues no request and must not fail for want
    # of a credential it will never use -- that is the cache-only dry run the
    # deployment path depends on. The requirement is enforced below, at the
    # point a billable request is actually about to be made.
    key = _google_api_key(api_key)

    # PIL is imported locally throughout this module rather than at module
    # level; follow that, or this function raises NameError *after* the HTTP
    # request has already been made and billed.
    import requests
    from PIL import Image

    df = tiles_df if max_tiles is None else tiles_df.head(max_tiles)
    paths, spent, cached, failed = [], 0, 0, 0
    capped = False

    for i, row in enumerate(df.itertuples(), 1):
        dest = out_dir / f"{row.tile_id}.png"
        if dest.exists():
            cached += 1
            paths.append(str(dest))
            continue
        if budget is not None and spent >= budget:
            if not capped:
                # Once, not once per tile: a national scan would otherwise emit
                # one identical line per remaining tile and bury everything else.
                logger.warning("request budget of %d reached; the remaining %d "
                               "tiles will NOT be fetched or scored",
                               budget, len(df) - i + 1)
                capped = True
            paths.append(None)
            continue
        if not key:
            raise RuntimeError(
                "No Google Maps API key, and tile %s is not cached so this run "
                "needs one. Set GOOGLE_MAPS_API_KEY, pass api_key=, or write it "
                "to manual_qa/regions/._google_api_key.txt. (Pass budget=0 to "
                "run on cached tiles only, which needs no key.)" % row.tile_id)

        lat = (row.miny + row.maxy) / 2.0
        lon = (row.minx + row.maxx) / 2.0
        width_m = (row.maxx - row.minx) * 111_320.0 * math.cos(math.radians(lat))
        m_per_px = 156543.03392 * math.cos(math.radians(lat)) / (2 ** zoom)
        need = int(round(width_m / max(m_per_px, 1e-9)))
        req = max(16, min(need, GOOGLE_MAX_SIZE))
        if need > GOOGLE_MAX_SIZE:
            logger.warning("tile %s needs %dpx at z%d but the API caps at %d; "
                           "the returned image covers more ground than the cell",
                           row.tile_id, need, zoom, GOOGLE_MAX_SIZE)

        try:
            r = requests.get(GOOGLE_STATIC_URL, timeout=30, params={
                "center": f"{lat:.6f},{lon:.6f}", "zoom": zoom,
                "size": f"{req}x{req}", "scale": 2, "maptype": "satellite",
                "format": "png", "key": key})
            spent += 1
            if r.status_code != 200 or "image" not in r.headers.get("Content-Type", ""):
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:160]}")
            img = Image.open(_io.BytesIO(r.content)).convert("RGB")
            if _is_placeholder(img):
                logger.warning("tile %s: blank imagery at z%d", row.tile_id, zoom)
                paths.append(None)
                continue
            img.resize((size_px, size_px), Image.LANCZOS).save(dest)
            paths.append(str(dest))
        except Exception as e:
            failed += 1
            logger.warning("tile %s failed: %s", row.tile_id, str(e)[:140])
            paths.append(None)

        if i % 50 == 0:
            logger.info("  %d/%d  (%d billable requests spent, %d cached)",
                        i, len(df), spent, cached)

    logger.info("Google Static Maps: %d billable requests, %d served from cache, "
                "%d failed", spent, cached, failed)
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aoi", default="conakry")
    ap.add_argument("--bounds", nargs=4, type=float, default=None,
                    metavar=("MINX", "MINY", "MAXX", "MAXY"))
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--arch", default="ins_resnet50fpn",
                    choices=["ins_resnet50fpn", "aerialwaste", "probe"],
                    help="ins_resnet50fpn: 02_train_baseline.py's from-scratch "
                         "model (default, unchanged). aerialwaste: the "
                         "pretrained/fine-tuned AerialWaste model as shipped "
                         "(see 02b) -- NOTE this scored ROC-AUC 0.287 on Guinea "
                         "labels, below coin flip, due to a collapsed head. "
                         "probe: linear probe on the same frozen backbone "
                         "(02e_train_probe_head.py), blocked-CV ROC-AUC 0.834 "
                         "[0.681-0.920] -- pass its probe.joblib as --checkpoint.")
    ap.add_argument("--scripts-dir", default=".")
    ap.add_argument("--source", default="nicfi", choices=["nicfi", "s2", "esri"])
    ap.add_argument("--tile-m", type=float, default=300.0)
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--month", type=int, default=1)
    ap.add_argument("--esri-zoom", type=int, default=19,
                    help="tile zoom for --source esri (~0.3m/px at zoom 19 "
                         "near the equator); only used with --source esri")
    ap.add_argument("--max-tiles", type=int, default=None,
                    help="cap for a first test run")
    ap.add_argument("--top-n", type=int, default=2000)
    ap.add_argument("--out", default="queue")
    args = ap.parse_args()

    if args.bounds:
        bounds = tuple(args.bounds)
    else:
        from plasticwaste_tool import api_connector as api
        api.initialize_gee()
        aoi = api.get_aoi_geometry(args.aoi)
        c = aoi.bounds().coordinates().getInfo()[0]
        xs = [p[0] for p in c]
        ys = [p[1] for p in c]
        bounds = (min(xs), min(ys), max(xs), max(ys))

    tiles = make_tile_grid(bounds, args.tile_m)
    estimate_cost(len(tiles), args.source)

    out_dir = Path(args.out)
    if args.source == "esri":
        paths, _ = fetch_tiles_esri(tiles, out_dir / "tiles",
                                    max_tiles=args.max_tiles, zoom=args.esri_zoom)
    else:
        paths = fetch_tiles_gee(tiles, out_dir / "tiles", args.source,
                                args.year, args.month, max_tiles=args.max_tiles)
    if args.max_tiles:
        tiles = tiles.head(args.max_tiles)

    probs = score_tiles(paths, args.checkpoint, args.scripts_dir, arch=args.arch)
    gdf, summary = export_queue(tiles, probs, out_dir, args.top_n)

    print("\n" + "=" * 62)
    print("ANNOTATION QUEUE READY")
    print("=" * 62)
    print(json.dumps(summary, indent=2))
    print(f"\nOpen {out_dir/'annotation_queue.geojson'} in QGIS over a "
          f"high-resolution basemap.")
    print("Fill the 'verified' column with 1 / 0. Then:")
    print("  ev.precision_at_n(verified_labels)   # your first real number")


if __name__ == "__main__":
    main()
