"""
Deforestation & degradation detection.

Primary signals (ready-made, recommended):
  * annual_loss_from_hansen  -> per-year forest LOSS masks (authoritative)
  * deforestation_from_lulc  -> forest(t1) & not forest(t2), between LULC epochs
  * conversion_from_lulc     -> WHAT the forest converted to (cropland/built/...)

Secondary:
  * degradation_from_vi      -> NDVI decline within forest (thinning, not clearing)

Unsupervised option (as requested):
  * kmeans_landcover         -> cluster the annual stack into K classes
  * clusters_to_forest       -> label clusters forest/non-forest by mean NDVI
  These are noisier and require label interpretation; Hansen/LULC are preferred.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Primary: Hansen annual loss
# ---------------------------------------------------------------------------
def annual_loss_from_hansen(lossyear_array, years=None, base_year=2000):
    """{year: boolean loss mask} for each requested year (default: all)."""
    arr = np.asarray(lossyear_array)
    codes = np.unique(arr[arr > 0])
    if years is None:
        years = [int(base_year + c) for c in codes]
    return {y: (arr == (y - base_year)) for y in sorted(years)}


def cumulative_loss(lossyear_array, up_to_year, base_year=2000):
    arr = np.asarray(lossyear_array)
    return (arr > 0) & (arr <= (up_to_year - base_year))


def vectorize_loss(loss_mask, profile, year, min_area_ha=1.0,
                   zones_gdf=None, name_field="shapeName"):
    """
    One polygon per contiguous cleared patch in a boolean forest-loss mask,
    with its area in hectares and, when `zones_gdf` is given, the
    administrative zone its centroid falls in.

    This is the vector form of the mask `stats.zonal_deforestation` measures:
    the zonal table answers "how much loss, per zone", these polygons answer
    "which clearings" -- a detection feed and a per-feature popup both need the
    second and cannot be built from the first.

    `year` is the Hansen loss year and is recorded as-is. Hansen dates loss to
    the year, not the month, so there is no finer date available and none is
    invented here.

    `min_area_ha` drops patches below the minimum mappable unit. The default of
    1 ha matches the lower bound of the interface's area filter and keeps a
    national run to a sane row count; at Hansen's 30 m grid a single pixel is
    0.09 ha, which is speckle, not a clearing.

    `zones_gdf` is a GeoDataFrame, or a {column_name: GeoDataFrame} mapping to
    attribute several administrative levels in one pass (e.g.
    {"adm1_name": adm1, "adm2_name": adm2}).

    Returns a GeoDataFrame in EPSG:4326 (the CRS the API stores geometry in),
    with columns: geometry, area_ha, detection_year, centroid_lat, centroid_lon
    and one column per attributed zone level.
    """
    import geopandas as gpd
    from rasterio.features import shapes as rasterio_shapes
    from scipy.ndimage import label as cc_label
    from shapely.geometry import shape as shapely_shape

    from . import stats as S

    mask = np.asarray(loss_mask, dtype=bool)
    crs = profile.get("crs")
    columns = ["geometry", "area_ha", "detection_year", "centroid_lat", "centroid_lon"]

    zone_layers = zones_gdf if isinstance(zones_gdf, dict) else (
        {} if zones_gdf is None else {"zone_name": zones_gdf})

    def _empty():
        # Typed empty frame: a year with no loss, or one whose every patch falls
        # under the minimum, must still return a usable GeoDataFrame. Building
        # one from an empty list raises on the missing geometry column.
        return gpd.GeoDataFrame(columns=columns + list(zone_layers),
                                geometry="geometry", crs="EPSG:4326")

    if not mask.any():
        return _empty()

    labeled, n_features = cc_label(mask, structure=np.ones((3, 3)))  # 8-connected
    if n_features == 0:
        return _empty()

    # Per-pixel area varies with latitude on a geographic grid, so patch area is
    # summed from the area array rather than taken as a pixel count times a
    # constant. bincount does every patch in one pass over the raster.
    area = S.pixel_area_ha(profile, shape=mask.shape)
    flat = labeled.ravel()
    if np.isscalar(area):
        area_by_label = np.bincount(flat, minlength=n_features + 1) * float(area)
    else:
        area_by_label = np.bincount(
            flat, weights=np.asarray(area, dtype="float64").ravel(),
            minlength=n_features + 1)

    records = []
    for geom, value in rasterio_shapes(labeled.astype("int32"), mask=mask,
                                       transform=profile["transform"], connectivity=8):
        label_id = int(value)
        if label_id == 0:
            continue
        area_ha = float(area_by_label[label_id])
        if min_area_ha and area_ha < min_area_ha:
            continue
        records.append({"geometry": shapely_shape(geom),
                        "area_ha": round(area_ha, 3),
                        "detection_year": int(year)})

    if not records:
        return _empty()

    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=crs)
    # Centroids are taken on the analysis grid's own CRS and then reprojected.
    # Taking them after reprojecting to EPSG:4326 computes them in degrees,
    # which is skewed and which geopandas warns about.
    centroids = gpd.GeoSeries(gdf.geometry.centroid, crs=crs).to_crs("EPSG:4326")
    gdf = gdf.to_crs("EPSG:4326")
    gdf["centroid_lat"] = centroids.y
    gdf["centroid_lon"] = centroids.x

    if zone_layers:
        # Joined on the centroid, not the polygon: a clearing straddling a
        # boundary belongs to one zone in a feed, and a polygon join would
        # duplicate the row once per zone it touches.
        points = gpd.GeoDataFrame(gdf[["area_ha"]].copy(),
                                  geometry=centroids, crs="EPSG:4326")
        for column, spec in zone_layers.items():
            # A value is a layer, or a (layer, field) pair so one layer can
            # fill both an ID and a name column.
            layer, field = spec if isinstance(spec, tuple) else (spec, name_field)
            zones = layer.to_crs("EPSG:4326")[[field, "geometry"]].rename(
                columns={field: column})
            joined = gpd.sjoin(points, zones, how="left", predicate="within")
            # sjoin can emit more than one row per point where zone polygons
            # overlap; keep the first so the result stays one row per patch.
            joined = joined[~joined.index.duplicated(keep="first")]
            gdf[column] = joined[column].reindex(gdf.index).values

    logger.info("Vectorized %d loss patches >= %s ha for %s.",
                len(gdf), min_area_ha, year)
    return gdf.sort_values("area_ha", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# LULC-transition-based deforestation (uses the Land Use multitemporal LULC)
# ---------------------------------------------------------------------------
def deforestation_from_lulc(forest_t1, forest_t2):
    """Deforestation = forest at t1 AND not forest at t2 (boolean)."""
    return np.asarray(forest_t1, dtype=bool) & ~np.asarray(forest_t2, dtype=bool)


def conversion_classes(deforested_mask, lulc_t2):
    """
    For pixels flagged as deforested, return the LULC class they became at t2
    (0 elsewhere). This "converted-to" map drives the forest -> {cropland, built,
    grassland, ...} transition statistics in stats.py.
    """
    out = np.zeros_like(np.asarray(lulc_t2))
    m = np.asarray(deforested_mask, dtype=bool)
    out[m] = np.asarray(lulc_t2)[m]
    return out


# ---------------------------------------------------------------------------
# Degradation from vegetation index
# ---------------------------------------------------------------------------
def degradation_from_vi(vi_t1, vi_t2, forest_mask, drop=0.10):
    """
    Degradation = significant NDVI decline WITHIN pixels that stayed forest.
    (Distinguished from deforestation, which is outright forest loss.)
    """
    vi_t1 = np.asarray(vi_t1, dtype="float32")
    vi_t2 = np.asarray(vi_t2, dtype="float32")
    change = vi_t2 - vi_t1
    return np.asarray(forest_mask, dtype=bool) & (change <= -abs(drop))


# ---------------------------------------------------------------------------
# Unsupervised K-means option
# ---------------------------------------------------------------------------
def kmeans_landcover(stack_arrays, n_clusters=3, mask=None, seed=1, sample=50000):
    """
    Cluster a stack of aligned feature rasters (e.g. [NDVI, NIR, RED] or the
    predictor stack) into n_clusters. Returns an int label raster (-1 outside
    mask). Trained on a random sample for speed.

    stack_arrays: list of 2D arrays, all the same shape.
    """
    from sklearn.cluster import KMeans
    arrs = [np.asarray(a, dtype="float32") for a in stack_arrays]
    h, w = arrs[0].shape
    X = np.stack([a.ravel() for a in arrs], axis=1)
    valid = np.all(np.isfinite(X), axis=1)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool).ravel()

    labels = np.full(h * w, -1, dtype=np.int16)
    idx = np.nonzero(valid)[0]
    if idx.size < n_clusters:
        logger.warning("K-means: too few valid pixels (%d)", idx.size)
        return labels.reshape(h, w)

    rng = np.random.default_rng(seed)
    train_idx = idx if idx.size <= sample else rng.choice(idx, sample, replace=False)
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    km.fit(X[train_idx])
    labels[idx] = km.predict(X[idx]).astype(np.int16)
    return labels.reshape(h, w)


def clusters_to_forest(labels, ndvi, forest_quantile=0.5):
    """
    Label clusters forest/non-forest by mean NDVI: clusters whose mean NDVI is in
    the upper part of the range are called forest. Returns a boolean forest mask.
    """
    labels = np.asarray(labels)
    ndvi = np.asarray(ndvi, dtype="float32")
    forest = np.zeros(labels.shape, dtype=bool)
    means = {}
    for c in np.unique(labels[labels >= 0]):
        means[c] = np.nanmean(ndvi[labels == c])
    if not means:
        return forest
    cutoff = np.quantile(list(means.values()), forest_quantile)
    for c, mv in means.items():
        if mv >= cutoff:
            forest[labels == c] = True
    return forest
