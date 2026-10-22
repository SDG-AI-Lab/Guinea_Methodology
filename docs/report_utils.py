"""Data loading, statistics and map helpers for the forest-loss report (forest_loss_assessment.qmd).

Everything is read from tables already saved by the notebooks in ../studies; nothing here calls Earth Engine.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import folium
import geopandas as gpd
import h3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from folium.plugins import Fullscreen
from h3.api import basic_int as h3_int

DATA_DIR = Path(__file__).resolve().parent.parent / "studies" / "output_datasets"
BENCHMARK_DIR = DATA_DIR / "lulc_benchmark"

CANOPY_THRESHOLDS = [0, 10, 30, 50]
CANOPY_THRESHOLD = 30
FIRST_YEAR, LAST_YEAR = 2001, 2025
METHOD_CHANGE_YEAR = 2013
RECENT_YEAR = 2020
FINE_H3_RESOLUTION = 8
HA_PER_KM2, M2_PER_HA = 100, 10_000

PLOT_TEMPLATE = "plotly_white"
THRESHOLD_COLORS = {0: "#c7e9c0", 10: "#74c476", 30: "#238b45", 50: "#00441b"}
PERIOD_COLORS = {f"{FIRST_YEAR}–{METHOD_CHANGE_YEAR - 1}": "#b0b0b0",
                 f"{METHOD_CHANGE_YEAR}–{RECENT_YEAR - 1}": "#e67e22",
                 f"{RECENT_YEAR}–{LAST_YEAR}": "#c0392b"}
ECOREGION_SHORT = {
    "Guinean forest-savanna": "Forest-savanna", "Guinean mangroves": "Mangroves",
    "Guinean montane forests": "Montane forests", "West Sudanian savanna": "Sudanian savanna",
    "Western Guinean lowland forests": "Lowland forests",
}


def fmt(value, digits=0):
    """Number with thousands separators, for the text."""
    return f"{value:,.{digits}f}"


# --------------------------------------------------------------------------------------------- data

@lru_cache
def boundaries():
    guinea = gpd.read_file(DATA_DIR / "ecoregions_guinea.gpkg", layer="guinea_boundary")
    ecoregions = gpd.read_file(DATA_DIR / "ecoregions_guinea.gpkg", layer="ecoregions_guinea")
    return guinea, ecoregions


def ecoregion_colors():
    names = sorted(boundaries()[1]["ECO_NAME"])
    return dict(zip(names, px.colors.qualitative.Safe))


@lru_cache
def hansen_table():
    """Wall-to-wall Hansen v1.13: every pixel of Guinea by ecoregion polygon x 1° cell x loss year, in hectares."""
    table = pd.read_parquet(DATA_DIR / "hansen_loss_by_polygon_cell_year.parquet")
    for column in [c for c in table.columns if c.endswith("_km2")]:
        table[column.replace("_km2", "_ha")] = table.pop(column) * HA_PER_KM2
    return table


def hansen_long():
    table = hansen_table()
    long = table.melt(id_vars=["ECO_NAME", "lossyear", "year"],
                      value_vars=[f"forest2000_gt{t}_ha" for t in CANOPY_THRESHOLDS],
                      var_name="threshold", value_name="area_ha")
    long["threshold"] = long["threshold"].str.extract(r"gt(\d+)_ha", expand=False).astype(int)
    return long


@lru_cache
def fine_hexagons():
    """Forest and loss per H3 resolution-8 hexagon (made by studies/report_1.ipynb, section 0.3)."""
    path = DATA_DIR / f"hansen_forest_loss_h3res{FINE_H3_RESOLUTION}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path.name} is missing: run section 0.3 of studies/report_1.ipynb once.")
    fine = pd.read_parquet(path)
    fine["h3"] = [h3.str_to_int(cell) for cell in fine["h3_id"]]
    fine["loss_2013_2025_ha"] = fine["loss_2013_2019_ha"] + fine["loss_2020_2025_ha"]
    fine["forest_2012_ha"] = fine["forest2000_gt30_ha"] - fine["loss_2001_2012_ha"]
    fine["forest_2025_ha"] = fine["forest_2012_ha"] - fine["loss_2013_2025_ha"]
    return fine


def national_loss_by_year(table=None):
    table = hansen_table() if table is None else table
    return (table[table["lossyear"] > 0].groupby("year")["forest2000_gt30_ha"].sum()
            .reindex(range(FIRST_YEAR, LAST_YEAR + 1), fill_value=0))


def ecoregion_forest():
    table = hansen_table()
    forest = table.groupby("ECO_NAME")["forest2000_gt30_ha"]
    by_period = lambda first, last: table[table["year"].between(first, last)].groupby("ECO_NAME")["forest2000_gt30_ha"].sum()
    eco = pd.DataFrame({
        "land_ha": table.groupby("ECO_NAME")["land_ha"].sum(),
        "forest_2000_ha": forest.sum(),
        "loss_2001_2012_ha": by_period(FIRST_YEAR, METHOD_CHANGE_YEAR - 1),
        "loss_2013_2019_ha": by_period(METHOD_CHANGE_YEAR, RECENT_YEAR - 1),
        "loss_2020_2025_ha": by_period(RECENT_YEAR, LAST_YEAR),
    })
    eco["forest_2012_ha"] = eco["forest_2000_ha"] - eco["loss_2001_2012_ha"]
    eco["forest_2019_ha"] = eco["forest_2012_ha"] - eco["loss_2013_2019_ha"]
    eco["forest_2025_ha"] = eco["forest_2019_ha"] - eco["loss_2020_2025_ha"]
    return eco


def loss_rates():
    """Annual loss (ha) and loss rate (% of the forest at the start of the year), per ecoregion and for Guinea."""
    table = hansen_table()
    loss = (table[table["lossyear"] > 0].groupby(["ECO_NAME", "year"])["forest2000_gt30_ha"].sum()
            .unstack(fill_value=0).reindex(columns=range(FIRST_YEAR, LAST_YEAR + 1), fill_value=0))
    loss.loc["Guinea"] = loss.sum()
    forest_2000 = table.groupby("ECO_NAME")["forest2000_gt30_ha"].sum()
    forest_2000["Guinea"] = forest_2000.sum()
    forest_at_start = loss.cumsum(axis=1).shift(1, axis=1, fill_value=0).rsub(forest_2000, axis=0)
    return loss, 100 * loss / forest_at_start


# ------------------------------------------------------------------------------- concentration, hotspots

def concentration_curve(table, loss_column, basis_column="land_ha"):
    """Share of the loss (y) held by a share of the land or forest (x), hexagons sorted from most to least loss."""
    ordered = table.sort_values(loss_column, ascending=False)
    return pd.DataFrame({
        "share_of_basis_%": 100 * ordered[basis_column].cumsum() / ordered[basis_column].sum(),
        "share_of_loss_%": 100 * ordered[loss_column].cumsum() / ordered[loss_column].sum(),
    })


def share_at(curve, basis_share):
    return float(np.interp(basis_share, curve["share_of_basis_%"], curve["share_of_loss_%"]))


HOTSPOT_CLASSES = ["No loss", "Diffuse loss (last 20 %)", "Hotspot (50–80 % of loss)", "Core hotspot (first 50 % of loss)"]
HOTSPOT_COLORS = ["#f0f0f0", "#fdd49e", "#ef6548", "#7f0000"]


def hotspot_classes(loss):
    """0 = no loss, 1 = the hexagons holding the last 20 % of the loss, 2 = 50–80 %, 3 = the first 50 %."""
    ordered = loss.sort_values(ascending=False)
    cumulative = 100 * ordered.cumsum() / ordered.sum()
    classes = np.select([cumulative <= 50, cumulative <= 80, ordered > 0], [3, 2, 1], 0)
    return pd.Series(classes, index=ordered.index).reindex(loss.index)


# ------------------------------------------------------------------------------------------ static maps

def map_extent():
    guinea, _ = boundaries()
    west, south, east, north = guinea.total_bounds
    return west - 0.1, south - 0.1, east + 0.1, north + 0.1


def map_frame(ax, title=None, extent=None):
    guinea, ecoregions = boundaries()
    ecoregions.boundary.plot(ax=ax, color="#555555", linewidth=0.4)
    guinea.boundary.plot(ax=ax, color="black", linewidth=0.9)
    west, south, east, north = extent or map_extent()
    ax.set_xlim(west, east)
    ax.set_ylim(south, north)
    ax.set_aspect(1 / np.cos(np.radians((south + north) / 2)))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    for side in ax.spines.values():
        side.set_visible(False)
    if title:
        ax.set_title(title, fontsize=11, loc="left")


DISPLAY_DEG = 0.0025  # display grid for hexagon images (~275 m)


@lru_cache
def display_grid():
    """Row of the hexagon table under every point of a regular lon/lat grid over Guinea (-1 = none)."""
    west, south, east, north = map_extent()
    lons = np.arange(west + DISPLAY_DEG / 2, east, DISPLAY_DEG)
    lats = np.arange(north - DISPLAY_DEG / 2, south, -DISPLAY_DEG)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    cells = np.array([h3_int.latlng_to_cell(lat, lon, FINE_H3_RESOLUTION)
                      for lat, lon in zip(lat_grid.ravel().tolist(), lon_grid.ravel().tolist())], dtype="int64")
    position = pd.Index(fine_hexagons()["h3"]).get_indexer(cells).reshape(lat_grid.shape)
    bounds = [[north - len(lats) * DISPLAY_DEG, west], [north, west + len(lons) * DISPLAY_DEG]]  # [[S, W], [N, E]]
    return position, bounds


def hexagon_image(values):
    """2-D array of one value per row of fine_hexagons() (NaN where there is no hexagon)."""
    position, _ = display_grid()
    values = np.asarray(values, dtype=float)
    return np.where(position >= 0, values[np.maximum(position, 0)], np.nan)


def hexagon_map(ax, values, cmap, vmin, vmax, title, label):
    position, ((south, west), (north, east)) = display_grid()
    cmap = plt.get_cmap(cmap).copy()
    cmap.set_bad("white")
    shown = ax.imshow(hexagon_image(values), extent=[west, east, south, north], cmap=cmap, vmin=vmin, vmax=vmax,
                      interpolation="nearest")
    map_frame(ax, title)
    plt.colorbar(shown, ax=ax, shrink=0.7, label=label)


# ------------------------------------------------------------------------------------- interactive maps

def classed_colors(cmap_name, n, low=0.15, high=1.0):
    cmap = plt.get_cmap(cmap_name)
    return [tuple(int(255 * c) for c in cmap(x)[:3]) for x in np.linspace(low, high, n)]


def classed_overlay(values, bins, colors, name, show=True, opacity=0.85):
    """Leaflet image layer of resolution-8 hexagons coloured by class; values below bins[0] are transparent."""
    _, bounds = display_grid()
    image = hexagon_image(values)
    rgba = np.zeros((*image.shape, 4), dtype=np.uint8)
    classes = np.digitize(np.nan_to_num(image, nan=-np.inf), bins)  # 0 = below the first break
    for number, color in enumerate(colors, start=1):
        rgba[classes == number] = (*color, 255)
    return folium.raster_layers.ImageOverlay(rgba, bounds=bounds, mercator_project=True, name=name, show=show,
                                             opacity=opacity, interactive=False, zindex=2)


def base_map(zoom=7):
    guinea, ecoregions = boundaries()
    m = folium.Map(location=[10.4, -11.0], zoom_start=zoom, tiles=None, control_scale=True, prefer_canvas=True)
    folium.TileLayer("cartodbpositron", name="Light basemap").add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri, Maxar, Earthstar Geographics", name="Satellite imagery",
    ).add_to(m)
    Fullscreen().add_to(m)
    return m


def add_borders(m):
    guinea, ecoregions = boundaries()
    borders = ecoregions[["ECO_NAME", "geometry"]].copy()
    borders["geometry"] = borders.simplify(0.003)
    folium.GeoJson(borders, name="Ecoregions", style_function=lambda _: {"color": "#333333", "weight": 1, "fill": False},
                   tooltip=folium.GeoJsonTooltip(["ECO_NAME"], aliases=["Ecoregion"])).add_to(m)
    folium.GeoJson(guinea.simplify(0.002), name="Guinea", control=False,
                   style_function=lambda _: {"color": "black", "weight": 2, "fill": False}).add_to(m)


def value_layer(name, fields, aliases, resolution=6, show=False):
    """Transparent resolution-6 hexagons (~3,000 ha) that show their values on hover."""
    fine = fine_hexagons()
    parent = [h3_int.cell_to_parent(int(cell), resolution) for cell in fine["h3"]]
    sums = fine.groupby(parent)[["land_ha", "forest_2012_ha", "forest_2025_ha", "loss_2013_2019_ha",
                                 "loss_2020_2025_ha", "loss_2013_2025_ha"]].sum()
    sums["forest_cover_2025"] = 100 * sums["forest_2025_ha"] / sums["land_ha"]
    sums["annual_2013_2019"] = sums["loss_2013_2019_ha"] / 7
    sums["annual_2020_2025"] = sums["loss_2020_2025_ha"] / 6
    sums["loss_pct_2013"] = 100 * sums["loss_2013_2025_ha"] / sums["forest_2012_ha"].where(sums["forest_2012_ha"] > 0)
    # only hexagons with forest or loss; coordinates to ~100 m and values to 1 decimal keep the page light
    sums = sums[(sums["forest_2012_ha"] > 0) | (sums["loss_2013_2025_ha"] > 0)].round(1)
    features = []
    for cell, row in sums.iterrows():
        ring = [[round(lng, 3), round(lat, 3)] for lat, lng in h3_int.cell_to_boundary(int(cell))]
        properties = {field: (None if pd.isna(row[field]) else round(float(row[field]), 1)) for field in fields}
        features.append({"type": "Feature", "properties": properties,
                         "geometry": {"type": "Polygon", "coordinates": [ring + ring[:1]]}})
    return folium.GeoJson(
        {"type": "FeatureCollection", "features": features}, name=name, show=show,
        style_function=lambda _: {"color": "#666666", "weight": 0.3, "fillOpacity": 0},
        highlight_function=lambda _: {"color": "#1f4e9c", "weight": 2},
        tooltip=folium.GeoJsonTooltip(fields, aliases=aliases, localize=True, sticky=True),
    )


def legend(title_and_items, position="bottomleft"):
    """Fixed HTML legend. title_and_items = [(title, [(color, label), ...]), ...]."""
    corner = {"bottomleft": "bottom: 28px; left: 10px;", "bottomright": "bottom: 28px; right: 10px;"}[position]
    blocks = []
    for title, items in title_and_items:
        rows = "".join(
            f'<div><span style="display:inline-block;width:14px;height:12px;margin-right:6px;'
            f'background:{color};border:1px solid #999;"></span>{label}</div>' for color, label in items)
        blocks.append(f'<div style="font-weight:600;margin:4px 0 2px;">{title}</div>{rows}')
    html = (f'<div style="position:absolute;{corner}z-index:9999;background:rgba(255,255,255,0.92);padding:6px 10px;'
            f'border-radius:4px;font-size:12px;line-height:1.35;box-shadow:0 1px 4px rgba(0,0,0,0.3);">'
            f'{"".join(blocks)}</div>')
    return folium.Element(html)


def class_labels(bins, unit, decimals=1):
    labels = [f"{bins[i]:.{decimals}f} – {bins[i + 1]:.{decimals}f} {unit}" for i in range(len(bins) - 1)]
    return labels + [f"> {bins[-1]:.{decimals}f} {unit}"]


def hex_color(rgb):
    return "#{:02x}{:02x}{:02x}".format(*rgb)


# ------------------------------------------------------------------------------------- land-cover benchmark

NODATA = -1
HARMONISED_CLASSES = {1: "Tree cover", 2: "Shrub & grassland", 3: "Cropland", 4: "Built-up",
                      5: "Bare / sparse vegetation", 6: "Water", 7: "Herbaceous wetland"}
HARMONISED_COLORS = {"Tree cover": "#2e7d32", "Shrub & grassland": "#c5b358", "Cropland": "#f39c12",
                     "Built-up": "#c0392b", "Bare / sparse vegetation": "#a1887f", "Water": "#2980b9",
                     "Herbaceous wetland": "#7fcdbb"}
PRODUCTS = {  # same definitions as studies/lulc_benchmark.ipynb, section 3
    "esri": {"label": "Esri LULC", "year": 2020, "forest_classes": [2], "nodata": [0, 9, 10],
             "crosswalk": {1: 6, 2: 1, 4: 7, 5: 3, 7: 4, 8: 5, 11: 2}},
    "dynamic_world": {"label": "Dynamic World", "year": 2020, "forest_classes": [1], "nodata": [8],
                      "crosswalk": {0: 6, 1: 1, 2: 2, 3: 7, 4: 3, 5: 2, 6: 4, 7: 5}},
    "worldcover": {"label": "ESA WorldCover", "year": 2020, "forest_classes": [10, 95], "nodata": [0, 70],
                   "crosswalk": {10: 1, 20: 2, 30: 2, 40: 3, 50: 4, 60: 5, 80: 6, 90: 7, 95: 1, 100: 2}},
    "gfc2020": {"label": "JRC GFC2020", "year": 2020, "forest_classes": [1], "nodata": []},
    "glc_fcs30d": {"label": "GLC_FCS30D", "year": 2020, "nodata": [0, 220, 250, 255],
                   "forest_classes": [51, 52, 61, 62, 71, 72, 81, 82, 91, 92, 185]},
    "cgls_lc100": {"label": "Copernicus LC100", "year": 2019, "nodata": [0, 70, 255],
                   "forest_classes": [*range(111, 117), *range(121, 127)]},
    "hansen": {"label": "Hansen GFC (> 30 %)", "year": 2020},
}
PRODUCT_COLORS = {"Esri LULC": "#1f77b4", "Dynamic World": "#17becf", "ESA WorldCover": "#2ca02c",
                  "JRC GFC2020": "#9467bd", "GLC_FCS30D": "#8c564b", "Copernicus LC100": "#e377c2",
                  "Hansen GFC (> 30 %)": "#d62728", "Hansen GFC (> 10 %)": "#ff9896"}


@lru_cache
def benchmark():
    """Sample points, forest (1/0/NaN) and harmonised land cover per product, as read by lulc_benchmark.ipynb."""
    points = pd.read_parquet(BENCHMARK_DIR / "sample_points_h3res7.parquet").set_index("point_id")
    forest, lulc = {}, {}
    for key, spec in PRODUCTS.items():
        values = pd.read_parquet(BENCHMARK_DIR / f"{key}_h3res7.parquet").set_index("point_id").reindex(points.index)
        if key == "hansen":
            lost = (values["lossyear"] > 0) & (values["lossyear"] <= spec["year"] - 2000)
            no_data = values["datamask"] == NODATA
            for threshold, label in [(30, spec["label"]), (10, "Hansen GFC (> 10 %)")]:
                forest[label] = ((values["treecover2000"] > threshold) & ~lost).astype(float).mask(no_data)
            continue
        native = values[str(spec["year"])]
        no_data = native.isin([NODATA, *spec["nodata"]])
        forest[spec["label"]] = native.isin(spec["forest_classes"]).astype(float).mask(no_data)
        if "crosswalk" in spec:
            lulc[spec["label"]] = native.map(spec["crosswalk"]).where(~no_data)
    return points, pd.DataFrame(forest), pd.DataFrame(lulc)


def area_estimate(indicator, weights_ha):
    """Area (ha) where `indicator` is 1 and its standard error. Missing values count as 0."""
    indicator = indicator.fillna(0).astype(float)
    weights_ha = weights_ha.reindex(indicator.index)
    share = np.average(indicator, weights=weights_ha)
    total = weights_ha.sum()
    return total * share, total * np.sqrt(share * (1 - share) / len(indicator))


def markdown_table(frame, floatfmt=",.0f", index=True):
    """A pandas table as a Markdown pipe table (rendered and styled by Quarto)."""
    from IPython.display import Markdown
    return Markdown(frame.to_markdown(floatfmt=floatfmt, index=index))
