"""Shared helpers for the MODIS reproduction notebooks (01-04).

The notebooks run the intern's code unchanged, from local_landuse/, exactly
as its command-line entry points do. Nothing here changes a result; it starts
runs, loads their saved outputs and draws the comparison tables and figures.
"""

import os

# PostGIS sets PROJ_LIB/GDAL_DATA machine-wide; they override rasterio's own
# proj.db and every CRS lookup fails. Must run before rasterio is imported.
for _key in ("PROJ_LIB", "PROJ_DATA", "GDAL_DATA"):
    os.environ.pop(_key, None)

import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

# landuse_experiments/: this file is in scripts/. The intern's package is in
# 1_intern_improvement/local_landuse and reads data/ and writes outputs/ here.
ROOT = Path(__file__).resolve().parents[1]
INTERN = ROOT / "1_intern_improvement"
_ENV = {**os.environ, "PYTHONIOENCODING": "utf-8",            # subprocesses find local_landuse
        "PYTHONPATH": os.pathsep.join(filter(None, [str(INTERN), os.environ.get("PYTHONPATH")]))}
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
SUMMARY = OUTPUTS / "summary"      # small tables, committed
LOGS = OUTPUTS / "logs"

# MODIS MCD12Q1 LC_Type1 (IGBP) legend; codes present in the Guinea rasters.
IGBP = {1: "Evergreen needleleaf forest", 2: "Evergreen broadleaf forest",
        4: "Deciduous broadleaf forest", 5: "Mixed forest", 8: "Woody savanna",
        9: "Savanna", 10: "Grassland", 11: "Permanent wetland", 12: "Cropland",
        13: "Urban and built-up", 14: "Cropland/natural mosaic", 16: "Barren",
        17: "Water"}

# Chart styling: reference palette, light surface (dataviz skill, palette.md).
# Only the first three categorical slots are used together on a map, the cap
# the palette documents as colour-blind safe for all-pairs comparisons.
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]          # blue, orange, aqua
NEUTRAL_FILL = "#e1e0d9"


def style_axes(ax):
    """Recessive axes: hairline grid, no top/right spines, muted ticks."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def environment():
    """Library versions, for the notebook header."""
    import sklearn, xgboost, scipy
    return pd.DataFrame([
        ("python", sys.version.split()[0]), ("numpy", np.__version__),
        ("scikit-learn", sklearn.__version__), ("xgboost", xgboost.__version__),
        ("scipy", scipy.__version__), ("rasterio", rasterio.__version__),
        ("pandas", pd.__version__)], columns=["package", "version"])


def sha256(path, length=12):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:length]


def read_map(name):
    """A raster from the vendored data/ folder, as (array, profile)."""
    with rasterio.open(DATA / name) as src:
        return src.read(1), src.profile


def read_output_raster(relative):
    with rasterio.open(OUTPUTS / relative) as src:
        return src.read(1)


# --------------------------------------------------------------------------
# Running the intern's code

# The intern's pinned environment (README "Optional: the intern's exact environment").
# Runs saved with it carry this suffix; it is used only for the reproduction checks.
INTERN_PY = Path("D:/envs/guinea-intern-py314/Scripts/python.exe")
INTERN_SUFFIX = "_py314"


def ensure_run(module, run_name, extra_args=(), rerun=False, base="outputs", python=None):
    """Return the folder <base>/<run_name>, running the module if needed.

    rerun=False: use the saved folder; run only if it does not exist yet (first setup).
    rerun=True:  run again into <base>/<run_name>_rerun_<timestamp>, so the saved
                 folder that later notebooks depend on is never overwritten.
    base is "outputs" for model runs and "data" for satellite downloads.
    The command is the intern's own entry point, run with this kernel's Python
    (or `python`, e.g. INTERN_PY) from this folder; the log goes to outputs/logs/.
    """
    target = ROOT / base / run_name
    if target.exists() and any(target.iterdir()) and not rerun:
        print(f"Using saved: {base}/{run_name}")
        return target
    if rerun:
        target = ROOT / base / f"{run_name}_rerun_{datetime.now():%Y%m%d_%H%M}"
    LOGS.mkdir(parents=True, exist_ok=True)
    log = LOGS / f"{target.name}.log"
    command = [str(python or sys.executable), "-m", module, "--output-dir", f"{base}/{target.name}", *extra_args]
    print("Running:", " ".join(command[1:]), f"\n  log: {log.relative_to(ROOT)}")
    started = datetime.now()
    with open(log, "w", encoding="utf-8") as stream:
        result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                env=_ENV)
    if result.returncode:
        raise RuntimeError(f"{module} failed (exit {result.returncode}); see {log}")
    print(f"  finished in {(datetime.now() - started).total_seconds() / 60:.1f} min")
    return target


def ensure_hmm_reference(run_name, ensemble_run, rerun=False, python=None):
    """Run the intern's hmm_reference with the 'best single member' entry pointed
    at the member OUR probability ensemble saved.

    hmm_reference.py hard-codes that member as expanded_typicality_xgb, the one the
    intern's ensemble picked. Ours started from fused_expanded_xgb (ahead of the
    typicality model by 0.003 validation points), so that file does not exist here. The vendored file is
    not edited: the list entry is replaced in memory for this run only. Every other
    line of the method runs as written.
    """
    best = json.loads((OUTPUTS / ensemble_run / "report.json").read_text())["best_single_member"]
    best = best["name"] if isinstance(best, dict) else best
    target = OUTPUTS / run_name
    if target.exists() and any(target.iterdir()) and not rerun:
        print(f"Using saved run: outputs/{run_name}  (best single member: {best})")
        return target, best
    if rerun:
        target = OUTPUTS / f"{run_name}_rerun_{datetime.now():%Y%m%d_%H%M}"
    script = (
        "import logging, local_landuse.hmm_reference as h\n"
        "logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')\n"
        f"h.ENSEMBLE_RUN = {ensemble_run!r}\n"
        "h.FORECASTS = [f for f in h.FORECASTS if f[2] != 'candidate']\n"
        f"h.FORECASTS.insert(3, ({best!r}, 'Best single member ({best})', 'candidate', "
        f"'{ensemble_run}/final_{best}.tif'))\n"
        "h.FORECASTS[-1] = (h.FORECASTS[-1][0], h.FORECASTS[-1][1], h.FORECASTS[-1][2], "
        f"'{ensemble_run}/final_probability_ensemble.tif')\n"
        f"h.run_hmm_reference('outputs/{target.name}')\n")
    LOGS.mkdir(parents=True, exist_ok=True)
    log = LOGS / f"{target.name}.log"
    print(f"Running hmm_reference with best single member = {best}\n  log: {log.relative_to(ROOT)}")
    with open(log, "w", encoding="utf-8") as stream:
        result = subprocess.run([str(python or sys.executable), "-c", script], cwd=ROOT, stdout=stream,
                                stderr=subprocess.STDOUT, env=_ENV)
    if result.returncode:
        raise RuntimeError(f"hmm_reference failed (exit {result.returncode}); see {log}")
    return target, best


# --------------------------------------------------------------------------
# 2020->2024 test (notebook 04): the intern's code hard-codes 2001/2005/2010/2015
# as history and 2020 as the target. A copy of data/ in which every real map is
# saved under the previous slot's name runs the identical code one interval later:
# select on (real) 2015->2020, forecast (real) 2020->2024.

SHIFT_ROOT = OUTPUTS / "test_2020_2024"   # own data/ + outputs/, as the code expects
SHIFT_DATA = SHIFT_ROOT / "data"
SHIFT_OUTPUTS = SHIFT_ROOT / "outputs"
SHIFT = {2001: 2005, 2005: 2010, 2010: 2015, 2015: 2020, 2020: 2024}   # slot -> real year


def prepare_shifted_data():
    """Build shift_2020_2024/data once; returns a table of slot -> real file and hash."""
    SHIFT_DATA.mkdir(parents=True, exist_ok=True)
    rows = []
    for slot, real in SHIFT.items():
        source, target = DATA / f"Guinea_LandCover_{real}.tif", SHIFT_DATA / f"Guinea_LandCover_{slot}.tif"
        if not target.exists():
            target.write_bytes(source.read_bytes())
        if sha256(source) != sha256(target):
            raise ValueError(f"{target.name} is not a copy of {source.name}")
        rows.append((f"Guinea_LandCover_{slot}.tif", f"Guinea_LandCover_{real}.tif", sha256(target)))
    terrain = SHIFT_DATA / "Guinea_SRTM_DEM_Slope_Aspect.tif"
    if not terrain.exists():
        terrain.write_bytes((DATA / terrain.name).read_bytes())
    return pd.DataFrame(rows, columns=["File name the code reads", "Real map it contains", "SHA-256 (first 12)"])


def ensure_shifted_run(run_name, module, function, extra="", rerun=False):
    """Call local_landuse.<module>.<function>(data_dir=shifted data, output_dir=...) unchanged."""
    target = SHIFT_OUTPUTS / run_name
    if target.exists() and any(target.iterdir()) and not rerun:
        print(f"Using saved run: outputs/test_2020_2024/outputs/{run_name}")
        return target
    if rerun:
        target = SHIFT_OUTPUTS / f"{run_name}_rerun_{datetime.now():%Y%m%d_%H%M}"
    script = ("import logging\n"
              "logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')\n"
              f"from local_landuse.{module} import {function}\n"
              f"{function}(data_dir=r'{SHIFT_DATA}', output_dir=r'{target}'{extra})\n")
    LOGS.mkdir(parents=True, exist_ok=True)
    log = LOGS / f"shift_{target.name}.log"
    print(f"Running {module}.{function} on the shifted data\n  log: {log.relative_to(ROOT)}")
    started = datetime.now()
    with open(log, "w", encoding="utf-8") as stream:
        result = subprocess.run([sys.executable, "-c", script], cwd=ROOT, stdout=stream,
                                stderr=subprocess.STDOUT, env=_ENV)
    if result.returncode:
        raise RuntimeError(f"{module}.{function} failed (exit {result.returncode}); see {log}")
    print(f"  finished in {(datetime.now() - started).total_seconds() / 60:.1f} min")
    return target


def read_csv(path):
    return pd.read_csv(path)


# --------------------------------------------------------------------------
# Comparison with the intern's write-ups

# Reproduction tolerances, measured in notebook 01 section 4: the same code run in our
# environment (Python 3.10, xgboost 3.2, scikit-learn 1.7) and in the intern's pinned one
# (Python 3.14, xgboost 3.4.1, scikit-learn 1.9.0). Versions alone moved RF and fixed-demand
# rows by at most 0.33 points, and XGBoost rows with validation-tuned thresholds by up to 0.85.
TOL = 0.4                 # RF, fixed-demand and rule rows
TOL_XGB_THRESHOLD = 0.85  # XGBoost (or XGBoost-based ensemble) rows with tuned decision thresholds
REPRODUCES = "Reproduces (|Δ| ≤ tolerance)"


def compare(rows):
    """rows: (configuration, intern FoM, our FoM, intern acc., our acc.[, tolerance]); FoM and
    acc. in %. Tolerance defaults to TOL; pass TOL_XGB_THRESHOLD for threshold-tuned XGBoost rows."""
    rows = [tuple(r) + (TOL,) * (6 - len(r)) for r in rows]
    frame = pd.DataFrame(rows, columns=["Configuration", "Intern FoM %", "Ours FoM %",
                                        "Intern accuracy %", "Ours accuracy %", "Tolerance (pts)"])
    frame["Δ FoM (pts)"] = frame["Ours FoM %"] - frame["Intern FoM %"]
    frame[REPRODUCES] = (frame["Δ FoM (pts)"].abs() <= frame["Tolerance (pts)"] + 1e-9).map({True: "yes", False: "NO"})
    cols = ["Configuration", "Intern FoM %", "Ours FoM %", "Δ FoM (pts)",
            "Intern accuracy %", "Ours accuracy %", "Tolerance (pts)", REPRODUCES]
    return frame[cols]


# Our reruns are named by their package versions. Each notebook sets RERUN_LABEL to the
# environment its "Ours" columns come from; show(), relabel() and plot_dumbbell() use it.
RERUN_PY310 = "Rerun (Python 3.10, xgboost 3.2)"
RERUN_PY314 = "Rerun (Python 3.14, xgboost 3.4.1)"
RERUN_LABEL = "Rerun"


def relabel(frame):
    """Column names with 'Ours'/'ours' replaced by RERUN_LABEL (for display and saved tables)."""
    import re
    return frame.rename(columns=lambda c: re.sub(r"\b[Oo]urs\b", RERUN_LABEL, c) if isinstance(c, str) else c)


def show(frame, precision=2):
    """Notebook display with fixed decimals, blank cells for missing values, rerun labels."""
    return relabel(frame).style.format(precision=precision, na_rep="").hide(axis="index")


def row(frame, key_col, key, value_col):
    match = frame.loc[frame[key_col] == key, value_col]
    if match.empty:
        raise KeyError(f"{key} not found in {key_col}")
    return float(match.iloc[0])


# --------------------------------------------------------------------------
# Figures

def plot_dumbbell(frame, title, ax=None, left="Intern FoM %", right="Ours FoM %",
                  labels=None, xlabel="Forecast Figure of Merit, 2015→2020 (%)"):
    """Two values per configuration (default: intern's write-up vs our rerun), on one axis."""
    import matplotlib.pyplot as plt
    labels = labels or ("Intern (write-up)", RERUN_LABEL)
    data = frame.iloc[::-1]
    if ax is None:
        _, ax = plt.subplots(figsize=(8.5, 0.42 * len(data) + 1.2))
    y = np.arange(len(data))
    for yi, a, b in zip(y, data[left], data[right]):
        ax.plot([a, b], [yi, yi], color=AXIS, linewidth=2, zorder=1)
    # First value as a hollow ring, second as a smaller filled dot inside it: both
    # stay visible when the two values coincide.
    ax.scatter(data[left], y, s=150, facecolor="none", edgecolor=SERIES[0],
               linewidth=2, zorder=2, label=labels[0])
    ax.scatter(data[right], y, s=40, color=SERIES[1], edgecolor=SURFACE,
               linewidth=1, zorder=3, label=labels[1])
    ax.set_yticks(y, data["Configuration"])
    ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
    style_axes(ax)
    ax.set_title(title, loc="left", color=INK, fontsize=11, pad=26)
    ax.legend(frameon=False, fontsize=9, loc="lower left", bbox_to_anchor=(0, 1.0),
              ncol=2, labelcolor=INK_2, handletextpad=0.4, borderaxespad=0.2)
    return ax


def outcome_map(start, observed, predicted, mask):
    """Per-pixel outcome codes: 0 outside, 1 no change either, 2 hit,
    3 false alarm, 4 missed, 5 wrong destination."""
    out = np.zeros(start.shape, np.uint8)
    obs_change = (observed != start) & mask
    pred_change = (predicted != start) & mask
    out[mask] = 1
    out[obs_change & pred_change & (predicted == observed)] = 2
    out[~obs_change & pred_change] = 3
    out[obs_change & ~pred_change] = 4
    out[obs_change & pred_change & (predicted != observed)] = 5
    return out


def plot_outcome_map(codes, title, ax=None, legend=True, profile=None, window=None):
    """Hits, false alarms and misses in the three palette slots; wrong
    destination in muted ink; unchanged pixels as a neutral fill. With `profile`
    (and `window` for a crop), unchanged pixels show the satellite image instead."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6.2))
    cmap = ListedColormap(["#ffffff", NEUTRAL_FILL, SERIES[0], SERIES[1], SERIES[2], MUTED])
    if profile is not None:
        plot_satellite(ax, profile, window)
        codes_shown = np.ma.masked_less_equal(codes, 1)
    else:
        codes_shown = codes
    ax.imshow(codes_shown, cmap=cmap, vmin=0, vmax=5, interpolation="nearest")
    counts = {k: int((codes == k).sum()) for k in range(2, 6)}
    labels = {2: "Hit (right place, right class)", 3: "False alarm",
              4: "Missed change", 5: "Wrong destination class"}
    colours = {2: SERIES[0], 3: SERIES[1], 4: SERIES[2], 5: MUTED}
    if legend:
        box = dict(frameon=True, facecolor=SURFACE, edgecolor="none", framealpha=0.9) if profile is not None else dict(frameon=False)
        ax.legend(handles=[Patch(color=colours[k], label=f"{labels[k]}: {counts[k]:,} px") for k in labels],
                  loc="lower left", fontsize=8.5, labelcolor=INK_2, **box)
        if profile is not None:
            satellite_credit(ax.figure)
    ax.set_axis_off()
    ax.set_title(title, loc="left", color=INK, fontsize=10 if not legend else 11)
    return ax


# --------------------------------------------------------------------------
# ESRI notebooks (05-06): the deployed landuse_tool package, imported read-only.

REPO = ROOT.parent
LANDUSE_TOOL_PARENT = REPO / "Land Use-Deforestation"
if str(LANDUSE_TOOL_PARENT) not in sys.path:
    sys.path.insert(0, str(LANDUSE_TOOL_PARENT))

# The run behind the methodology document's 13.80% (copied from
# E:/SDG AI Lab/Guinea/Land Use-Deforestation/outputs_tiff_landuse2).
DEPLOYED_RUN = DATA / "esri_deployed_run_250m"
BOUNDARY = LANDUSE_TOOL_PARENT / "data" / "geoBoundaries-GIN-ADM0.geojson"

ESRI_CLASSES = {1: "Water", 2: "Trees", 4: "Flooded Vegetation", 5: "Crops",
                7: "Built Area", 8: "Bare Ground", 11: "Rangeland"}
ESRI_INVALID = [0, 10, 255]          # nodata, Clouds, export nodata


def read_tif(path):
    with rasterio.open(path) as src:
        return src.read(1), src.profile


def guinea_mask(profile):
    """True for pixels whose centre lies inside Guinea's geoBoundaries ADM0 polygon."""
    from rasterio.features import geometry_mask
    shapes = [f["geometry"] for f in json.loads(BOUNDARY.read_text())["features"]]
    return ~geometry_mask(shapes, out_shape=(profile["height"], profile["width"]),
                          transform=profile["transform"])


def write_masked_copy(source, target, keep):
    """Copy a class raster, setting pixels outside `keep` to 255 (export nodata)."""
    array, profile = read_tif(source)
    array = np.where(keep, array, 255).astype(array.dtype)
    profile.update(nodata=255)
    Path(target).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(target, "w", **profile) as dst:
        dst.write(array, 1)
    return Path(target)


def collect_surfaces(prefix, folder=None):
    """Exported suitability surfaces {(from_id, to_id): path}, named like
    'expander_FloodedVegetation_to_Crops.tif' (class names, spaces removed)."""
    import re
    folder = Path(folder or DEPLOYED_RUN)
    squashed = {n.replace(" ", ""): i for i, n in ESRI_CLASSES.items()}
    found = {}
    for path in folder.glob(f"{prefix}_*_to_*.tif"):
        m = re.match(rf"^{prefix}_(.+?)_to_(.+?)\.tif$", path.name, re.I)
        if m and m.group(1) in squashed and m.group(2) in squashed:
            found[(squashed[m.group(1)], squashed[m.group(2)])] = str(path)
    return found


def deployed_atlases(folder=None):
    """The three simulation modes exactly as the validation cells built them."""
    exp, pat = collect_surfaces("expander", folder), collect_surfaces("patcher", folder)
    growth = {(f, t, "expander"): v for (f, t), v in exp.items() if (f, t) in pat}
    growth.update({(f, t, "patcher"): v for (f, t), v in pat.items() if (f, t) in exp})
    return {"Expander-only": exp, "Patcher-only": pat, "Growth-mode (dilation)": growth}


def score(baseline, observed, simulated, mask):
    """landuse_tool.validation.figure_of_merit on `mask`, minus invalid codes in any map."""
    from landuse_tool import validation as val
    m = mask & ~(np.isin(baseline, ESRI_INVALID) | np.isin(observed, ESRI_INVALID)
                 | np.isin(simulated, ESRI_INVALID))
    return val.figure_of_merit(baseline, observed, simulated, mask=m)


# --------------------------------------------------------------------------
# ESRI maps

# The class colours ESRI publishes with its Land Cover product, so the maps read
# like the source data.
ESRI_COLOURS = {1: "#419bdf", 2: "#397d49", 4: "#7a87c6", 5: "#e49635", 7: "#c4281b",
                8: "#a59b8f", 11: "#e3e2c3"}


def busiest_window(changed, size=160, inside=None, min_inside=0.98):
    """Top-left (row, col, size) of the size x size block with the most `changed` pixels,
    among non-overlapping blocks lying (almost) entirely inside `inside`.
    160 px is ~40 km on the 250 m grid."""
    h, w = (changed.shape[0] // size) * size, (changed.shape[1] // size) * size
    block = lambda a: a[:h, :w].reshape(h // size, size, w // size, size).sum(axis=(1, 3))
    counts = block(changed).astype(float)
    if inside is not None:
        counts[block(inside) < min_inside * size * size] = -1
    r, c = np.unravel_index(np.argmax(counts), counts.shape)
    return int(r * size), int(c * size), size


def window_label(window, profile):
    """Centre of a window in degrees, for the caption."""
    r, c, size = window
    lon, lat = profile["transform"] * (c + size / 2, r + size / 2)
    return f"{size * 0.25:.0f} km window centred {abs(lat):.2f}°{'N' if lat >= 0 else 'S'}, {abs(lon):.2f}°{'W' if lon < 0 else 'E'}"


def crop(array, window):
    r, c, size = window
    return array[r:r + size, c:c + size]


def mark_window(ax, window, label=None, colour=INK):
    from matplotlib.patches import Rectangle
    r, c, size = window
    ax.add_patch(Rectangle((c, r), size, size, fill=False, edgecolor=colour, linewidth=1.5))
    if label:
        ax.annotate(label, (c + size, r), xytext=(3, 0), textcoords="offset points", va="top",
                    color=colour, fontsize=9, fontweight="bold")


def plot_classes(land_cover, title, ax, mask=None):
    """ESRI classes in the product's own colours; outside `mask` white."""
    from matplotlib.colors import ListedColormap
    codes = sorted(ESRI_COLOURS)
    index = np.zeros(land_cover.shape, np.uint8)
    for i, k in enumerate(codes, start=1):
        index[land_cover == k] = i
    if mask is not None:
        index[~mask] = 0
    ax.imshow(index, cmap=ListedColormap(["#ffffff"] + [ESRI_COLOURS[k] for k in codes]),
              vmin=0, vmax=len(codes), interpolation="nearest")
    ax.set_axis_off()
    ax.set_title(title, loc="left", color=INK, fontsize=10)
    return ax


def esri_class_legend(fig, present, **kw):
    from matplotlib.patches import Patch
    handles = [Patch(color=ESRI_COLOURS[k], label=ESRI_CLASSES[k]) for k in sorted(present) if k in ESRI_COLOURS]
    fig.legend(handles=handles, frameon=False, fontsize=8.5, labelcolor=INK_2, ncol=len(handles), **kw)


def outcome_legend(fig, satellite=False, credit=True, **kw):
    """One shared legend for a row of outcome maps (counts are printed separately)."""
    from matplotlib.patches import Patch
    items = [(SERIES[0], "Hit (right place, right class)"), (SERIES[1], "False alarm"),
             (SERIES[2], "Missed change"), (MUTED, "Wrong destination class")]
    if not satellite:
        items.append((NEUTRAL_FILL, "No change, either map"))
    fig.legend(handles=[Patch(color=c, label=l) for c, l in items], frameon=False, fontsize=8.5,
               labelcolor=INK_2, ncol=len(items), **kw)
    if satellite and credit:
        satellite_credit(fig)


# --------------------------------------------------------------------------
# Satellite basemap: Esri World Imagery, reprojected onto a raster's own pixel
# grid so it lines up with every map drawn in pixel coordinates (windows too).

BASEMAP_CACHE = OUTPUTS / "basemap"
BASEMAP_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
BASEMAP_CREDIT = ("Satellite image: Esri World Imagery (Esri, Maxar, Earthstar Geographics, GIS User Community); "
                  "a mosaic of recent dates, not the map year")
_MERC = 20037508.342789244


def _tile(z, x, y):
    import io
    import urllib.request
    from PIL import Image
    req = urllib.request.Request(BASEMAP_URL.format(z=z, x=x, y=y), headers={"User-Agent": "guinea-landuse-notebooks"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return np.asarray(Image.open(io.BytesIO(r.read())).convert("RGB"))


def satellite(profile, window=None, scale=None, max_tiles=400):
    """RGB uint8 image of `profile`'s grid (or a (row, col, size) window of it),
    `scale` image pixels per raster pixel (default: 1 for a whole map, 8 for a
    window). Cached in outputs/basemap/."""
    import math
    from concurrent.futures import ThreadPoolExecutor
    from PIL import Image
    from rasterio.transform import Affine
    from rasterio.warp import Resampling, reproject, transform_bounds
    r0, c0, n = window or (0, 0, None)
    scale = scale or (8 if window else 1)
    h, w = (n, n) if n else (profile["height"], profile["width"])
    dst_t = profile["transform"] * Affine.translation(c0, r0) * Affine.scale(1 / scale)
    dh, dw = h * scale, w * scale
    key = hashlib.sha1(f"{profile['crs']}|{tuple(dst_t)[:6]}|{dw}|{dh}".encode()).hexdigest()[:16]
    cached = BASEMAP_CACHE / f"{key}.png"
    if cached.exists():
        return np.asarray(Image.open(cached))
    left, top = dst_t * (0, 0)
    right, bottom = dst_t * (dw, dh)
    lon0, lat0, lon1, lat1 = transform_bounds(profile["crs"], "EPSG:4326", left, bottom, right, top)
    # Zoom whose tile pixels are about as fine as the output pixels
    res_m = (lon1 - lon0) * 111320 * math.cos(math.radians((lat0 + lat1) / 2)) / dw
    z = int(min(17, max(1, math.ceil(math.log2(156543.03 * math.cos(math.radians((lat0 + lat1) / 2)) / res_m)))))
    tx = lambda lon, z: int((lon + 180) / 360 * 2 ** z)
    ty = lambda lat, z: int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * 2 ** z)
    while (tx(lon1, z) - tx(lon0, z) + 1) * (ty(lat0, z) - ty(lat1, z) + 1) > max_tiles:
        z -= 1
    xs, ys = range(tx(lon0, z), tx(lon1, z) + 1), range(ty(lat1, z), ty(lat0, z) + 1)
    with ThreadPoolExecutor(8) as pool:
        tiles = dict(zip([(x, y) for y in ys for x in xs], pool.map(lambda xy: _tile(z, *xy), [(x, y) for y in ys for x in xs])))
    mosaic = np.zeros((3, len(ys) * 256, len(xs) * 256), np.uint8)
    for (x, y), t in tiles.items():
        i, j = (y - ys[0]) * 256, (x - xs[0]) * 256
        mosaic[:, i:i + 256, j:j + 256] = t.transpose(2, 0, 1)
    size = 2 * _MERC / 2 ** z
    src_t = Affine(size / 256, 0, -_MERC + xs[0] * size, 0, -size / 256, _MERC - ys[0] * size)
    out = np.zeros((3, dh, dw), np.uint8)
    for b in range(3):
        reproject(mosaic[b], out[b], src_transform=src_t, src_crs="EPSG:3857",
                  dst_transform=dst_t, dst_crs=profile["crs"], resampling=Resampling.bilinear)
    img = out.transpose(1, 2, 0)
    BASEMAP_CACHE.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(cached)
    return img


def plot_satellite(ax, profile, window=None, scale=None, title=None):
    """Draw the satellite image in the raster's pixel coordinates, so rasters,
    windows and marks drawn on the same axes line up with it."""
    h, w = (window[2], window[2]) if window else (profile["height"], profile["width"])
    ax.imshow(satellite(profile, window, scale), extent=(-0.5, w - 0.5, h - 0.5, -0.5), interpolation="bilinear")
    ax.set_axis_off()
    if title:
        ax.set_title(title, loc="left", color=INK, fontsize=10)
    return ax


def satellite_credit(fig, y=0.0):
    fig.text(0.01, y, BASEMAP_CREDIT, fontsize=7, color=MUTED, ha="left", va="bottom")
