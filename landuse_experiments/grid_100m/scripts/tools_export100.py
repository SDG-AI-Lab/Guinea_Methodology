"""Export the inputs for running MOLUSCE, PLUS and FLUS by hand at 100 m (notebook 07): the same files
as the 250 m export (scripts/benchmark_export.py), on the platform's 100 m grid, with the predictors
as main builds them (predictors_lu100) and the class codes of the 250 m legend.

    python tools_export100.py      ->  E:\\SDG AI Lab\\Guinea\\data_100m\\landuse\\benchmark\\

See grid_250m/2_improvement_and_benchmark/BENCHMARK_GUIDE.md, "Runs at 100 m".
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
import predictors_lu100 as P                                        # noqa: E402
import classes100 as C                                              # noqa: E402

G = P.G
OUT = P.LANDUSE / "benchmark"
LEGEND_250 = HERE.parent / "outputs" / "benchmark" / "legend_and_demand.csv"
YEARS_IN = (2023, 2024)                                             # learn 2023 -> 2024, forecast 2025
YEAR_REF = 2025                                                     # for checking only, never an input
PROJECTED_FROM = P.LANDUSE / "runs" / "test_s42" / "demand.csv"     # the platform's projection for 2025


def _profile(dtype, nodata):
    """The grid's profile, untiled like the 250 m export the three tools have already read."""
    prof = {k: v for k, v in G.profile(dtype, nodata).items() if k not in ("blockxsize", "blockysize")}
    return {**prof, "tiled": False}


def _legend():
    leg = pd.read_csv(LEGEND_250)[["code", "esri_code", "class"]]
    return leg


def _recode(values, leg):
    out = np.zeros(values.shape, np.uint8)                          # 0 = nodata (outside Guinea, cloud)
    for code, esri in zip(leg.code, leg.esri_code):
        out[values == esri] = code
    return out


def landcover(leg):
    lc_prof = _profile("uint8", 0)
    for y, folder in [(y, "inputs") for y in YEARS_IN] + [(YEAR_REF, "reference")]:
        name = f"landcover_{y}.tif" if folder == "inputs" else f"landcover_{y}_reference.tif"
        path = OUT / folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(path, "w", **lc_prof) as dst:
            dst.write(G.to_grid(_recode(C.class_100(y), leg), fill=0, dtype="uint8"), 1)


def factors(year):
    """The 12 predictors dated to `year`, -9999 outside Guinea; and the 0-1 version for FLUS (0 outside,
    no nodata value: FLUS rescales by each layer's minimum and maximum)."""
    folder, flus = OUT / "inputs" / f"factors_{year}", OUT / "inputs" / f"factors_{year}_flus"
    folder.mkdir(parents=True, exist_ok=True)
    flus.mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(P.NAMES, start=1):
        fname = f"{i:02d}_{name.lower()}.tif"
        v = P.predictor(name, year)
        with rasterio.open(folder / fname, "w", **_profile("float32", -9999.0)) as dst:
            # cells without a value inside Guinea (17 of slope and aspect, at the DEM's edge) as nodata too: a
            # NaN makes MOLUSCE's normalisation mask the whole layer, and it then finds no valid sample
            dst.write(G.to_grid(np.where(np.isnan(v), -9999.0, v), fill=-9999.0, dtype="float32"), 1)
        lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
        scaled = np.nan_to_num((v - lo) / (hi - lo) if hi > lo else np.zeros_like(v)).astype("float32")
        with rasterio.open(flus / fname, "w", **_profile("float32", None)) as dst:
            dst.write(G.to_grid(scaled, fill=0.0, dtype="float32"), 1)


def demand(leg):
    """Pixels per class in 2023, 2024 and 2025 (observed), and the 2025 total of each class under the
    platform's projected transitions (the test run's demand.csv: change analysis 2017-2024, trend
    fitted per transition), clouds left out; rounding absorbed by the largest class so the totals add
    up to the 2024 map, as CLUE-S, PLUS and FLUS require."""
    counts = {y: np.bincount(_recode(C.class_100(y), leg), minlength=len(leg) + 1)[1:] for y in (*YEARS_IN, YEAR_REF)}
    d = pd.read_csv(PROJECTED_FROM, index_col=0)
    d.columns, d.index = d.columns.astype(int), d.index.astype(int)
    esri = list(leg.esri_code)
    d = d.reindex(index=esri, columns=esri).fillna(0)
    np.fill_diagonal(d.values, 0)
    projected = counts[2024] + d.sum(axis=0).values - d.sum(axis=1).values
    projected = np.round(projected).astype(int)
    projected[np.argmax(projected)] += int(counts[2024].sum() - projected.sum())
    out = leg.copy()
    out["pixels_2023"], out["pixels_2024"] = counts[2023], counts[2024]
    out[f"pixels_{YEAR_REF}_observed"], out[f"pixels_{YEAR_REF}_projected"] = counts[YEAR_REF], projected
    out.to_csv(OUT / "legend_and_demand.csv", index=False)
    d.to_csv(OUT / f"transitions_{YEAR_REF}_projected.csv")
    return out


def stack_leas(folder=None):
    """PLUS's LEAS writes one file per class (leas_result_band_1..7.tif); CARS reads one file with one
    band per class. Stacks them in class order into leas_probability_7band.tif."""
    folder = Path(folder or OUT / "output_plus")
    files = [folder / f"leas_development_potential_band_{i}.tif" for i in range(1, 8)]
    with rasterio.open(files[0]) as src:
        prof = {**src.profile, "count": len(files)}
    with rasterio.open(folder / "leas_probability_7band.tif", "w", **prof) as dst:
        for i, f in enumerate(files, start=1):
            with rasterio.open(f) as src:
                dst.write(src.read(1), i)
    print("Written", folder / "leas_probability_7band.tif")


if __name__ == "__main__":
    if sys.argv[1:] == ["stack-leas"]:
        stack_leas()
        sys.exit(0)
    if sys.argv[1:] == ["factors"]:                                 # the factor files only (fix of 2026-10-31)
        for y in YEARS_IN:
            factors(y)
        sys.exit(0)
    leg = _legend()
    landcover(leg)
    for y in YEARS_IN:
        factors(y)
    print(demand(leg).to_string(index=False))
    sizes = sum(f.stat().st_size for f in OUT.rglob("*.tif")) / 1e9
    print(f"Written to {OUT} ({sizes:.1f} GB)")
