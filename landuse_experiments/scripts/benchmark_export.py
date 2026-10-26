"""Export the inputs for testing other land-change tools (MOLUSCE, PLUS) on exactly
the data the notebooks use: ESRI land cover on the deployed 250 m grid, Guinea only,
and the deployed method's twelve predictors dated to each year.

    python benchmark_export.py        ->  outputs/benchmark/

See 2_improvement_and_benchmark/BENCHMARK_GUIDE.md for how to use them.
"""
import numpy as np
import pandas as pd
import rasterio

import esri_methods as E
import nb_helpers as H

OUT = H.OUTPUTS / "benchmark"
YEARS_IN = (2023, 2024)      # training interval: the tools learn 2023 -> 2024 and forecast 2025
YEAR_REF = 2025              # the year scored; for checking only, never as an input


def main():
    prof = E.grid()
    g = H.guinea_mask(prof)
    maps = {y: E.esri_cached(y) for y in (*YEARS_IN, YEAR_REF)}
    present = sorted(set(np.unique(np.concatenate([maps[y][g] for y in maps]))) - set(H.ESRI_INVALID))
    code = {esri: i for i, esri in enumerate(present, start=1)}     # tools such as PLUS want 1..N

    def recode(a):
        out = np.zeros(a.shape, np.uint8)                          # 0 = nodata (outside Guinea / cloud)
        for esri, i in code.items():
            out[g & (a == esri)] = i
        return out

    lc_prof = {**prof, "dtype": "uint8", "nodata": 0, "count": 1, "compress": "deflate"}
    (OUT / "inputs").mkdir(parents=True, exist_ok=True)
    (OUT / "reference").mkdir(parents=True, exist_ok=True)
    for y in YEARS_IN:
        with rasterio.open(OUT / "inputs" / f"landcover_{y}.tif", "w", **lc_prof) as dst:
            dst.write(recode(maps[y]), 1)
    ref = recode(maps[YEAR_REF])
    with rasterio.open(OUT / "reference" / f"landcover_{YEAR_REF}_reference.tif", "w", **lc_prof) as dst:
        dst.write(ref, 1)

    # Predictors: the deployed stack, time-varying layers dated to the year (none later)
    dmask = E.deployed_mask(prof)
    f_prof = {**prof, "dtype": "float32", "nodata": -9999.0, "count": 1, "compress": "deflate"}
    for y in YEARS_IN:
        stack = E.predictor_stack(prof, dmask, y)
        folder = OUT / "inputs" / f"factors_{y}"
        folder.mkdir(exist_ok=True)
        for i, name in enumerate(E.PREDICTORS, start=1):
            fname = f"{i:02d}_{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.tif"
            with rasterio.open(folder / fname, "w", **f_prof) as dst:
                dst.write(np.where(g, stack[i - 1], -9999.0).astype(np.float32), 1)

    legend = pd.DataFrame({"code": list(code.values()), "esri_code": list(code.keys()),
                           "class": [H.ESRI_CLASSES[k] for k in code]})
    legend["pixels_2023"] = [int((recode(maps[2023]) == i).sum()) for i in legend.code]
    legend["pixels_2024"] = [int((recode(maps[2024]) == i).sum()) for i in legend.code]
    legend[f"pixels_{YEAR_REF}_observed"] = [int((ref == i).sum()) for i in legend.code]
    legend.to_csv(OUT / "legend_and_demand.csv", index=False)
    print(legend.to_string(index=False))
    print("Written to", OUT)


def another_year(year):
    """Land cover, predictors and FLUS's 0-1 predictors for one more start year (the 2022 start of
    the back-test, notebook 11), with the same class codes as legend_and_demand.csv. Writes only
    files that do not exist yet."""
    prof = E.grid()
    g = H.guinea_mask(prof)
    legend = pd.read_csv(OUT / "legend_and_demand.csv")
    land = OUT / "inputs" / f"landcover_{year}.tif"
    if not land.exists():
        a, out = E.esri_cached(year), np.zeros((prof["height"], prof["width"]), np.uint8)
        for code, esri in zip(legend.code, legend.esri_code):
            out[g & (a == esri)] = code
        with rasterio.open(land, "w", **{**prof, "dtype": "uint8", "nodata": 0, "count": 1, "compress": "deflate"}) as dst:
            dst.write(out, 1)
    folder = OUT / "inputs" / f"factors_{year}"
    if not folder.exists():
        folder.mkdir()
        stack = E.predictor_stack(prof, E.deployed_mask(prof), year)
        f_prof = {**prof, "dtype": "float32", "nodata": -9999.0, "count": 1, "compress": "deflate"}
        for i, name in enumerate(E.PREDICTORS, start=1):
            fname = f"{i:02d}_{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.tif"
            with rasterio.open(folder / fname, "w", **f_prof) as dst:
                dst.write(np.where(g, stack[i - 1], -9999.0).astype(np.float32), 1)
    if not (OUT / "inputs" / f"factors_{year}_flus").exists():
        flus_factors(year)


def cubes():
    """The same predictors as one multi-band raster per year (Dinamica EGO's 'static variables'
    cube), band names = predictor names."""
    folder = OUT / "inputs"
    for y in YEARS_IN:
        files = sorted((folder / f"factors_{y}").glob("*.tif"))
        with rasterio.open(files[0]) as src:
            prof = {**src.profile, "count": len(files)}
        with rasterio.open(folder / f"factors_{y}_cube.tif", "w", **prof) as dst:
            for i, f in enumerate(files, start=1):
                with rasterio.open(f) as src:
                    dst.write(src.read(1), i)
                dst.set_band_description(i, f.stem[3:])
        print("Written", folder / f"factors_{y}_cube.tif")


def flus_factors(year=2024):
    """The predictors scaled to 0-1 inside Guinea and 0 outside, with no nodata value, for
    GeoSOS-FLUS: it rescales every layer by its minimum and maximum, which the -9999 nodata
    would distort. Used with FLUS's "No Normalization" option."""
    src_dir, dst_dir = OUT / "inputs" / f"factors_{year}", OUT / "inputs" / f"factors_{year}_flus"
    dst_dir.mkdir(parents=True, exist_ok=True)
    for f in sorted(src_dir.glob("*.tif")):
        with rasterio.open(f) as src:
            a, prof = src.read(1).astype("float32"), src.profile
        ok = a != -9999
        lo, hi = float(a[ok].min()), float(a[ok].max())
        scaled = np.where(ok, (a - lo) / (hi - lo) if hi > lo else 0.0, 0.0).astype("float32")
        prof.update(dtype="float32", nodata=None)
        with rasterio.open(dst_dir / f.name, "w", **prof) as dst:
            dst.write(scaled, 1)
    print("Written", dst_dir)


def projected_demand():
    """The amount of change a user would have for 2025: 2023 -> 2024 projected one step by a
    Markov chain (E.projected_counts). Writes the projected transition counts (ESRI codes) and
    adds the projected total of each class to legend_and_demand.csv, for the comparison with
    projected amounts (notebook 10). Rounding is absorbed by the largest class, so the totals add
    up to the 2024 map, as CLUE-S, PLUS and FLUS require."""
    g = H.guinea_mask(E.grid())
    counts = E.projected_counts(*YEARS_IN, g)
    counts.to_csv(OUT / f"transitions_{YEAR_REF}_projected.csv")
    legend = pd.read_csv(OUT / "legend_and_demand.csv")
    projected = np.array([int(counts[e].sum()) for e in legend.esri_code])
    projected[np.argmax(projected)] += int(legend.pixels_2024.sum() - projected.sum())
    legend[f"pixels_{YEAR_REF}_projected"] = projected
    legend.to_csv(OUT / "legend_and_demand.csv", index=False)
    print(legend.to_string(index=False))


if __name__ == "__main__":
    main()
    cubes()
    flus_factors()
    projected_demand()
