"""
Rayleigh-corrected reflectance (rhorc) for the two Conakry dates, as MADOS was built.

Kikaki et al. (2024) made MADOS from Acolite (Vanhellemont & Ruddick 2018) Rayleigh-corrected
reflectance at 10 m. This runs Acolite's Windows release (20260421.0, github acolite/acolite,
GPL-3.0) on the two L1C scenes of 05, cropped to the L2A scene box, with output_rhorc on, and
writes the 11 MariNeXt bands onto the grid of the L2A scenes, so every later step compares like
with like.

Acolite (and netCDF4, only if an output must be rebuilt) sit in E:\\SDG AI Lab\\Guinea\\marinext_test, outside the GPU environment:

    "E:\\SDG AI Lab\\Guinea\\national_scan_2026\\venv_gpu\\Scripts\\python" 7_published_detector/06_acolite_rhorc.py
"""
import os
import subprocess
import sys
from pathlib import Path

os.environ.pop("PROJ_LIB", None)
os.environ.pop("GDAL_DATA", None)

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject, transform_bounds

MX = Path(r"E:\SDG AI Lab\Guinea\marinext_test")

OUT = Path(__file__).resolve().parent / "outputs"
L1C = MX / "l1c"
WORK = MX / "acolite_out"
ACOLITE = MX / "acolite_bin" / "acolite_py_win" / "dist" / "acolite" / "acolite.exe"  # release 20260421.0
SCENES = {"2024-01-11": "S2B_MSIL1C_20240111T111329_N0510_R137_T28PFR_20240115T143406.SAFE",
          "2024-01-26": "S2A_MSIL1C_20240126T111331_N0510_R137_T28PFR_20240126T145153.SAFE"}
SKIP = ((930, 960), (1360, 1390))  # water vapour (B9) and cirrus (B10) bands, which MariNeXt does not use


def run_acolite(date, safe):
    out = WORK / date
    out.mkdir(parents=True, exist_ok=True)
    done = list(out.glob("*_L2R.nc"))
    if done:
        return done[0]
    with rasterio.open(OUT / f"conakry_{date}.tif") as s:
        w, so, e, n = transform_bounds(s.crs, "EPSG:4326", *s.bounds)
    settings = out / "settings.txt"
    settings.write_text("\n".join([
        f"inputfile={L1C / safe}", f"output={out}", f"limit={so:.4f},{w:.4f},{n:.4f},{e:.4f}",
        "output_rhorc=True", "s2_target_res=10", "l2w_parameters=None", "rgb_rhot=False", "rgb_rhos=False",
        "map_l2w=False", "l1r_delete_netcdf=False"]), encoding="utf-8")
    r = subprocess.run([str(ACOLITE), "--cli", f"--settings={settings}"], cwd=str(ACOLITE.parent),
                       capture_output=True, text=True)
    (out / "acolite.log").write_text(r.stdout + r.stderr, encoding="utf-8")
    done = list(out.glob("*_L2R.nc"))
    if r.returncode or not done:
        raise SystemExit(f"Acolite failed for {date}; see {out / 'acolite.log'}")
    return done[0]


def to_grid(nc_path, date):
    target = OUT / f"conakry_{date}_rhorc.tif"
    if target.exists():
        return target
    sys.path.insert(0, str(MX / "libs_acolite"))
    try:
        import netCDF4
    except ImportError:
        raise SystemExit('netCDF4 missing: python -m pip install --target "' + str(MX / "libs_acolite") +
                         '" --no-deps netCDF4 cftime')
    nc = netCDF4.Dataset(nc_path)
    names = sorted((v for v in nc.variables if v.startswith("rhorc_")), key=lambda v: float(v.split("_")[1]))
    names = [v for v in names if not any(lo < float(v.split("_")[1]) < hi for lo, hi in SKIP)]
    if len(names) != 11:
        raise SystemExit(f"expected 11 rhorc bands, got {names}")
    g = nc.__dict__
    crs = CRS.from_string(g["proj4_string"])
    px = float(g["pixel_size"][0])
    transform = from_origin(float(g["xrange"][0]), float(g["yrange"][0]), px, px)
    with rasterio.open(OUT / f"conakry_{date}.tif") as s:
        prof = {**s.profile, "count": 11, "dtype": "uint16", "nodata": 0}
        shape, dst_t, dst_crs = (s.height, s.width), s.transform, s.crs
    stack = np.zeros((11,) + shape, "uint16")
    for i, v in enumerate(names):
        a = np.ma.filled(nc.variables[v][:].astype("float32"), np.nan)
        dst = np.full(shape, np.nan, "float32")
        reproject(a, dst, src_transform=transform, src_crs=crs, dst_transform=dst_t, dst_crs=dst_crs,
                  src_nodata=np.nan, dst_nodata=np.nan, resampling=Resampling.bilinear)
        stack[i] = np.where(np.isfinite(dst), np.clip(dst * 1e4, 1, 65535), 0).astype("uint16")
    nc.close()
    with rasterio.open(target, "w", **prof) as d:
        d.write(stack)
        d.descriptions = tuple(names)
    print(date, "bands", names, flush=True)
    return target


def main():
    for date, safe in SCENES.items():
        if not (L1C / safe).exists():
            raise SystemExit(f"run 05 first: {safe} missing")
        nc = run_acolite(date, safe)
        to_grid(nc, date)


if __name__ == "__main__":
    main()
