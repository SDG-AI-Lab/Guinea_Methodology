"""Download GLAD's global land cover and land use 2000-2020 (Potapov et al. 2022, v2) for Guinea.

The four 10-degree tiles that hold Guinea, for each year asked, into STORE/downloads/glclu/<year>/.
Public files on Google Cloud Storage; no account needed.
Usage: python download_glclu.py 2010 2015 2020
"""
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grid100 as G                                             # noqa: E402

BASE = "https://storage.googleapis.com/earthenginepartners-hansen/GLCLU2000-2020/v2"
TILES = ["20N_020W", "20N_010W", "10N_020W", "10N_010W"]       # top-left corners; Guinea is 7-13 N, 8-15 W


def download(year):
    folder = G.STORE / "downloads" / "glclu" / str(year)
    folder.mkdir(parents=True, exist_ok=True)
    for tile in TILES:
        path = folder / f"{tile}.tif"
        if path.exists():
            continue
        part = path.with_suffix(".part")
        with requests.get(f"{BASE}/{year}/{tile}.tif", stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(part, "wb") as f:
                for chunk in r.iter_content(1 << 22):
                    f.write(chunk)
        part.rename(path)
        print("downloaded", year, tile, round(path.stat().st_size / 1e6), "MB", flush=True)


if __name__ == "__main__":
    for y in sys.argv[1:]:
        download(int(y))
