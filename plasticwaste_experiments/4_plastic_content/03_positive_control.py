"""
Can the same EnMAP test see plastic where plastic certainly fills the pixel?

A negative result needs this: if the test cannot find plastic where it is obvious,
"no signal" in Guinea says nothing. Two places, both far larger than a pixel:

  almeria   the plastic greenhouses of Campo de Dalías, Spain
  ghazipur  the Ghazipur landfill, Delhi, about 28 ha inside a 69 ha box

Check the boxes below on a map before downloading. Needs the same DLR login as script
01. Each scene is about 0.5 GB.

    python 4_plastic_content/03_positive_control.py --search almeria
    python 4_plastic_content/03_positive_control.py --download almeria best
    python 4_plastic_content/03_positive_control.py --score almeria
"""
import argparse
import json
import sys
import urllib.parse
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from common import ROOT, TOOL  # noqa: E402

from plasticwaste_tool.characterisation_tool import enmap_download as ED  # noqa: E402
from plasticwaste_tool.characterisation_tool import enmap_spectra as ES  # noqa: E402
from plasticwaste_tool.characterisation_tool import hyperspectral_coverage as HC  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_spectra import FEATURES, depth  # noqa: E402

SCENES = Path(r"D:\guinea_data\enmap_control")
OUT = ROOT / "4_plastic_content" / "outputs"
# west, south, east, north. The control is nearby ground without plastic cover.
SITES = {
    "almeria": {"target": (-2.85, 36.72, -2.70, 36.77), "control": (-2.85, 36.84, -2.70, 36.90)},
    # core: the landfill's share of the box. Its spectrum is the strongest pixels at 1730 nm.
    "ghazipur": {"target": (77.323, 28.620, 77.331, 28.628), "control": (77.35, 28.60, 77.37, 28.62), "core": .4},
    # Conakry's landfill, about 20 ha, boxes drawn on imagery: its active southern face and
    # its older northern part. The Guinean scenes carry along-track stripes at 1730 nm, so
    # the control is city along the same stripes, north and south of the landfill.
    "dar_es_salam": {"target": (-13.6435, 9.5718, -13.6406, 9.5754), "scenes": TOOL / "runs" / "enmap_scenes",
                     "control": [(-13.64105, 9.5840, -13.63815, 9.5876), (-13.64595, 9.5596, -13.64305, 9.5632)]},
    "dar_es_salam_north": {"target": (-13.6454, 9.5756, -13.6411, 9.5803), "scenes": TOOL / "runs" / "enmap_scenes",
                           "control": [(-13.6430, 9.5878, -13.6387, 9.5925), (-13.6479, 9.5634, -13.6436, 9.5681)]},
}


def search(site):
    t = SITES[site]["target"]
    # GET with no Accept header: DLR's catalogue answers 500 otherwise (see check_dlr_stac).
    query = urllib.parse.urlencode({"collections": "ENMAP_HSI_L2A", "bbox": ",".join(map(str, t)), "limit": 100})
    found, _ = HC._get_json(f"{HC.DLR_STAC_SEARCH}?{query}", headers={})
    items = found.get("features", [])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"03_{site}_items.json").write_text(json.dumps(items), encoding="utf-8")
    rows = [{"id": i["id"], "date": i["properties"].get("datetime", "")[:10],
             "cloud": i["properties"].get("eo:cloud_cover")} for i in items]
    d = pd.DataFrame(rows)
    print(d.sort_values("cloud").head(15).to_string() if len(d) else "No scene found")


def download(site, scene_id):
    import requests
    items = json.loads((OUT / f"03_{site}_items.json").read_text(encoding="utf-8"))
    if scene_id == "best":
        # Daylight only: EnMAP also records night passes, which hold no reflectance.
        lon = (SITES[site]["target"][0] + SITES[site]["target"][2]) / 2

        def daylight(i):
            hour = int(i["properties"]["datetime"][11:13]) + int(i["properties"]["datetime"][14:16]) / 60
            return 8 <= (hour + lon / 15) % 24 <= 16

        items = sorted(filter(daylight, items), key=lambda i: i["properties"]["datetime"], reverse=True)
        item = min(items, key=lambda i: i["properties"].get("eo:cloud_cover", 100))
        scene_id = item["id"]
        print("lowest cloud:", scene_id, item["properties"].get("eo:cloud_cover"))
    else:
        item = next(i for i in items if i["id"] == scene_id)
    urls = {k: v["href"] for k, v in item["assets"].items() if k in ("image", "metadata")}
    user, pwd = ED.load_credentials()
    session = requests.Session()
    session.headers["User-Agent"] = "GEDH-DPWMS/1.0"
    if not ED.cas_login(session, next(iter(urls.values())), user, pwd):
        raise SystemExit("login failed")
    for name, url in urls.items():
        ext = Path(url.split("?")[0]).suffix or ".bin"
        # The first file after login can come back as the login page, so a second pass fetches it.
        if not ED.download(session, url, SCENES / site / scene_id / f"{name}{ext}"):
            ED.download(session, url, SCENES / site / scene_id / f"{name}{ext}")


def read_box(src, box):
    from rasterio.warp import transform_bounds
    from rasterio.errors import WindowError
    from rasterio.windows import Window, from_bounds
    w = from_bounds(*transform_bounds("EPSG:4326", src.crs, *box), transform=src.transform)
    try:
        w = w.intersection(Window(0, 0, src.width, src.height))
    except WindowError:  # the scene misses this box
        return np.empty((0, src.count), "float32")
    a = src.read(window=w.round_offsets().round_lengths()).astype("float32")
    s = a.reshape(a.shape[0], -1).T
    s = s[~np.isin(s, ES.NODATA_SENTINELS).all(axis=1)] * ES.REFL_SCALE
    return s[((s > -0.05) & (s < 1.5)).mean(axis=1) > 0.9]


def score(site):
    import rasterio
    from sklearn.metrics import roc_auc_score
    rows = []
    control_boxes = SITES[site]["control"]
    control_boxes = control_boxes if isinstance(control_boxes, list) else [control_boxes]
    for scene in sorted(SITES[site].get("scenes", SCENES / site).iterdir()):
        if not any(scene.glob("metadata*")):
            print(scene.name[:50], "no metadata, skipped: run the download again")
            continue
        wl = ES.read_wavelengths(next(scene.glob("metadata*")))
        with rasterio.open(next(scene.glob("image*"))) as src:
            target = read_box(src, SITES[site]["target"])
            control = np.vstack([read_box(src, box) for box in control_boxes])
        print(scene.name[:50], "pixels", len(target), len(control))
        if len(target) == 0 or len(control) == 0:
            continue
        # The site's own spectrum, for the mixing limit: real dips at EnMAP's scale.
        pd.DataFrame({"wavelength_nm": wl, "reflectance": np.median(target, axis=0)}).to_csv(
            OUT / f"03_{site}_spectrum.csv", index=False)
        if "core" in SITES[site]:
            strength = depth(target, wl, 1730)
            core = target[strength >= np.nanquantile(strength, 1 - SITES[site]["core"])]
            pd.DataFrame({"wavelength_nm": wl, "reflectance": np.median(core, axis=0)}).to_csv(
                OUT / f"03_{site}-core_spectrum.csv", index=False)
            print("core depths", {f: round(float(np.nanmedian(depth(core, wl, f))), 4) for f in FEATURES})
        for feature in FEATURES:
            a, b = depth(target, wl, feature), depth(control, wl, feature)
            a, b = a[np.isfinite(a)], b[np.isfinite(b)]
            y = np.r_[np.ones(len(a)), np.zeros(len(b))]
            rows.append({"site": site, "scene": scene.name[:40], "feature_nm": feature,
                         "plastic_median_depth": float(np.median(a)), "control_median_depth": float(np.median(b)),
                         "auc": roc_auc_score(y, np.r_[a, b]), "plastic_px": len(a), "control_px": len(b)})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / f"03_positive_control_{site}.csv", index=False)
    print(t.round(4).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--search", choices=SITES)
    ap.add_argument("--download", nargs=2, metavar=("SITE", "SCENE_ID"))
    ap.add_argument("--score", choices=SITES)
    a = ap.parse_args()
    if a.search:
        search(a.search)
    elif a.download:
        download(*a.download)
    elif a.score:
        score(a.score)
    else:
        ap.print_help()
