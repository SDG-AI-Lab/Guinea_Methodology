"""
Can the same EnMAP test see plastic where plastic certainly fills the pixel?

A negative result needs this: if the test cannot find plastic where it is obvious,
"no signal" in Guinea says nothing. Two places, both far larger than a pixel:

  almeria   the plastic greenhouses of Campo de Dalías, Spain
  ghazipur  the Ghazipur landfill, Delhi, where Dutta et al. (2022) used PRISMA

Check the boxes below on a map before downloading. Needs the same DLR login as script
01. Each scene is about 0.5 GB.

    python 4_plastic_content/03_positive_control.py --search almeria
    python 4_plastic_content/03_positive_control.py --download almeria SCENE_ID
    python 4_plastic_content/03_positive_control.py --score almeria
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from common import ROOT  # noqa: E402

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
    "ghazipur": {"target": (77.323, 28.620, 77.331, 28.628), "control": (77.35, 28.60, 77.37, 28.62)},
}


def search(site):
    t = SITES[site]["target"]
    found = HC._post_json(HC.DLR_STAC_SEARCH, {"collections": ["ENMAP_HSI_L2A"], "bbox": list(t), "limit": 100})
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
    item = next(i for i in items if i["id"] == scene_id)
    urls = {k: v["href"] for k, v in item["assets"].items() if k in ("image", "metadata")}
    user, pwd = ED.load_credentials()
    session = requests.Session()
    session.headers["User-Agent"] = "GEDH-DPWMS/1.0"
    if not ED.cas_login(session, next(iter(urls.values())), user, pwd):
        raise SystemExit("login failed")
    for name, url in urls.items():
        ext = Path(url.split("?")[0]).suffix or ".bin"
        ED.download(session, url, SCENES / site / scene_id / f"{name}{ext}")


def read_box(src, box):
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds
    w = from_bounds(*transform_bounds("EPSG:4326", src.crs, *box), transform=src.transform)
    a = src.read(window=w.round_offsets().round_lengths()).astype("float32")
    s = a.reshape(a.shape[0], -1).T
    s = s[~np.isin(s, ES.NODATA_SENTINELS).all(axis=1)] * ES.REFL_SCALE
    return s[((s > -0.05) & (s < 1.5)).mean(axis=1) > 0.9]


def score(site):
    import rasterio
    from sklearn.metrics import roc_auc_score
    rows = []
    for scene in sorted((SCENES / site).iterdir()):
        wl = ES.read_wavelengths(next(scene.glob("metadata*")))
        with rasterio.open(next(scene.glob("image*"))) as src:
            target, control = read_box(src, SITES[site]["target"]), read_box(src, SITES[site]["control"])
        print(scene.name[:50], "pixels", len(target), len(control))
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
