"""
Does HISUI see Conakry's landfill more clearly than EnMAP?

HISUI (on the International Space Station) has 20 m pixels and covers 400-2500 nm, so
it samples both plastic features. Baruti et al. (2026) used it on eight African landfills,
one in Guinea-Bissau. Its scenes are free from Tellus, as Level-1G radiance.

Needs a free Tellus account (https://www.tellusxdp.com) and its API token, set in the
project's .env as TELLUS_API_TOKEN=... The token stays in .env.

    python 4_plastic_content/05_hisui_dar_es_salam.py --search
    python 4_plastic_content/05_hisui_dar_es_salam.py --download SCENE_ID
    python 4_plastic_content/05_hisui_dar_es_salam.py --inspect

The scoring step is written once a scene is on disk, because the file layout must be
seen first. It will use the same boxes and depths as script 03.
"""
import argparse
import json
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from common import PROJECT, ROOT  # noqa: E402

API = "https://www.tellusxdp.com/api/traveler/v1"
SCENES = Path(r"D:\guinea_data\hisui")
OUT = ROOT / "4_plastic_content" / "outputs"
# west, south, east, north: the landfill with about 1 km around it, for the control
BOX = (-13.66, 9.56, -13.62, 9.60)


def token():
    for line in (PROJECT / ".env").read_text(encoding="utf-8").splitlines():
        if line.startswith("TELLUS_API_TOKEN="):
            return line.split("=", 1)[1].strip().strip('"')
    raise SystemExit("Add TELLUS_API_TOKEN=... to the project's .env first")


def call(method, path, **kwargs):
    r = requests.request(method, API + path, headers={"Authorization": f"Bearer {token()}"},
                         allow_redirects=False, timeout=120, **kwargs)
    if r.is_redirect:
        return r.headers["Location"]
    if not r.ok:
        raise SystemExit(f"{r.status_code}: {r.text[:300]}")
    return r.json()


def hisui_datasets():
    found, url = [], "/datasets/"
    while url:
        page = call("get", url)
        found += [d for d in page["results"] if "HISUI" in json.dumps(d, ensure_ascii=False).upper()]
        url = page.get("next", "").replace(API, "") if page.get("next") else None
    return found


def search():
    sets = hisui_datasets()
    for d in sets:
        print("dataset", d["id"], d.get("name"))
    if not sets:
        raise SystemExit("No HISUI dataset visible to this account")
    w, s, e, n = BOX
    body = {"datasets": [d["id"] for d in sets],
            "intersects": {"type": "Polygon", "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]]}}
    scenes = call("post", "/data-search/", json=body).get("features", [])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "05_hisui_scenes.json").write_text(json.dumps(scenes, indent=1), encoding="utf-8")
    print(len(scenes), "scenes over Dar es Salam")
    for sc in scenes:
        p = sc.get("properties", {})
        print(sc["id"], p.get("tellus:dataset_id", ""), p.get("start_datetime", p.get("datetime", "")),
              "cloud", p.get("eo:cloud_cover", p.get("cloud_cover", "?")))


def download(scene_id):
    scenes = json.loads((OUT / "05_hisui_scenes.json").read_text(encoding="utf-8"))
    sc = next(x for x in scenes if x["id"] == scene_id)
    dataset = sc["properties"]["tellus:dataset_id"]
    folder = SCENES / scene_id
    folder.mkdir(parents=True, exist_ok=True)
    for f in call("get", f"/datasets/{dataset}/data/{scene_id}/files/")["results"]:
        target = folder / f["name"]
        if target.exists():
            print("have", f["name"])
            continue
        url = call("post", f"/datasets/{dataset}/data/{scene_id}/files/{f['id']}/download-url/")["download_url"]
        with requests.get(url, stream=True, timeout=600) as r:
            r.raise_for_status()
            with open(target.with_suffix(target.suffix + ".part"), "wb") as out:
                for chunk in r.iter_content(1 << 20):
                    out.write(chunk)
        target.with_suffix(target.suffix + ".part").replace(target)
        print("got", f["name"], round(target.stat().st_size / 1e6), "MB")


def inspect():
    """Print what the files hold, so the scoring step can be written."""
    import rasterio
    for path in sorted(SCENES.rglob("*")):
        if path.is_file():
            print(path.relative_to(SCENES), round(path.stat().st_size / 1e6, 1), "MB")
            try:
                with rasterio.open(path) as src:
                    print("   ", src.driver, src.count, "bands", src.dtypes[0], src.crs, src.shape)
                    print("   ", list(src.tags().items())[:8], src.subdatasets[:6])
            except Exception as e:  # not a raster: metadata, archive or HDF
                print("   ", type(e).__name__)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--search", action="store_true")
    ap.add_argument("--download", metavar="SCENE_ID")
    ap.add_argument("--inspect", action="store_true")
    a = ap.parse_args()
    if a.search:
        search()
    elif a.download:
        download(a.download)
    elif a.inspect:
        inspect()
    else:
        ap.print_help()
