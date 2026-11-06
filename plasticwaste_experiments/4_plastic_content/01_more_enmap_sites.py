"""
Repeat the EnMAP test on the sites the first run did not reach.

The first test read 6 scenes over 30 distinct sites. 27 scenes hold a labeled point,
so 21 are left. This script lists them, downloads the ones you choose and reruns the
same extraction and separability code on them. Nothing in the tool's own `runs`
folder is touched.

Needs DLR_EOC_USERNAME and DLR_EOC_PASSWORD in `Land Use-Deforestation/.env`.
Each scene is about 0.5 GB and goes to D:\\guinea_data\\enmap_more.

    python 4_plastic_content/01_more_enmap_sites.py --list
    python 4_plastic_content/01_more_enmap_sites.py --download 8
    python 4_plastic_content/01_more_enmap_sites.py --analyse
"""
import argparse
import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from common import ROOT, TOOL  # noqa: E402

from plasticwaste_tool.characterisation_tool import enmap_download as ED  # noqa: E402
from plasticwaste_tool.characterisation_tool import enmap_spectra as ES  # noqa: E402
from plasticwaste_tool.characterisation_tool import separability as SE  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_spectra import FEATURES, depth  # noqa: E402

SCENES = Path(r"D:\guinea_data\enmap_more")
OUT = ROOT / "4_plastic_content" / "outputs" / "more_enmap"


def candidates():
    """Scenes not read yet, those adding the most new sites first."""
    items = json.loads(ED.COVERAGE_JSON.read_text(encoding="utf-8"))["missions"]["EnMAP"]["items"]
    done = {p.name for p in ED.OUT_DIR.iterdir() if p.is_dir()}
    seen = set(pd.read_csv(TOOL / "runs" / "enmap_spectra" / "points.csv").query("kind == 'dump'").tile_id)
    rows = []
    for it in items:
        if it["id"] in done:
            continue
        points = ast.literal_eval(it["dump_points"]) if isinstance(it["dump_points"], str) else it["dump_points"]
        tiles = {p.split("@")[0].rsplit("_p", 1)[0] for p in points}
        rows.append({"id": it["id"], "date": it["datetime"][:10], "cloud": float(it["cloud_cover"]),
                     "points": len(points), "new_sites": len(tiles - seen), "swir": it.get("swir_status")})
    return pd.DataFrame(rows).sort_values(["new_sites", "cloud"], ascending=[False, True]).reset_index(drop=True)


def download(n):
    picked = candidates().query("swir == 'on' and cloud <= 20 and new_sites > 0").head(n)
    print(picked.to_string())
    ED.OUT_DIR = SCENES
    match = ",".join(picked.id)
    sys.argv = ["enmap_download", "--match", match]
    return ED.main()


def analyse():
    """The tool's own extraction and separability code, pointed at the new scenes."""
    OUT.mkdir(parents=True, exist_ok=True)
    ES.SCENE_GLOB = str(SCENES / "*")
    ES.OUT_DIR = OUT
    SE.RUN = OUT
    ES.main()
    SE.main()
    d = np.load(OUT / "spectra.npz", allow_pickle=True)
    pts = pd.read_csv(OUT / "points.csv")
    from sklearn.metrics import roc_auc_score
    rows = []
    use = pts.kind.isin(["dump", "ring_control"]).to_numpy()
    y = (pts.kind[use] == "dump").astype(int).to_numpy()
    for feature in FEATURES:
        x = depth(d["spectra"][use], d["wavelengths"], feature)
        ok = np.isfinite(x)
        rows.append({"feature_nm": feature, "auc_dump_vs_ring": roc_auc_score(y[ok], x[ok]),
                     "dump_median_depth": float(np.median(x[ok & (y == 1)])),
                     "ring_median_depth": float(np.median(x[ok & (y == 0)])),
                     "sites": int(pts[pts.kind == "dump"].tile_id.nunique())})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "ch_depth_summary.csv", index=False)
    print(t.round(4).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--download", type=int)
    ap.add_argument("--analyse", action="store_true")
    a = ap.parse_args()
    if a.list:
        print(candidates().to_string())
    elif a.download:
        download(a.download)
    elif a.analyse:
        analyse()
    else:
        ap.print_help()
