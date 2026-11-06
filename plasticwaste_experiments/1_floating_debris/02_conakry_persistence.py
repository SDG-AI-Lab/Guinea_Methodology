"""
Do published patches behave like floating material?

A monthly median keeps what sits in a pixel on most clear dates. Material that
drifts cannot do that, so a patch in the median should be stationary: moored,
fixed, stranded or shallow bottom. Two tests on the cached Conakry imagery, all
on one grid, with the deployed detector unchanged:

  1. Recurrence: is a patch of one month at the same place in another month?
     Compared with the overlap expected if the same patches fell at random on
     the same water, which is what drifting material would give.
  2. Single dates: is a January patch detected, or at least above the FDI
     threshold, on the clear dates of 11 and 26 January?

Also counts what each candidate chain gives on each product, in pixels.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from common import ROOT, RASTERS, TOOL  # noqa: E402

from plasticwaste_tool.geo_env import fix_proj  # noqa: E402
fix_proj()
from plasticwaste_tool import data_loader as DL  # noqa: E402
from plasticwaste_tool.fdi_tool import debris_filter as DF  # noqa: E402
from plasticwaste_tool.fdi_tool import detection as D  # noqa: E402
from plasticwaste_tool.fdi_tool import spectral as SP  # noqa: E402

from scipy import ndimage as ndi  # noqa: E402

OUT = ROOT / "1_floating_debris" / "outputs"
ARCH = RASTERS.parent / "archive" / "fdi"
B10 = ["B3", "B4", "B6", "B8", "B8A", "B11", "B12", "FDI", "NDVI", "WATER"]
PRODUCTS = {
    "2023-11 median": (ARCH / "conakry_a410b95f_2023-11_de_africa_10b.tif", B10),
    "2023-12 median": (ARCH / "conakry_a410b95f_2023-12_de_africa_10b.tif", B10),
    "2024-01 median": (RASTERS / "conakry_a410b95f_2024-01-01_2024-02-01_composite.tif", None),
    "2024-01-11 single": (RASTERS / "conakry_a410b95f_2024-01-11_2024-01-12_composite.tif", None),
    "2024-01-26 single": (RASTERS / "conakry_a410b95f_2024-01-26_2024-01-27_composite.tif", None),
}


def detect(bands):
    fdi, ndvi = SP.compute_fdi(bands), SP.compute_ndvi(bands)
    water = bands["WATER"]
    rule, thr, _ = D.debris_binary_mask_de_africa(fdi, ndvi, water)
    kept, p, info = DF.apply(rule, bands)
    w = np.nan_to_num(water, nan=0.0) > 0
    t_f = float(np.nanmean(fdi[w]) + 3 * np.nanstd(fdi[w]))
    fdi_arm = w & (fdi > t_f)
    kept_f, _, _ = DF.apply(fdi_arm, bands)
    return dict(fdi=fdi, ndvi=ndvi, water=w, rule=rule, kept=kept, fdi_arm=fdi_arm,
                kept_f=kept_f, t_fdi=float(thr["fdi"]), t_ndvi=float(thr["ndvi"]))


def patches(mask, min_px=2):
    """8-connected patches of at least 2 pixels (500 m2 at 20 m), as published."""
    lab, n = ndi.label(mask, structure=np.ones((3, 3)))
    sizes = ndi.sum(mask, lab, np.arange(1, n + 1))
    keep = np.flatnonzero(sizes >= min_px) + 1
    return lab, keep


def main():
    res = {}
    for name, (path, names) in PRODUCTS.items():
        bands, _ = DL.read_composite(str(path), band_names=names) if names else DL.read_composite(str(path))
        res[name] = detect(bands)
        r = res[name]
        print(name, f"t_fdi {r['t_fdi']:.4f} t_ndvi {r['t_ndvi']:.4f} rule {r['rule'].sum()} "
              f"kept {r['kept'].sum()} fdi {r['fdi_arm'].sum()} fdi+rf {r['kept_f'].sum()}", flush=True)

    rows = [{"product": k, "t_fdi": v["t_fdi"], "t_ndvi": v["t_ndvi"], "rule_px": int(v["rule"].sum()),
             "published_px": int(v["kept"].sum()), "published_patches": len(patches(v["kept"])[1]),
             "fdi_arm_px": int(v["fdi_arm"].sum()), "fdi_arm_rf_px": int(v["kept_f"].sum()),
             "fdi_arm_rf_patches": len(patches(v["kept_f"])[1]),
             "water_px": int(v["water"].sum())} for k, v in res.items()]
    pd.DataFrame(rows).to_csv(OUT / "02_products.csv", index=False)

    # Recurrence between the two months that have patches.
    rng = np.random.default_rng(7)
    a, b = res["2023-11 median"], res["2024-01 median"]
    water = a["water"] & b["water"]
    grow = ndi.binary_dilation(b["kept"], structure=np.ones((3, 3)), iterations=1)
    la, ka = patches(a["kept"])

    hit = len(set(la[np.isin(la, ka) & grow]))
    wy, wx = np.nonzero(water)
    # Random placement over the same water: a patch hits if its pixel lands on `grow`.
    p_cell = (grow & water).sum() / water.sum()
    sizes = np.array([(la == k).sum() for k in ka])
    exp = float((1 - (1 - p_cell) ** sizes).sum())
    sim = [(rng.random(len(sizes)) < 1 - (1 - p_cell) ** sizes).sum() for _ in range(5000)]
    rec = {"nov_patches": int(len(ka)), "jan_patches": len(patches(b["kept"])[1]),
           "nov_patches_within_1px_of_jan": int(hit), "expected_if_random": exp,
           "enrichment": hit / exp if exp else None,
           "p_value": float((np.array(sim) >= hit).mean()),
           "share_of_water_near_jan": float(p_cell)}

    # Shore: distance to the nearest non-water pixel, 20 m cells.
    dist = ndi.distance_transform_edt(b["water"]) * 20.0
    lb, kb = patches(b["kept"])
    jan = []
    s1, s2 = res["2024-01-11 single"], res["2024-01-26 single"]
    for k in kb:
        m = lb == k
        jan.append({"patch": int(k), "n_px": int(m.sum()),
                    "lat_row": float(np.nonzero(m)[0].mean()), "lon_col": float(np.nonzero(m)[1].mean()),
                    "dist_shore_m": float(dist[m].min()),
                    "kept_11": bool(s1["kept"][m].any()), "kept_26": bool(s2["kept"][m].any()),
                    "rule_11": bool(s1["rule"][m].any()), "rule_26": bool(s2["rule"][m].any()),
                    "fdi_above_thr_11": bool(s1["fdi_arm"][m].any()), "fdi_above_thr_26": bool(s2["fdi_arm"][m].any()),
                    "in_nov": bool(ndi.binary_dilation(a["kept"])[m].any())})
    jan = pd.DataFrame(jan)
    jan.to_csv(OUT / "02_january_patches.csv", index=False)
    rec.update({
        "jan_kept_on_either_single_date": int((jan.kept_11 | jan.kept_26).sum()),
        "jan_rule_on_either_single_date": int((jan.rule_11 | jan.rule_26).sum()),
        "jan_fdi_above_thr_on_11": int(jan.fdi_above_thr_11.sum()),
        "jan_fdi_above_thr_on_26": int(jan.fdi_above_thr_26.sum()),
        "jan_fdi_above_thr_on_both": int((jan.fdi_above_thr_11 & jan.fdi_above_thr_26).sum()),
        "jan_within_100m_of_shore": int((jan.dist_shore_m <= 100).sum()),
        "jan_median_dist_shore_m": float(jan.dist_shore_m.median()),
    })
    # Random expectation for "FDI above threshold on a single date".
    rec["share_water_above_thr_11"] = float(s1["fdi_arm"].sum() / s1["water"].sum())
    rec["share_water_above_thr_26"] = float(s2["fdi_arm"].sum() / s2["water"].sum())
    (OUT / "02_persistence.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
