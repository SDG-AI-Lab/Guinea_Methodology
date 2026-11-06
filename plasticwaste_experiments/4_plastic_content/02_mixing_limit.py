"""
How much plastic must a pixel hold before EnMAP could show it?

Takes real control spectra from Guinea, mixes in a plastic spectrum at growing shares
of the pixel, and finds the share where the C-H dip rises above what Guinean ground
shows anyway. That turns "no signal" into a size: how wide a dump would have to be.

A mixed pixel is (1 - f) x ground + f x plastic (linear mixing). "Detectable" means
the median mixed spectrum has a dip deeper than 95 % of the unmixed controls.

The plastic spectrum: put a measured one in `4_plastic_content/plastic_spectrum.csv`,
columns wavelength_nm and reflectance, for example a polyethylene from the USGS
Spectral Library. Without it the script uses a simple bright spectrum with two
Gaussian dips. Those depths are an assumption, so three strengths are run. Use the
result without the file only as a rough guide.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from common import ROOT, TOOL  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_spectra import FEATURES, depth  # noqa: E402

OUT = ROOT / "4_plastic_content" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
MEASURED = ROOT / "4_plastic_content" / "plastic_spectrum.csv"
SHARES = np.array([.005, .01, .02, .03, .05, .075, .1, .15, .2, .3, .4, .5, .75, 1.0])
PIXEL_M = 30.0
PLASTIC_IN_WASTE = 0.20  # Gbilimou et al. 2022, through Bah 2026
ASSUMED = {"weak": (.05, .12), "middle": (.10, .25), "strong": (.15, .40)}  # dip at 1215 and 1730 nm


def assumed_plastic(wl, dip_1215, dip_1730):
    g = lambda c, w: np.exp(-0.5 * ((wl - c) / w) ** 2)  # noqa: E731
    return 0.6 * (1 - dip_1215 * g(1215, 15) - dip_1730 * g(1730, 20))


def plastics(wl):
    if MEASURED.exists():
        d = pd.read_csv(MEASURED).sort_values("wavelength_nm")
        return {"measured": np.interp(wl, d.wavelength_nm, d.reflectance)}
    print("plastic_spectrum.csv not found: using assumed dips, as a rough guide only")
    return {name: assumed_plastic(wl, *dips) for name, dips in ASSUMED.items()}


def main():
    d = np.load(TOOL / "runs" / "enmap_spectra" / "spectra.npz", allow_pickle=True)
    spectra, wl, kind = d["spectra"], d["wavelengths"], d["kind"]
    ground = spectra[np.isin(kind, ["ring_control", "labelled_negative"])]
    sizes = pd.read_csv(TOOL / "runs" / "dumpsite_extent" / "extent_annotated_usable.csv").area_equiv_m.dropna()
    fill = np.minimum((sizes.to_numpy() / PIXEL_M) ** 2, 1.0)
    expected = float(np.median(fill) * PLASTIC_IN_WASTE)
    print(f"Median site: {sizes.median():.1f} m, plastic share of the pixel {expected:.3f} if all plastic lies on top")

    rows, curve = [], []
    for name, plastic in plastics(wl).items():
        for feature in FEATURES:
            limit = np.nanquantile(depth(ground, wl, feature), .95)
            found = np.nan
            for f in SHARES:
                x = depth((1 - f) * ground + f * plastic, wl, feature)
                med, rate = float(np.nanmedian(x)), float(np.nanmean(x > limit))
                curve.append({"plastic": name, "feature_nm": feature, "share_of_pixel": f,
                              "median_depth": med, "share_above_limit": rate})
                if np.isnan(found) and med > limit:
                    found = float(f)
            width = PIXEL_M * np.sqrt(found / PLASTIC_IN_WASTE) if found == found else np.nan
            rows.append({"plastic": name, "feature_nm": feature, "ground_95pct_depth": float(limit),
                         "needed_share_of_pixel": found, "dump_width_needed_m": width,
                         "sites_reaching_it": int((fill * PLASTIC_IN_WASTE >= found).sum()) if found == found else 0,
                         "sites": int(len(fill))})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "02_mixing_limit.csv", index=False)
    pd.DataFrame(curve).to_csv(OUT / "02_mixing_curve.csv", index=False)
    print(t.round(3).to_string())


if __name__ == "__main__":
    main()
