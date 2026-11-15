"""
If plastic were there, would this test have seen it?

"No signal" has two meanings: the plastic is absent, or it is too little to see. This
separates them with a standard check, injection and recovery. It adds a measured
plastic spectrum to real Guinean ground spectra, in the amount each site's size would
give, then runs the same comparison as the real test. The real test compared dump
pixels with the ring of pixels around them.

Each spectrum from script 03 is tried: greenhouse film at Almeria, and real waste
surfaces at Ghazipur and at Conakry's Dar es Salam landfill. Film is far stronger than
waste, so the waste surfaces are the fairer reference.

For each share of plastic on the dump's surface:

  expected AUC   what the test would show if the dumps held that much plastic
  observed AUC   what it did show

If the expected AUC is clearly above the observed one, the dumps do not hold that much
exposed plastic. If the two overlap, this much plastic is too little to see, so EnMAP
cannot say either way. Both are firm statements. Intervals are 90 %, resampling sites.

"Clearly above" means the expected interval lies wholly above the observed one. Intervals
that do not overlap are a strict test of a difference (Schenker and Gentleman 2001).

Spectra are the first test's 30 sites plus the sites of script 01. The plastic spectrum
is the one measured at the positive control. Run script 03 first.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from common import ROOT, TOOL  # noqa: E402

from scipy.stats import rankdata  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_spectra import FEATURES, depth  # noqa: E402

OUT = ROOT / "4_plastic_content" / "outputs"
SURFACE_SHARES = (.05, .10, .20, .40, .60, .80, 1.0)  # of the dump's surface, not of its mass
PIXEL_M = 30.0
N_BOOT = 1000
LARGE_M = 21.0  # the width script 02 found a site needs at 1730 nm
SEED = 11


def load():
    parts = []
    for folder in (TOOL / "runs" / "enmap_spectra", OUT / "more_enmap"):
        d = np.load(folder / "spectra.npz", allow_pickle=True)
        parts.append((d["spectra"], d["wavelengths"], d["kind"], d["point_id"], d["tile_id"]))
    wl = parts[0][1]
    return (np.vstack([p[0] for p in parts]), wl, np.concatenate([p[2] for p in parts]),
            np.concatenate([p[3] for p in parts]), np.concatenate([p[4] for p in parts]))


def auc(pos, neg):
    r = rankdata(np.r_[pos, neg])
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def boot_auc(pos, pos_site, neg, neg_site, rng):
    sites = np.unique(np.r_[pos_site, neg_site])
    by_pos = {s: pos[pos_site == s] for s in sites}
    by_neg = {s: neg[neg_site == s] for s in sites}
    out = []
    for _ in range(N_BOOT):
        pick = rng.choice(sites, len(sites))
        a, b = np.concatenate([by_pos[s] for s in pick]), np.concatenate([by_neg[s] for s in pick])
        if len(a) and len(b):
            out.append(auc(a, b))
    return float(np.quantile(out, .05)), float(np.quantile(out, .95))


def main():
    spectra, wl, kind, point, tile = load()
    good = np.isfinite(spectra).all(axis=1)
    spectra, kind, point, tile = spectra[good], kind[good], point[good], tile[good]
    sizes = pd.read_csv(TOOL / "runs" / "dumpsite_extent" / "extent_annotated_usable.csv").set_index("point_id").area_equiv_m
    base_points = np.array([p.split("_ring")[0] for p in point])
    width = pd.Series(base_points).map(sizes).fillna(sizes.median()).to_numpy()
    fill = np.minimum((width / PIXEL_M) ** 2, 1.0)

    dump, ring = kind == "dump", kind == "ring_control"
    print(f"Spectra: {dump.sum()} dump, {ring.sum()} ring, sites {len(np.unique(tile[dump]))}")
    rows = []
    rng = np.random.default_rng(SEED)
    for source in sorted(OUT.glob("03_*_spectrum.csv")):
        s = pd.read_csv(source)
        plastic = np.interp(wl, s.wavelength_nm, s.reflectance)
        name = source.stem[3:-len("_spectrum")]
        for feature in FEATURES:
            clean = depth(spectra, wl, feature)
            keep = np.isfinite(clean)
            obs = auc(clean[dump & keep], clean[ring & keep])
            obs_low, obs_high = boot_auc(clean[dump & keep], tile[dump & keep], clean[ring & keep], tile[ring & keep], rng)
            for share in SURFACE_SHARES:
                f = (fill * share)[ring, None]
                mixed = depth((1 - f) * spectra[ring] + f * plastic, wl, feature)
                ok = np.isfinite(mixed)
                exp = auc(mixed[ok], clean[ring & keep])
                low, high = boot_auc(mixed[ok], tile[ring][ok], clean[ring & keep], tile[ring & keep], rng)
                rows.append({"plastic_spectrum": name, "feature_nm": feature, "plastic_on_surface": share,
                             "median_share_of_pixel": float(np.median(fill[ring]) * share),
                             "expected_auc": exp, "expected_low": low, "expected_high": high,
                             "observed_auc": obs, "observed_low": obs_low, "observed_high": obs_high,
                             "would_have_been_seen": low > obs_high})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "04_injection_recovery.csv", index=False)
    print(t.drop(columns=["expected_low", "expected_high"]).round(3).to_string())

    # The sites wide enough to reach that share of a pixel, on the real data.
    big = width >= LARGE_M
    largest = []
    for feature in FEATURES:
        clean = depth(spectra, wl, feature)
        keep = np.isfinite(clean) & big
        a, b = clean[dump & keep], clean[ring & keep]
        low, high = boot_auc(a, tile[dump & keep], b, tile[ring & keep], rng)
        largest.append({"feature_nm": feature, "dump_spectra": len(a), "sites": len(np.unique(tile[dump & keep])),
                        "observed_auc": auc(a, b), "observed_low": low, "observed_high": high})
    largest = pd.DataFrame(largest)
    largest.to_csv(OUT / "04_largest_sites.csv", index=False)
    print(largest.round(3).to_string())


if __name__ == "__main__":
    main()
