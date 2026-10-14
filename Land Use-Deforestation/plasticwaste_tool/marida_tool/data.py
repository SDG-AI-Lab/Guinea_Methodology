"""
Loading MARIDA: the only labelled marine-debris benchmark for Sentinel-2.

WHAT MARIDA IS

Kikaki, K., Kakogeorgiou, I., Mikeli, P., Raitsos, D.E. & Karantzalos, K.
(2022), MARIDA: A benchmark for Marine Debris detection from Sentinel-2
remote sensing data, PLoS ONE 17(1): e0262247.

It is a pixel-level annotation of 63 Sentinel-2 scenes across 2015-2021,
covering eleven sites where floating material was independently reported. The
part that matters for this module is not the debris class but the CONFUSER
classes: MARIDA separately labels Dense and Sparse Sargassum, Natural Organic
Material, Foam, Ships, Wakes, Turbid Water and Sediment-Laden Water.

WHY THAT MATTERS HERE

Guinea's FDI layer uses a mean+3sigma anomaly rule over water pixels. Such a
rule flags material that is spectrally unusual against the local water
background. It cannot, by construction, say WHAT the material is -- and the
things that float on tropical Atlantic water and look anomalous in the
near-infrared include Sargassum, sea foam, sediment plumes and boat wakes.

MARIDA is the only dataset that can answer the resulting question directly:
when the FDI rule fires, what is it actually firing on? That question is
answered in `baseline.py` by running our own FDI rule, unmodified, against
MARIDA's labels and tabulating what it hits.

THE LIMITATION TO STATE FIRST

MARIDA's scenes are globally distributed but sparse, and West Africa is
thinly represented. Any model trained on it and applied to Guinea is making a
geographic extrapolation. `site_coverage()` below reports exactly which sites
are present so that extrapolation is quantified in the methodology rather
than assumed away.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
TOOL = HERE.parent
DATA = TOOL / "data" / "marida"
ROOT = DATA / "MARIDA"

# Band order inside each patch GeoTIFF. MARIDA ships Sentinel-2 L1C
# reflectance with the two atmospheric bands (B09 water vapour, B10 cirrus)
# dropped, leaving 11. Verified against the archive by `describe()`.
BANDS = ["B01", "B02", "B03", "B04", "B05", "B06",
         "B07", "B08", "B8A", "B11", "B12"]

# MARIDA's class codes. Grouped here by the role they play in THIS analysis,
# which is not how the benchmark presents them: what we need to know is which
# classes an anomaly detector could plausibly mistake for debris.
CLASSES: Dict[int, str] = {
    1: "Marine Debris",
    2: "Dense Sargassum",
    3: "Sparse Sargassum",
    4: "Natural Organic Material",
    5: "Ship",
    6: "Clouds",
    7: "Marine Water",
    8: "Sediment-Laden Water",
    9: "Foam",
    10: "Turbid Water",
    11: "Shallow Water",
    12: "Waves",
    13: "Cloud Shadows",
    14: "Wakes",
    15: "Mixed Water",
}

TARGET = 1  # Marine Debris

# Floating material that is NOT plastic debris. These are the classes an FDI
# anomaly rule is at risk of firing on, and the reason a spectral index alone
# cannot support a claim about plastic.
FLOATING_CONFUSERS = [2, 3, 4, 5, 9, 14]

# Water backgrounds. An FDI threshold is estimated over these.
WATER_CLASSES = [7, 8, 10, 11, 15]

# Not water at all; excluded from the water-population statistics.
NON_WATER = [6, 12, 13]

CONFIDENCE = {1: "High", 2: "Moderate", 3: "Low"}


def ensure_extracted(zip_path: Optional[Path] = None) -> Path:
    """
    Unpack MARIDA.zip once. Returns the directory holding `patches/`.

    The archive's internal layout is not assumed: the directory containing
    `patches` is located after extraction rather than hard-coded, because a
    wrong guess would surface as an empty dataset and an apparently
    well-behaved model trained on nothing.
    """
    zip_path = Path(zip_path or (DATA / "MARIDA.zip"))
    marker = DATA / ".extracted"

    if not marker.exists():
        if not zip_path.exists():
            raise FileNotFoundError(
                f"{zip_path} not found. Download MARIDA (1.16 GB) from "
                f"https://doi.org/10.5281/zenodo.5151941 into {DATA}.")
        DATA.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(DATA)
        marker.write_text("extracted\n", encoding="utf-8")

    for cand in [DATA, ROOT, *sorted(DATA.glob("*"))]:
        if cand.is_dir() and (cand / "patches").is_dir():
            return cand
    raise RuntimeError(
        f"no 'patches/' directory found under {DATA} after extraction; "
        f"the archive layout has changed -- inspect it before proceeding.")


def load_split(name: str, root: Optional[Path] = None) -> List[str]:
    """
    Patch ids for one of MARIDA's official splits ('train', 'val', 'test').

    The published split is used unmodified. Re-splitting randomly would leak
    between train and test, because patches from one scene share atmosphere,
    sun geometry and often the same physical debris field -- a random pixel
    split on this dataset reports accuracy that does not survive contact with
    a new scene, which is precisely the failure this module exists to detect.
    """
    root = Path(root or ensure_extracted())
    for cand in (root / "splits" / f"{name}_X.txt", root / "splits" / f"{name}.txt"):
        if cand.exists():
            return [ln.strip() for ln in
                    cand.read_text(encoding="utf-8").splitlines() if ln.strip()]
    avail = sorted(p.name for p in (root / "splits").glob("*")) \
        if (root / "splits").is_dir() else []
    raise FileNotFoundError(
        f"split {name!r} not found under {root / 'splits'}; available: {avail}")


def normalise_id(pid: str) -> str:
    """
    Split files list ids WITHOUT the 'S2_' prefix the patch files carry:
    '1-12-19_48MYU_0' in splits/train_X.txt against S2_1-12-19_48MYU_0.tif on
    disk. Normalising here rather than at each call site means a future
    layout change has one place to be fixed.
    """
    pid = pid.strip()
    return pid if pid.startswith("S2_") else f"S2_{pid}"


def _patch_paths(root: Path, pid: str) -> Tuple[Path, Path, Path]:
    """
    Resolve one patch id to its (bands, classes, confidence) files.

    Ids look like 'S2_1-12-19_48MYU_0'; the scene directory is the id with the
    trailing patch number removed.
    """
    pid = normalise_id(pid)
    scene = pid.rsplit("_", 1)[0]
    d = root / "patches" / scene
    return d / f"{pid}.tif", d / f"{pid}_cl.tif", d / f"{pid}_conf.tif"


def read_patch(pid: str, root: Optional[Path] = None):
    """One patch as (bands[11,H,W], classes[H,W], confidence[H,W])."""
    import rasterio

    root = Path(root or ensure_extracted())
    b_path, c_path, f_path = _patch_paths(root, pid)
    with rasterio.open(b_path) as s:
        bands = s.read().astype("float32")
    with rasterio.open(c_path) as s:
        cl = s.read(1)
    conf = None
    if f_path.exists():
        with rasterio.open(f_path) as s:
            conf = s.read(1)
    return bands, cl, conf


def load_pixels(split: str, root: Optional[Path] = None,
                min_confidence: Optional[int] = None,
                drop_classes: Optional[List[int]] = None):
    """
    Every labelled pixel in a split, as a table.

    Returns (X[n, 11] reflectance, y[n] class code, pid[n] patch id).

    Pixels are the unit of analysis rather than patches because MARIDA's
    annotations are sparse polygons inside each patch -- most pixels carry no
    label at all, and a patch-level model would be trained mostly on unlabelled
    background. `pid` is retained so that evaluation can be grouped by scene,
    which is the only grouping that reflects how the model would be deployed.
    """
    root = Path(root or ensure_extracted())
    drop = set(drop_classes or [])
    Xs, ys, ps = [], [], []

    listed = load_split(split, root)
    missing, empty = [], []
    for pid in listed:
        try:
            bands, cl, conf = read_patch(pid, root)
        except Exception:
            missing.append(pid)           # listed in the split but not on disk
            continue
        m = cl > 0
        if min_confidence is not None and conf is not None:
            m &= (conf > 0) & (conf <= min_confidence)
        if drop:
            m &= ~np.isin(cl, list(drop))
        if not m.any():
            empty.append(pid)
            continue
        Xs.append(bands[:, m].T)
        ys.append(cl[m])
        ps.append(np.full(int(m.sum()), pid, dtype=object))

    if not Xs:
        raise RuntimeError(
            f"no labelled pixels found in split {split!r}: "
            f"{len(missing)} of {len(listed)} patches could not be read and "
            f"{len(empty)} held no labels. Check the id convention -- split "
            f"files omit the 'S2_' prefix that the patch files carry.")
    if missing:
        import warnings
        warnings.warn(
            f"{len(missing)} of {len(listed)} patches in split {split!r} "
            f"could not be read, e.g. {missing[:3]}", RuntimeWarning)
    X = np.concatenate(Xs).astype("float32")
    y = np.concatenate(ys).astype("int16")
    pid = np.concatenate(ps)

    # MARIDA ships reflectance already scaled to [0, 1]; a stray fill value
    # would otherwise propagate silently into every index.
    ok = np.isfinite(X).all(axis=1)
    return X[ok], y[ok], pid[ok]


def describe(root: Optional[Path] = None) -> Dict:
    """Structure and class balance -- run this before trusting anything else."""
    root = Path(root or ensure_extracted())
    out: Dict = {"root": str(root)}
    for split in ("train", "val", "test"):
        try:
            ids = load_split(split, root)
        except FileNotFoundError:
            continue
        out[f"n_patches_{split}"] = len(ids)
    scenes = sorted({p.name for p in (root / "patches").glob("*") if p.is_dir()})
    out["n_scenes"] = len(scenes)
    out["scenes"] = scenes
    return out


def class_counts(y: np.ndarray) -> "object":
    import pandas as pd
    v, c = np.unique(y, return_counts=True)
    df = pd.DataFrame({"code": v, "class": [CLASSES.get(int(i), "?") for i in v],
                       "pixels": c})
    df["share"] = df["pixels"] / df["pixels"].sum()
    df["role"] = np.where(df["code"] == TARGET, "TARGET",
                  np.where(df["code"].isin(FLOATING_CONFUSERS), "floating confuser",
                  np.where(df["code"].isin(WATER_CLASSES), "water background",
                           "other")))
    return df.sort_values("pixels", ascending=False)


def site_coverage(root: Optional[Path] = None):
    """
    Which of MARIDA's sites exist, from the MGRS tile in each scene name.

    Reported because applying a model trained here to Guinea is a geographic
    extrapolation, and the size of that extrapolation should be a number in
    the methodology rather than a caveat sentence.
    """
    import pandas as pd
    root = Path(root or ensure_extracted())
    rows = []
    for d in sorted((root / "patches").glob("*")):
        if not d.is_dir():
            continue
        parts = d.name.split("_")
        rows.append({"scene": d.name,
                     "date": parts[1] if len(parts) > 1 else "",
                     "mgrs_tile": parts[2] if len(parts) > 2 else "",
                     "n_patches": len(list(d.glob("*_cl.tif")))})
    df = pd.DataFrame(rows)
    # MGRS zone number is the leading digits; Guinea lies in zones 28-29.
    df["utm_zone"] = df["mgrs_tile"].str.extract(r"^(\d+)").astype(float)
    return df
