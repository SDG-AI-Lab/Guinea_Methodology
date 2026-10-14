"""
The MARIDA debris classifier, applied to the FDI rule's detections.

The mean+3sigma FDI-and-NDVI rule (Digital Earth Africa) flags anything that
raises near-infrared above the water baseline. With the Floating Debris Index as
Biermann et al. (2020) define it, that includes vegetation: at Conakry the
rule's pixels have a median NDVI of 0.47 and sit mostly on the shoreline, and on
MARIDA Sargassum is its largest false-positive class. Biermann et al. separate
debris from vegetation and seaweed with a classifier trained on labelled
spectra, not with a threshold. This module does the same with a random forest
trained on MARIDA (39_freeze_debris_filter.py): a rule detection is kept only
where the classifier gives P(Marine Debris) >= p_min (0.5, the binary
classifier's own decision boundary).

Measured on MARIDA's test split exactly as deployed (FDI-and-NDVI rule, then
this classifier), see models/debris_filter_meta.json: precision 0.076 -> 0.895
at unchanged recall 0.247, and on geographically unseen tiles 0.654 -> 0.921 at
recall 0.886.

The model ships inside the package, so a workstation and the backend container
load the same file. If it or a band it needs is missing the filter raises
rather than passing detections through unfiltered.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from .. import config as pwcfg

logger = logging.getLogger(__name__)

MODELS = Path(__file__).resolve().parent / "models"


def settings() -> Dict:
    return pwcfg.DETECTION_CONFIG.get("debris_filter", {})


def enabled() -> bool:
    return bool(settings().get("enabled", False))


def detector_id() -> str:
    """What produced a set of FDI detections: the index definition and, when
    on, the classifier that filters them. Stored with run outputs so a result
    made by a different detector is never resumed or reported as current."""
    formula = pwcfg.S2_CONFIG.get("fdi_formula", "")
    if not enabled():
        return f"{formula}|rule"
    meta_p = MODELS / (Path(settings().get("model", "debris_filter.joblib")).stem
                       + "_meta.json")
    frozen = (json.loads(meta_p.read_text(encoding="utf-8")).get("frozen", "?")
              if meta_p.exists() else "missing")
    return f"{formula}|rule+marida-rf@{frozen}|p>={settings().get('p_min', 0.5)}"


@lru_cache(maxsize=1)
def load() -> Tuple[object, Dict]:
    """The frozen classifier and its metadata (features, p_min, metrics)."""
    import joblib

    name = settings().get("model", "debris_filter.joblib")
    path = MODELS / name
    meta_path = path.with_name(path.stem + "_meta.json")
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Freeze it with "
            f"`python 39_freeze_debris_filter.py` (needs the MARIDA archive).")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return joblib.load(path), meta


def apply(hotspot_mask: np.ndarray,
          bands: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Keep the rule's detections the classifier calls debris.

    Returns (kept_mask, p_debris, info). `p_debris` is NaN outside the rule's
    detections; `info` records the counts before and after and what was used.
    """
    from ..marida_tool import features as F

    model, meta = load()
    p_min = float(settings().get("p_min", meta.get("p_min", 0.5)))
    feats = meta["features"]

    p_debris = np.full(hotspot_mask.shape, np.nan, dtype="float32")
    idx = np.flatnonzero(hotspot_mask)
    info = {"model": settings().get("model", "debris_filter.joblib"),
            "frozen": meta.get("frozen"), "p_min": p_min,
            "n_px_rule": int(idx.size)}
    if idx.size == 0:
        info["n_px_kept"] = 0
        return hotspot_mask.copy(), p_debris, info

    flat = {k: np.asarray(v).ravel()[idx] for k, v in bands.items()
            if hasattr(v, "shape") and np.asarray(v).shape == hotspot_mask.shape}
    X = F.from_composite(flat, feats)
    ok = np.isfinite(X).all(axis=1)
    p = np.full(idx.size, np.nan, dtype="float32")
    if ok.any():
        proba = model.predict_proba(X[ok])
        p[ok] = proba[:, list(model.classes_).index(1)]

    p_debris.ravel()[idx] = p
    kept = np.zeros_like(hotspot_mask, dtype=bool)
    kept.ravel()[idx] = np.nan_to_num(p, nan=0.0) >= p_min

    info["n_px_kept"] = int(kept.sum())
    info["n_px_unscored"] = int((~ok).sum())
    logger.info("debris filter: %s of %s rule pixels kept (P(debris) >= %.2f)",
                f"{info['n_px_kept']:,}", f"{info['n_px_rule']:,}", p_min)
    return kept, p_debris, info
