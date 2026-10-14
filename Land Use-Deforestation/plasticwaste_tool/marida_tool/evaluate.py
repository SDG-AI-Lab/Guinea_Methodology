"""
Comparing the deployed FDI rule against a learned classifier, fairly.

THE COMPARISON HAS TO BE MATCHED OR IT IS MEANINGLESS

The FDI rule is binary: a pixel is above mean+3sigma or it is not. The forest
emits a probability. Comparing them at the forest's default 0.5 cut compares
two arbitrary operating points and tells you nothing about which method is
better -- it tells you which happened to be tuned more aggressively.

So every comparison here is made at MATCHED RECALL: the forest's threshold is
moved until it finds the same fraction of true debris the FDI rule finds, and
the two are then compared on precision. That isolates the question actually
being asked -- given the same detection sensitivity, how much cleaner is the
result?

ROC-AUC is reported alongside because it is prevalence-invariant, and debris
prevalence differs between MARIDA and Guinea by an unknown factor. Precision
does not transfer across prevalence; AUC does.

WHAT A GOOD RESULT WOULD AND WOULD NOT LICENSE

Even a large improvement here does not license calling Guinea's detections
plastic. MARIDA's "Marine Debris" label comes from expert visual
interpretation of Sentinel-2 imagery at sites with reported debris events, not
from anyone retrieving material from the water. The strongest claim this
evidence can support is about separability from OTHER FLOATING MATERIAL --
which is a real and useful claim, and a different one.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from . import baseline as B
from . import model as M
from .data import CLASSES, FLOATING_CONFUSERS, TARGET, WATER_CLASSES


def compare_at_matched_recall(y_true_binary: np.ndarray,
                              rule_pred: np.ndarray,
                              model_score: np.ndarray,
                              scored: Optional[np.ndarray] = None) -> Dict:
    """
    The headline comparison: same recall, whose positives are cleaner?
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    m = np.ones(len(y_true_binary), bool) if scored is None else scored
    t = np.asarray(y_true_binary, bool)[m]
    rp = np.asarray(rule_pred, bool)[m]
    ms = np.asarray(model_score, float)[m]

    tp = int((t & rp).sum()); fp = int((~t & rp).sum())
    rule_rec = tp / t.sum() if t.sum() else np.nan
    rule_prec = tp / (tp + fp) if (tp + fp) else np.nan

    cut = M.threshold_for_recall(t, ms, rule_rec) if np.isfinite(rule_rec) else np.nan
    mp = ms >= cut if np.isfinite(cut) else np.zeros_like(t)
    mtp = int((t & mp).sum()); mfp = int((~t & mp).sum())
    model_rec = mtp / t.sum() if t.sum() else np.nan
    model_prec = mtp / (mtp + mfp) if (mtp + mfp) else np.nan

    base = float(t.mean())
    return {
        "n_scored": int(m.sum()), "n_debris": int(t.sum()), "base_rate": base,
        "rule_recall": rule_rec, "rule_precision": rule_prec,
        "rule_lift": rule_prec / base if base else np.nan,
        "rule_false_positives": fp,
        "model_cut_at_matched_recall": float(cut),
        "model_recall": model_rec, "model_precision": model_prec,
        "model_lift": model_prec / base if base else np.nan,
        "model_false_positives": mfp,
        "precision_ratio": (model_prec / rule_prec
                            if (rule_prec and np.isfinite(rule_prec)) else np.nan),
        "false_positives_removed": fp - mfp,
        # Prevalence-invariant, so these are the numbers that transfer to a
        # setting where debris is rarer or commoner than in MARIDA.
        "model_roc_auc": float(roc_auc_score(t, ms)),
        "model_average_precision": float(average_precision_score(t, ms)),
        "rule_roc_auc_equivalent": float(roc_auc_score(t, rp.astype(float))),
    }


def confuser_breakdown(y: np.ndarray, rule_pred: np.ndarray,
                       model_score: np.ndarray, cut: float,
                       scored: Optional[np.ndarray] = None):
    """
    Per-class false-positive comparison at the matched operating point.

    This is the table that says what machine learning actually buys: not a
    better score, but the removal of specific, nameable confusions. A reader
    can check it against their own expectation of what floats in the tropical
    Atlantic.
    """
    import pandas as pd

    m = np.ones(len(y), bool) if scored is None else scored
    yy = np.asarray(y)[m]
    rp = np.asarray(rule_pred, bool)[m]
    mp = np.asarray(model_score, float)[m] >= cut

    rows = []
    for code, name in CLASSES.items():
        sel = yy == code
        n = int(sel.sum())
        if n == 0:
            continue
        r_hit, m_hit = int((sel & rp).sum()), int((sel & mp).sum())
        rows.append({
            "code": code, "class": name, "pixels": n,
            "role": ("TARGET" if code == TARGET
                     else "floating confuser" if code in FLOATING_CONFUSERS
                     else "water background" if code in WATER_CLASSES
                     else "other"),
            "rule_flagged": r_hit, "rule_rate": r_hit / n,
            "model_flagged": m_hit, "model_rate": m_hit / n,
            "change": m_hit - r_hit,
        })
    df = pd.DataFrame(rows)
    return df.sort_values("rule_flagged", ascending=False)


def what_a_detection_probably_is(y: np.ndarray, pred: np.ndarray):
    """
    Given that a detector fired, the posterior over what the pixel was.

    Reported for both detectors. For the methodology this is the sentence that
    matters: 'of every 100 pixels the deployed rule flags, N are annotated
    debris and M are Sargassum'. It is a statement about MARIDA's class mix,
    not Guinea's, and must be labelled as such -- but the RANKING of confusers
    is informative wherever the same materials float.
    """
    import pandas as pd
    sel = np.asarray(pred, bool)
    if not sel.any():
        return pd.DataFrame(columns=["class", "flagged", "share_of_detections"])
    yy = np.asarray(y)[sel]
    v, c = np.unique(yy, return_counts=True)
    df = pd.DataFrame({"code": v,
                       "class": [CLASSES.get(int(i), "?") for i in v],
                       "flagged": c})
    df["share_of_detections"] = df["flagged"] / df["flagged"].sum()
    return df.sort_values("flagged", ascending=False)


def transfer_warning(marida_sites, guinea_zone: int = 28) -> str:
    """
    The caveat that must accompany any Guinean prediction from this model.
    """
    zones = sorted({int(z) for z in marida_sites["utm_zone"].dropna().unique()})
    near = [z for z in zones if abs(z - guinea_zone) <= 2]
    return (
        f"MARIDA covers UTM zones {zones}. Guinea lies in zone {guinea_zone}. "
        f"{'No' if not near else str(len(near))} MARIDA scene"
        f"{'s are' if len(near) != 1 else ' is'} within two zones of it, so "
        f"applying this model to Guinea is a geographic extrapolation: the "
        f"water optical properties, atmospheric correction residuals and "
        f"floating-material mix are all unverified there. The model's MARIDA "
        f"test scores are an upper bound on its Guinean performance, not an "
        f"estimate of it. Additionally, MARIDA's 'Marine Debris' label is "
        f"expert visual interpretation of Sentinel-2 imagery at sites with "
        f"reported debris events -- not physical sampling -- so it supports "
        f"claims about separability from other floating material, not claims "
        f"that detected material is plastic.")
