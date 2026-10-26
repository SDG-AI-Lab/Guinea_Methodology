"""CLUE-S on the benchmark data, for the tool comparison (notebook 00).

The allocation is a line-by-line port of the authors' own source code, Dyna-CLUE 2.7
(Verburg & Overmars 2009; https://doi.org/10.34894/RLDNDX, CC0), functions calc_reg,
calc_change_ch, demand_dir and comp_demand in clue_logist2dneigh.cpp. That code is a
Borland C++Builder program and cannot be built here, so it is ported to numpy.

CLUE-S in short (Verburg et al. 2002):

1. One logistic regression per class: where is that class found, given the driving
   factors? This gives a probability map per class.
2. Every pixel takes the class with the highest
       probability + iteration variable of the class (+ conversion elasticity if it is
       the pixel's current class),
   among the conversions allowed.
3. The iteration variables are adjusted until the area of every class matches the
   demand (the total number of pixels of each class at the end of the step).

Settings, following the Dyna-CLUE manual and how Pontius et al. (2008) ran it:
regressions on the start map (2024) with the start year's 12 predictors; conversion
elasticity of each class = the share of its pixels that stayed the same in 2023-2024;
every conversion allowed; demand = the observed 2025 total of each class (Pontius et al.
gave every model the correct quantity); no neighbourhood option (standard CLUE-S).
"""
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import esri_methods as E
import nb_helpers as H

OUT = E.RUNS / "clue_s" / "test"


def regressions(start, features, domain, classes, n_sample=300_000, seed=0):
    """Per class, logistic regression of presence on the standardised predictors, fitted
    on a random sample of `domain`. Returns P (n_classes, n_domain_pixels), the ROC AUC of
    each regression on its own sample (the fit check CLUE-S users report), the domain's
    pixel indices, and each class's (constant, betas) on the unstandardised predictors, the
    form the original program reads from alloc1.reg."""
    rng = np.random.default_rng(seed)
    idx = np.flatnonzero(domain.ravel())
    sample = rng.choice(idx, min(n_sample, idx.size), replace=False)
    X_all = features.reshape(features.shape[0], -1)[:, idx].T
    scaler = StandardScaler().fit(features.reshape(features.shape[0], -1)[:, sample].T)
    X_all = scaler.transform(X_all)
    Xs = scaler.transform(features.reshape(features.shape[0], -1)[:, sample].T)
    y_all = start.ravel()
    P, auc, raw = np.zeros((len(classes), idx.size), np.float32), {}, {}
    for k, c in enumerate(classes):
        y = (y_all[sample] == c).astype(int)
        model = LogisticRegression(max_iter=2000).fit(Xs, y)
        auc[int(c)] = float(roc_auc_score(y, model.predict_proba(Xs)[:, 1]))
        P[k] = model.predict_proba(X_all)[:, 1]
        betas = model.coef_[0] / scaler.scale_
        raw[int(c)] = (float(model.intercept_[0] - np.dot(betas, scaler.mean_)), betas)
    return P, auc, idx, raw


def allocate(P, old, demand, stat, allowed=None, seed_value=0.05, avg_max=0.35, ind_max=3.0,
             max_loops=20_000, rng_seed=0, log=print):
    """Dyna-CLUE allocation. P: (k, n) probabilities; old: (n,) current class index 0..k-1;
    demand: (k,) pixels wanted per class; stat: (k,) conversion elasticities (0..1).
    Stops when the average and largest % difference between allocated area and demand
    are below avg_max and ind_max (Dyna-CLUE's itermode 0 and its two thresholds, at the
    values of the manual's example main.1)."""
    k, n = P.shape
    rng = np.random.default_rng(rng_seed)
    allowed = np.ones((k, k), bool) if allowed is None else allowed
    elas = np.zeros(k)
    speed = np.where(demand / n < 0.001, 0.01, seed_value)          # demand_dir
    bonus = np.where(stat < 1, stat, 0.0)
    cols = np.arange(n)
    for loop in range(1, max_loops + 1):
        score = P + elas[:, None].astype(np.float32)               # calc_change_ch
        score[old, cols] += bonus[old].astype(np.float32)
        if not allowed.all():
            score[~allowed[old].T] = -3
        new = np.argmax(score, axis=0)
        area = np.bincount(new, minlength=k).astype(float)          # comp_demand
        speed += 0.00005
        ran = rng.integers(0, 101, size=k) + 1                      # random(101) + 1
        elas -= (area - demand) / (n * speed * ran)
        elas[np.abs(elas) > 1.5] = 0
        pct = np.abs(area - demand) / np.maximum(demand, 1) * 100
        if pct.mean() <= avg_max and pct.max() <= ind_max:
            break
        if loop % 500 == 0:
            log(f"   iteration {loop}: largest difference {pct.max():.2f}%, average {pct.mean():.2f}%")
    done = "converged" if pct.mean() <= avg_max and pct.max() <= ind_max else "stopped at the iteration limit"
    log(f"CLUE-S {done} after {loop} iterations: largest difference {pct.max():.2f}%, average {pct.mean():.2f}%")
    return new, loop


def setup():
    """Inputs shared by the Python port and the original program: start map, domain,
    predictors, conversion elasticities and demand."""
    profile = E.grid()
    guinea = H.guinea_mask(profile)
    a, b, c = (E.esri_cached(y) for y in (2023, 2024, 2025))
    domain = guinea & E.valid_classes(b)
    classes = [int(x) for x in sorted(H.ESRI_COLOURS)]
    features = E.predictor_stack(profile, E.deployed_mask(profile), 2024)
    both = guinea & E.valid_classes(a, b)
    stat = np.array([float(((a == cl) & (b == cl) & both).sum() / max(((a == cl) & both).sum(), 1)) for cl in classes])
    demand = np.array([float(((c == cl) & domain).sum()) for cl in classes])
    # pixels of the domain that are nodata in 2025 still need a class: give the demand
    # their share so the totals add up to the domain size
    demand *= domain.sum() / demand.sum()
    return dict(profile=profile, guinea=guinea, start=b, domain=domain, classes=classes,
                features=features, stat=stat, demand=demand)


def run(avg_max=1.0, ind_max=5.0, log=print):
    """Fit and allocate one 2024 -> 2025 step inside Guinea. Writes the forecast (ESRI codes)
    and one probability surface per class, and returns a summary. avg_max, ind_max: the
    average and the largest % difference allowed between allocated area and demand (0.35 and 3
    in the manual's example; 1 and 5 here, because Bare ground has only 648 pixels: it could
    not be brought within 3% in 20,000 iterations, and its 4% alone makes the average of the 7
    classes 0.57%). The original program is run with the same values (export_original)."""
    s = setup()
    profile, guinea, b, classes, stat, demand = (s[k] for k in ("profile", "guinea", "start", "classes", "stat", "demand"))
    P, auc, idx, _ = regressions(b, s["features"], s["domain"], classes)
    lut = {cl: i for i, cl in enumerate(classes)}
    old = np.vectorize(lut.get)(b.ravel()[idx])
    new, loops = allocate(P, old, demand, stat, avg_max=avg_max, ind_max=ind_max, log=log)
    sim = b.copy()
    sim.flat[idx] = np.array(classes)[new]
    E.write_classes(np.where(guinea, sim, 255), OUT / "clue_s_2025.tif", profile)
    surfaces = {}
    for k, cl in enumerate(classes):
        s = np.full(b.size, -1.0, np.float32)
        s[idx] = P[k]
        surfaces[cl] = E.write_float(s.reshape(b.shape), OUT / "probability" / f"prob_{cl}.tif", profile)
    summary = {"classes": classes, "regression_auc": auc, "conversion_elasticity": dict(zip(classes, stat.round(3).tolist())),
               "demand": dict(zip(classes, demand.round().tolist())), "iterations": loops}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    return sim, surfaces, summary


# --------------------------------------------------------------------------
# The original program (Dyna-CLUE 2.7, clues.exe), run on the same inputs

ORIGINAL = E.RUNS / "clue_s_original" / "test"


def _ascii_grid(array, path, profile, fmt):
    """ArcView ASCII grid with header (main.1 line 15 = 1), -9999 outside the domain."""
    t = profile["transform"]
    header = (f"ncols {profile['width']}\nnrows {profile['height']}\nxllcorner {t.c}\n"
              f"yllcorner {t.f + t.e * profile['height']}\ncellsize {t.a}\nNODATA_value -9999")
    np.savetxt(path, array, fmt=fmt, header=header, comments="")


def export_original(folder=ORIGINAL, log=print):
    """Write the input files of the original CLUE-S program (Dyna-CLUE manual, 'Creating a
    new application'): cov_all.0, region1.fil, sc1gr0-11.fil, alloc1.reg (the same
    regressions as the port, on unstandardised predictors), allow.txt, main.1, demand.in1.
    Demand is in pixels, so the cell area (main.1 line 7) is 1. Convergence limits (line 12):
    1% average and 6% for any class. With 5%, as the port, the program reached its built-in
    limit of 20,000 iterations with Bare ground 5.7% off and wrote no map ("no solution")."""
    s = setup()
    folder.mkdir(parents=True, exist_ok=True)
    b, domain, classes, profile = s["start"], s["domain"], s["classes"], s["profile"]
    _, auc, _, raw = regressions(b, s["features"], domain, classes)
    lut = np.full(256, -9999, np.int32)
    lut[classes] = np.arange(len(classes))
    _ascii_grid(np.where(domain, lut[b], -9999), folder / "cov_all.0", profile, "%d")
    _ascii_grid(np.where(domain, 0, -9999), folder / "region1.fil", profile, "%d")
    for j, layer in enumerate(s["features"]):
        _ascii_grid(np.where(domain, layer, -9999), folder / f"sc1gr{j}.fil", profile, "%.7g")
        log(f"   wrote sc1gr{j}.fil")
    n = s["features"].shape[0]
    reg = []
    for k, cl in enumerate(classes):
        const, betas = raw[cl]
        reg += [str(k), f"\t{const:.8g}", str(n)] + [f"\t{beta:.8g} {j}" for j, beta in enumerate(betas)] + [""]
    (folder / "alloc1.reg").write_text("\n".join(reg) + "\n")
    (folder / "allow.txt").write_text("\n".join(" ".join(["1"] * len(classes)) for _ in classes) + "\n")
    t = profile["transform"]
    main = [len(classes), 1, n, n, profile["height"], profile["width"], 1, t.c, t.f + t.e * profile["height"],
            " ".join(str(k) for k in range(len(classes))), " ".join(f"{x:.3f}" for x in s["stat"]),
            "0 1 6", "2024 2025", 0, 1, 0, "1 5", 0, 0, 0.05]
    (folder / "main.1").write_text("\n".join(str(x) for x in main) + "\n")
    now = np.array([float(((b == cl) & domain).sum()) for cl in classes])
    (folder / "demand.in1").write_text("2\n" + "\t".join(f"{x:.1f}" for x in now) + "\n"
                                       + "\t".join(f"{x:.1f}" for x in s["demand"]) + "\n")
    (folder / "regression_auc.json").write_text(json.dumps(auc, indent=2))
    return folder


def read_original(folder=ORIGINAL, name="cov_all.1"):
    """The original program's output map, back to ESRI codes (0 outside the domain), and
    written as clue_s_original_2025.tif."""
    s = setup()
    with open(folder / name) as f:                   # values in row order, whatever the line breaks
        lines = f.read().split("\n", 6)
    grid = np.array(lines[6].split(), dtype=np.int32).reshape(s["start"].shape)
    sim = np.where(grid >= 0, np.array(s["classes"])[np.clip(grid, 0, None)], 0).astype(np.uint8)
    E.write_classes(np.where(s["guinea"], sim, 255), folder / "clue_s_original_2025.tif", s["profile"])
    return sim
