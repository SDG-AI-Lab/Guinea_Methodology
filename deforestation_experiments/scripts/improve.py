"""Notebook 09: improve the model with what the existing tools and the platform do differently.

Each variant changes one thing in the improved model (16 predictors, trained on the shares):

  equal-area classes    Verra's 30 classes cut by equal forest area (riskmapjnr's best slicing)
                        instead of the geometric series; scored for every variant below
  loss rates at several distances   the local loss rate of the 3 years before over 1.3, 2.7, 10
                        and 20 km, beside the 5 km one (riskmapjnr's moving window, several sizes)
  loss trend            the 5 km rate of the 3 years before minus that of the 3 years before those
  location              latitude and longitude: lets the model learn regional levels, as
                        forestatrisk's spatial random effect for each 10 km cell does
  palm-oil price        the platform's adopted predictor (WFP market prices, palm oil, US$,
                        inverse-distance weighted over the 8 nearest markets, power 2, as
                        data_loader.rasterize_market_points), at the window's start year instead
                        of a fixed 2018
  trained for the amount   a regression forest on each cell's lost share of its forest, cells
                        drawn in proportion to their forest (the expected share lost, instead of
                        lost against kept)

Verra's tests are run for three seeds (their mean), since one seed moves them by 10-30 ha.
"""

import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_variants as D                                     # noqa: E402
import model as MM                                              # noqa: E402
import verra as V                                              # noqa: E402

G, M = MM.G, MM.M
BASE = MM.FEATURES
SCALES = {"Recent loss rate (1.3 km)": 13, "Recent loss rate (2.7 km)": 27,
          "Recent loss rate (10 km)": 101, "Recent loss rate (20 km)": 201}
TREND = "Change in recent loss rate (5 km)"
LOCATION = ["Latitude", "Longitude"]
PRICE = "Palm-oil price (WFP markets)"
VARIANTS = {"Improved model": BASE,
            "Loss rates at several distances": BASE + list(SCALES),
            "Loss trend": BASE + [TREND],
            "Location": BASE + LOCATION,
            "Palm-oil price (platform)": BASE + [PRICE],
            "Trained for the amount of loss": BASE}
EXTRA = list(SCALES) + [TREND] + LOCATION + [PRICE]


@lru_cache(maxsize=1)
def _prices():
    import pandas as pd
    df = pd.read_parquet(G.STORE / "downloads" / "wfp_guinea_prices.parquet")
    df = df[df.commodity.str.strip().str.lower() == "oil (palm)"].copy()
    df["year"] = pd.to_datetime(df.date).dt.year
    return df.dropna(subset=["latitude", "longitude", "usdprice"])


def palm_oil_price(year, rows, neighbours=8, power=2.0, chunk=1_000_000):
    """data_loader.rasterize_market_points at `year`: mean price per market, IDW in degrees."""
    df = _prices()
    g = df[df.year == year].groupby(["latitude", "longitude"], as_index=False).usdprice.mean()
    lat, lon, val = g.latitude.to_numpy(), g.longitude.to_numpy(), g.usdprice.to_numpy()
    x, y = G.lon_lat(G.cells()[MM.hansen()["idx"][rows]])
    out = np.empty(rows.size, dtype="float32")
    k = min(neighbours, val.size)
    for i in range(0, rows.size, chunk):
        d = np.sqrt((x[i:i + chunk, None] - lon) ** 2 + (y[i:i + chunk, None] - lat) ** 2)
        d = np.where(d == 0, 1e-6, d)
        near = np.argsort(d, axis=1)[:, :k]
        w = 1.0 / np.take_along_axis(d, near, axis=1) ** power
        out[i:i + chunk] = (w * val[near]).sum(axis=1) / w.sum(axis=1)
    return out


def build_extra(year, rows):
    import warnings
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for name, size in SCALES.items():
            out[name] = MM.local_loss_rate(year, size=size)[rows]
        out[TREND] = MM.local_loss_rate(year)[rows] - MM.local_loss_rate(year - 3)[rows]
    x, y = G.lon_lat(G.cells()[MM.hansen()["idx"][rows]])
    out["Latitude"], out["Longitude"] = y.astype("float32"), x.astype("float32")
    out[PRICE] = palm_oil_price(year, rows)
    return out


def equal_area_classes(risk, forest_ha, n=V.N_CLASSES):
    """30 classes of equal forest area, by risk (1 = lowest); 0 where there is no forest or no risk."""
    valid = (forest_ha > 0) & np.isfinite(risk)
    idx = np.flatnonzero(valid)
    order = idx[np.argsort(risk[idx], kind="stable")]
    cum = np.cumsum(forest_ha[order]) - forest_ha[order] / 2
    classes = np.zeros(risk.shape, dtype="uint8")
    classes[order] = np.minimum((cum / forest_ha[order].sum() * n).astype(int), n - 1) + 1
    return classes


def _fit_regression(X, y, seed):
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(n_estimators=200, max_depth=18, min_samples_leaf=5, n_jobs=-1,
                                 random_state=seed).fit(X, y)


def _draw_amount(t1, t2, seed):
    """Cells drawn in proportion to their forest; target: the share of that forest lost."""
    f1, lw = MM.forest_pct(t1), MM.loss_pct(t1, t2)
    cells = np.flatnonzero(f1 > 0)
    rng = np.random.default_rng(seed)
    chosen = MM._pick(cells, 2 * MM.SAMPLES, rng, (f1[cells] * MM.hansen()["area"][cells]).astype("float64"))
    return chosen, np.clip(lw[chosen] / f1[chosen], 0, 1)


def _predict(model, values, names, count, chunk=1_000_000):
    risk = np.full(count, np.nan, dtype="float32")
    for i in range(0, count, chunk):
        X = np.column_stack([values[m][i:i + chunk] for m in names])
        ok = np.isfinite(X).all(axis=1)
        part = risk[i:i + chunk]
        part[ok] = model.predict_proba(X[ok])[:, 1] if hasattr(model, "predict_proba") else model.predict(X[ok])
    return risk


def run_split(split, seeds=(42, 1, 2), variants=None, log=print):
    variants = variants or VARIANTS
    t1, t2, t3 = split
    start = time.perf_counter()
    n = MM.hansen()["idx"].size
    area = MM.hansen()["area"]
    f1, f2 = MM.forest_pct(t1), MM.forest_pct(t2)
    rows1, rows2 = np.flatnonzero(f1 > 0), np.flatnonzero(f2 > 0)
    names_all = sorted({m for v in variants.values() for m in v})

    def values(year, rows):
        out = D.build(year, [m for m in names_all if m not in EXTRA], rows)
        extra = [m for m in names_all if m in EXTRA]
        if extra:
            built = build_extra(year, rows)
            out.update({m: built[m] for m in extra})
        return out

    cal = values(t1, rows1)
    log(f"{M.split_label(split)}: calibration predictors {time.perf_counter() - start:.0f} s")
    models, classes_fit = {}, {}
    for name, names in variants.items():
        for s in seeds:
            if name == "Trained for the amount of loss":
                cells, y = _draw_amount(t1, t2, s)
            else:
                cells, y = D._draw(t1, t2, s)
            X = np.column_stack([cal[m][np.searchsorted(rows1, cells)] for m in names])
            ok = np.isfinite(X).all(axis=1)
            model = _fit_regression(X[ok], y[ok], s) if name == "Trained for the amount of loss" else D._fit(X[ok], y[ok], s)
            models[name, s] = model
            risk = np.full(n, np.nan, dtype="float32")
            risk[rows1] = _predict(model, cal, names, rows1.size)
            classes_fit[name, s] = (M.alternative_classes(risk, f1 > 0), equal_area_classes(risk, f1 / 100 * area))
    del cal
    log(f"{M.split_label(split)}: models fitted {time.perf_counter() - start:.0f} s")

    test = values(t2, rows2)
    out = {"split": M.split_label(split), "variants": {}}
    for name, names in variants.items():
        r = out["variants"][name] = {"aucs": [], "geometric": {"fit": [], "prediction": []},
                                     "equal_area": {"fit": [], "prediction": []}}
        for s in seeds:
            risk = np.full(n, np.nan, dtype="float32")
            risk[rows2] = _predict(models[name, s], test, names, rows2.size)
            r["aucs"].append(V.pixel_auc(risk, t2, t3))
            geo_fit, ea_fit = classes_fit.pop((name, s))
            for key, fit, pred in (("geometric", geo_fit, M.alternative_classes(risk, f2 > 0)),
                                   ("equal_area", ea_fit, equal_area_classes(risk, f2 / 100 * area))):
                tests = V.vt0007_tests(split, classes=(fit, pred))
                r[key]["fit"].append(tests["fit_medae_ha"])
                r[key]["prediction"].append(tests["prediction_medae_ha"])
            if s == seeds[0] and hasattr(models[name, s], "feature_importances_"):
                r["importance"] = M.DR.driver_importance(models[name, s], names)
        log(f"{out['split']}: {name} AUC {np.mean(r['aucs']):.4f} prediction {np.mean(r['geometric']['prediction']):.0f} "
            f"/ equal area {np.mean(r['equal_area']['prediction']):.0f} ({time.perf_counter() - start:.0f} s)")
    out["seconds"] = time.perf_counter() - start
    out["peak_gb"] = MM._peak_gb()
    return out


def split_name(split, tag="improve"):
    return "{}_{}_{}_{}".format(tag, *split)


if __name__ == "__main__":
    for split in MM.SPLITS:
        MM.cached(split_name(split), lambda: run_split(split), rerun="--rerun" in sys.argv)
        print("SAVED", split_name(split), flush=True)


# --------------------------------------------------------------------------
# Classes fitted as riskmapjnr fits them

def run_fit_at_end(split, seeds=(42, 1, 2), log=print):
    """The improved model with Verra's class rates taken from its map at the END of the calibration
    window (predictors at t2 over the forest at t1, so the local loss rate holds the calibration
    window's own loss), as riskmapjnr builds its calibration map. Training and the test map are
    unchanged: the model learns from predictors at t1, and the prediction test scores its map at t2."""
    t1, t2, t3 = split
    start = time.perf_counter()
    n = MM.hansen()["idx"].size
    area = MM.hansen()["area"]
    f1, f2 = MM.forest_pct(t1), MM.forest_pct(t2)
    rows1 = np.flatnonzero(f1 > 0)
    cal = D.build(t1, BASE, rows1)
    models = {}
    for s in seeds:
        cells, y = D._draw(t1, t2, s)
        X = np.column_stack([cal[m][np.searchsorted(rows1, cells)] for m in BASE])
        ok = np.isfinite(X).all(axis=1)
        models[s] = D._fit(X[ok], y[ok], s)
    del cal
    end = D.build(t2, BASE, rows1)                     # predictors at t2, over the forest at t1
    out = {"split": M.split_label(split), "aucs": [], "geometric": {"fit": [], "prediction": []},
           "equal_area": {"fit": [], "prediction": []}}
    for s in seeds:
        risk_end = np.full(n, np.nan, dtype="float32")
        risk_end[rows1] = _predict(models[s], end, BASE, rows1.size)
        risk_fit = np.where(f1 > 0, risk_end, np.nan)
        risk_test = np.where(f2 > 0, risk_end, np.nan)       # the test map: the same predictors, forest at t2
        out["aucs"].append(V.pixel_auc(risk_test, t2, t3))
        for key, fit, pred in (("geometric", M.alternative_classes(risk_fit, f1 > 0), M.alternative_classes(risk_test, f2 > 0)),
                               ("equal_area", equal_area_classes(risk_fit, f1 / 100 * area), equal_area_classes(risk_test, f2 / 100 * area))):
            tests = V.vt0007_tests(split, classes=(fit, pred))
            out[key]["fit"].append(tests["fit_medae_ha"])
            out[key]["prediction"].append(tests["prediction_medae_ha"])
        log(f"{out['split']}: seed {s} prediction {out['geometric']['prediction'][-1]:.0f} / equal area "
            f"{out['equal_area']['prediction'][-1]:.0f} ({time.perf_counter() - start:.0f} s)")
    out["seconds"] = time.perf_counter() - start
    out["peak_gb"] = MM._peak_gb()
    return out


# --------------------------------------------------------------------------
# Saved maps, for adjustments that need no retraining

MAPS = MM.HERE / "data" / "maps"


def save_maps(split, seed=42):
    """The improved model's maps at t1 (forest at t1) and t2 (forest at t2), and riskmapjnr's
    rescaled local rate and classes, as .npy over the cells with forest in 2000 (data/maps/)."""
    import tool_comparison as T
    t1, t2, t3 = split
    MAPS.mkdir(parents=True, exist_ok=True)
    n = MM.hansen()["idx"].size
    f1, f2 = MM.forest_pct(t1), MM.forest_pct(t2)
    rows1, rows2 = np.flatnonzero(f1 > 0), np.flatnonzero(f2 > 0)
    cal = D.build(t1, BASE, rows1)
    cells, y = D._draw(t1, t2, seed)
    X = np.column_stack([cal[m][np.searchsorted(rows1, cells)] for m in BASE])
    ok = np.isfinite(X).all(axis=1)
    model = D._fit(X[ok], y[ok], seed)
    fit = np.full(n, np.nan, dtype="float32")
    fit[rows1] = _predict(model, cal, BASE, rows1.size)
    del cal
    test_values = D.build(t2, BASE, rows2)
    test = np.full(n, np.nan, dtype="float32")
    test[rows2] = _predict(model, test_values, BASE, rows2.size)
    del test_values
    classes, (rmj_fit, rmj_test), _ = T.riskmapjnr(split, log=lambda *a: None)
    tag = f"{t1}_{seed}"
    for name, values in (("improved_fit", fit), ("improved_test", test), ("rmj_fit", rmj_fit), ("rmj_test", rmj_test),
                         ("rmj_classes_fit", classes[0].astype("uint8")), ("rmj_classes_test", classes[1].astype("uint8"))):
        np.save(MAPS / f"{name}_{tag}.npy", values)


def load_map(name, split, seed=42):
    return np.load(MAPS / f"{name}_{split[0]}_{seed}.npy")


def smooth(risk, forest_pct, size):
    """Forest-weighted mean of the risk within size x size cells, over the forest cells."""
    from scipy.ndimage import uniform_filter
    w = np.where(np.isfinite(risk), forest_pct, 0).astype("float32")
    num = uniform_filter(MM._on_grid(np.nan_to_num(risk) * w, 0.0), size, mode="constant")
    den = uniform_filter(MM._on_grid(w, 0.0), size, mode="constant")
    num /= np.maximum(den, 1e-9)
    del den
    out = G.from_grid(num)[MM.hansen()["idx"]]
    return np.where(forest_pct > 0, out, np.nan).astype("float32")


def adjustments(split, seed=42):
    """Changes to how the improved model's saved maps (save_maps) become Verra's classes, without
    retraining: {label: (AUC, fit MedAE, prediction MedAE)}."""
    import tool_comparison as T
    from scipy.ndimage import uniform_filter
    t1, t2, t3 = split
    area = MM.hansen()["area"]
    f1, f2 = MM.forest_pct(t1), MM.forest_pct(t2)
    fit, test = load_map("improved_fit", split, seed), load_map("improved_test", split, seed)
    rfit, rtest = load_map("rmj_fit", split, seed), load_map("rmj_test", split, seed)
    cf_rmj, cp_rmj = load_map("rmj_classes_fit", split, seed), load_map("rmj_classes_test", split, seed)
    out = {}

    def score(label, a=None, b=None, classes=None):
        if classes is None:
            classes = (M.alternative_classes(a, f1 > 0), M.alternative_classes(b, f2 > 0))
        r = V.vt0007_tests(split, classes=classes)
        out[label] = (V.pixel_auc(b, t2, t3) if b is not None else np.nan, r["fit_medae_ha"], r["prediction_medae_ha"])

    score("Improved model", fit, test)
    for size, km in ((13, "1.3"), (27, "2.7"), (51, "5.1")):
        score(f"Risk smoothed over {km} km", smooth(fit, f1, size), smooth(test, f2, size))
    score("Zero risk beyond riskmapjnr's distance threshold",
          np.where(rfit == 1, np.nanmin(fit), fit), np.where(rtest == 1, np.nanmin(test), test))

    def quiet(risk, forest, rate):
        q = forest & (rate <= 0.001)
        c = M.alternative_classes(np.where(q, np.nan, risk), forest & ~q)
        c = np.where(c > 0, np.minimum(c, 29) + 1, 0).astype("uint8")
        c[q] = 1
        return c
    score("Forest without loss within 5 km in a class of its own",
          classes=(quiet(fit, f1 > 0, MM.local_loss_rate(t1)), quiet(test, f2 > 0, MM.local_loss_rate(t2))))

    def split_classes(rmj, risk, forest_ha):
        bucket = np.where(rmj > 0, (rmj.astype(int) - 1) // 2, -1)
        c = np.zeros(risk.shape, dtype="uint8")
        for b in np.unique(bucket[bucket >= 0]):
            inside = (bucket == b) & np.isfinite(risk) & (forest_ha > 0)
            sub = equal_area_classes(np.where(inside, risk, np.nan), np.where(inside, forest_ha, 0), n=2)
            c[inside] = b * 2 + sub[inside]
        missing = (bucket >= 0) & (c == 0) & (forest_ha > 0)
        c[missing] = bucket[missing] * 2 + 1
        return c
    score("riskmapjnr's classes in pairs, each split in two by our risk",
          classes=(split_classes(cf_rmj, fit, f1 / 100 * area), split_classes(cp_rmj, test, f2 / 100 * area)))

    w = np.where(np.isfinite(test), f2, 0).astype("float32")
    num = uniform_filter(MM._on_grid(np.nan_to_num(test) * w, 0.0), 27, mode="constant")
    num /= np.maximum(uniform_filter(MM._on_grid(w, 0.0), 27, mode="constant"), 1e-9)
    surface = G.from_grid(num)[MM.hansen()["idx"]].astype("float32")
    del num
    score("riskmapjnr's construction: our end-of-calibration risk as a 2.7 km surface for both maps",
          np.where(f1 > 0, surface, np.nan), np.where(f2 > 0, surface, np.nan))
    return {k: list(map(float, v)) for k, v in out.items()}
