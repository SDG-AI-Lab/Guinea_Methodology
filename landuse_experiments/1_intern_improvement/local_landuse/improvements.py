"""Predeclared local improvement experiments with earlier-period selection.

Training 2005->2010; validation 2010->2015 in held-out spatial blocks;
refit 2010->2015; final retrospective comparison 2015->2020. No choice depends
on 2020 labels. Historical features always end at the forecast start year.
"""

import argparse
import hashlib
import json
import logging
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
from scipy.ndimage import distance_transform_edt, uniform_filter
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from .comparison import DEFAULT_DATA, evaluate, plot_comparison, read_raster, write_csv, write_map
from .controlled import block_mask
from .models import NODATA, sample_digest, sample_training_indices, spatial_split

LOGGER = logging.getLogger(__name__)

# Fixed before validation. The original no-history baseline is always retained.
CANDIDATES = [
    {"name": "rf_baseline", "history": False, "multiscale": False, "family": "rf",
     "params": {"n_estimators": 100, "max_depth": 20, "min_samples_split": 5, "min_samples_leaf": 2}},
    {"name": "rf_history", "history": True, "multiscale": False, "family": "rf",
     "params": {"n_estimators": 100, "max_depth": 20, "min_samples_split": 5, "min_samples_leaf": 2}},
    {"name": "rf_history_multiscale", "history": True, "multiscale": True, "family": "rf",
     "params": {"n_estimators": 100, "max_depth": 20, "min_samples_split": 5, "min_samples_leaf": 2}},
    {"name": "rf_regularized", "history": True, "multiscale": True, "family": "rf",
     "params": {"n_estimators": 250, "max_depth": 12, "min_samples_split": 10, "min_samples_leaf": 10,
                "max_features": .7}},
    {"name": "rf_flexible", "history": True, "multiscale": True, "family": "rf",
     "params": {"n_estimators": 250, "max_depth": None, "min_samples_split": 5, "min_samples_leaf": 2,
                "max_features": .7}},
    {"name": "xgb_shallow", "history": True, "multiscale": True, "family": "xgb",
     "params": {"n_estimators": 180, "max_depth": 3, "learning_rate": .07,
                "min_child_weight": 5, "subsample": .8, "colsample_bytree": .8, "reg_lambda": 5}},
    {"name": "xgb_deeper", "history": True, "multiscale": True, "family": "xgb",
     "params": {"n_estimators": 150, "max_depth": 6, "learning_rate": .07,
                "min_child_weight": 5, "subsample": .8, "colsample_bytree": .8, "reg_lambda": 5}},
]


def build_features(current, covariates, classes, indices, *, previous=None,
                   history=False, multiscale=False, lag_years=5):
    """Extract only requested rows, avoiding a full-country 100-band stack.

    Baseline columns match models.feature_stack exactly. Missing historical
    neighbors do not become evidence of change: explicit availability is used.
    The 2001->2005 lag is four years; subsequent lags are five. Change-pressure
    fractions/differences are annualized and the interval is explicitly encoded.
    """
    if history and (previous is None or previous.shape != current.shape or lag_years <= 0):
        raise ValueError("History needs a matching previous map and positive lag duration")
    rows, names = [], []

    def add(name, values):
        rows.append(np.asarray(values, dtype=np.float32).ravel()[indices])
        names.append(name)

    valid = current != NODATA

    def fraction(indicator, available, size):
        numerator = uniform_filter(indicator.astype(np.float32), size, mode="constant", cval=0)
        denominator = uniform_filter(available.astype(np.float32), size, mode="constant", cval=0)
        return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)

    for name, a in covariates.items():
        if a.shape != current.shape:
            raise ValueError(f"Predictor grid mismatch: {name}")
        add(name, a)
    for c in classes:
        indicator = current == c
        add(f"current_class_{int(c)}", indicator)
        add(f"fraction_class_{int(c)}_3x3", fraction(indicator, valid, 3))
    if history:
        history_valid = valid & (previous != NODATA)
        changed = history_valid & (current != previous)
        add("history_available", history_valid)
        add("changed_since_previous", changed)
        add("history_interval_years", np.full(current.shape, lag_years, dtype=np.float32))
        for c in classes:
            add(f"previous_class_{int(c)}", (previous == c) & history_valid)
        add("recent_change_fraction_per_year_3x3", fraction(changed, history_valid, 3) / lag_years)
        distance = (distance_transform_edt(~changed) if changed.any()
                    else np.full(current.shape, max(current.shape), dtype=np.float32))
        add("distance_to_previous_change_pixels", distance)
    if multiscale:
        for size in (7, 15):
            for c in classes:
                now = fraction(current == c, valid, size)
                add(f"fraction_class_{int(c)}_{size}x{size}", now)
                if history:
                    # Same valid support for current and past fractions.
                    delta = (fraction((current == c) & history_valid, history_valid, size)
                             - fraction((previous == c) & history_valid, history_valid, size)) / lag_years
                    add(f"fraction_trend_class_{int(c)}_{size}x{size}", delta)
            if history:
                add(f"recent_change_fraction_per_year_{size}x{size}",
                    fraction(changed, history_valid, size) / lag_years)
    return np.column_stack(rows).astype(np.float32), names


class EncodedModel:
    """Preserve original class codes when XGBoost requires contiguous labels."""
    def __init__(self, model, classes):
        self.model = model
        self.classes_ = np.asarray(classes)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


def fit_candidate(spec, X, y, seed=42, jobs=4):
    if spec["family"] == "rf":
        model = RandomForestClassifier(**spec["params"], random_state=seed, n_jobs=jobs)
        return model.fit(X, y)
    if spec["family"] == "xgb":
        from xgboost import XGBClassifier
        encoder = LabelEncoder().fit(y)
        model = XGBClassifier(**spec["params"], random_state=seed, n_jobs=jobs,
                              tree_method="hist", objective="multi:softprob",
                              num_class=len(encoder.classes_), eval_metric="mlogloss")
        model.fit(X, encoder.transform(y))
        return EncodedModel(model, encoder.classes_)
    raise ValueError(f"Unknown model family: {spec['family']}")


def predict_probabilities(model, X, batch_size=50000):
    result = np.empty((len(X), len(model.classes_)), dtype=np.float32)
    for offset in range(0, len(X), batch_size):
        result[offset:offset + batch_size] = model.predict_proba(X[offset:offset + batch_size])
    return result


def training_counts(start, end, indices, eligible_calibration_pixels, min_change_pixels=200):
    pairs, counts = np.unique(np.column_stack([start.ravel()[indices], end.ravel()[indices]]),
                              axis=0, return_counts=True)
    return {(int(a), int(b)): int(n) for (a, b), n in zip(pairs, counts)
            if a != b and min(n, len(indices) - n) >= 10
            and n * eligible_calibration_pixels / len(indices) >= min_change_pixels}


def allocate_probabilities(start, indices, probabilities, classes, counts, train_size,
                           demand_factor=1., minimum_probability=0.):
    """Equivalent demand-first allocator, with optional abstention/demand tests.

    probabilities/indices have identical ordering. Counts are calibration sample
    counts; equal five-year forecast horizons are used by this experiment.
    Default arguments reproduce the shared local allocation behavior.
    """
    if demand_factor < 0 or not 0 <= minimum_probability <= 1 or train_size < 1:
        raise ValueError("Invalid allocation settings")
    if probabilities.shape != (len(indices), len(classes)):
        raise ValueError("Probability matrix and pixel/class indices do not match")
    result = np.full(start.shape, NODATA, dtype=np.int32)
    initial = start.ravel()[indices]
    result.ravel()[indices] = initial
    used = np.zeros(len(indices), dtype=bool)
    columns = {int(c): i for i, c in enumerate(classes)}
    demand = {key: int(round(n * len(indices) / train_size)) for key, n in counts.items()}
    ledger = []
    for key in sorted(demand, key=lambda k: (-demand[k], k)):
        source, destination = key
        requested = int(round(demand[key] * demand_factor))
        if destination not in columns:
            raise ValueError(f"Model cannot score supported destination {destination}")
        scores = probabilities[:, columns[destination]]
        available = np.flatnonzero((initial == source) & ~used & np.isfinite(scores)
                                    & (scores >= minimum_probability))
        selected = available[np.argsort(-scores[available], kind="stable")[:requested]]
        result.ravel()[indices[selected]] = destination
        used[selected] = True
        ledger.append({"from_class": source, "to_class": destination,
                       "requested_pixels": requested, "allocated_pixels": len(selected),
                       "unfulfilled_pixels": requested - len(selected)})
    return result, ledger


def choose_candidate(rows):
    """Validation-only choice, deterministic ties; no final metrics accepted."""
    if not rows or any(row.get("phase") != "validation" for row in rows):
        raise ValueError("Selection requires validation rows only")
    return max(rows, key=lambda r: (r["figure_of_merit_pct"], r["overall_accuracy"], r["name"]))


def audit_labels(maps):
    """Descriptive reversals are not proof of label noise; no labels are edited."""
    years = sorted(maps)
    rows = []
    for a, b, c in zip(years, years[1:], years[2:]):
        x, y, z = maps[a], maps[b], maps[c]
        valid = (x != NODATA) & (y != NODATA) & (z != NODATA)
        changed = valid & (x != y)
        reversed_mask = changed & (z == x)
        rows.append({"years": [a, b, c], "first_interval_changed_pixels": int(changed.sum()),
                     "returned_to_initial_class_pixels": int(reversed_mask.sum()),
                     "reversal_fraction_of_first_changes": float(reversed_mask.sum() / max(changed.sum(), 1))})
    return {"triplets": rows,
            "interpretation": "Reversals may reflect real cycling or classification instability; not verified errors.",
            "label_edits": False,
            "source_evidence": "LC_Type1 raster band names, ~500m grid and local notebook MODIS/IGBP identification; exact collection version not recorded.",
            "source_guide": "https://modis.ornl.gov/documentation/guides/MCD12_User_Guide_V6.pdf",
            "limitations": "No independent reference imagery or source QA/confidence rasters are bundled. Audit does not validate change labels.",
            "excluded_covariates": {"population": "Bundled raster is dated 2020; not used for 2005/2010/2015 starting conditions.",
                                    "roads_POI_protected_areas": "Local snapshots lack verified historical dates.",
                                    "soil_moisture_biomass": "Local files lack sufficient temporal provenance."}}


def run_improvements(data_dir=DEFAULT_DATA, output_dir=None, candidates=None,
                     max_samples=100000, block_size=100, jobs=4, seed=42,
                     min_change_pixels=200, make_plot=True):
    data_dir = Path(data_dir).resolve()
    output = Path(output_dir or data_dir.parent / "outputs" / "improvements").resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory is not empty: {output}")
    specs = CANDIDATES if candidates is None else candidates
    if not specs or specs[0]["name"] != "rf_baseline" or len({s["name"] for s in specs}) != len(specs):
        raise ValueError("Candidates need unique names and rf_baseline first")
    maps, profile = {}, None
    # 2020 labels are deliberately absent until selection and final predictions end.
    for year in (2001, 2005, 2010, 2015):
        maps[year], profile = read_raster(data_dir / f"Guinea_LandCover_{year}.tif", profile, categorical=True)
    if maps[2005].size > 5_000_000:
        raise ValueError("Crop/resample to at most five million cells before running locally")
    covariates, _ = read_raster(data_dir / "Guinea_SRTM_DEM_Slope_Aspect.tif", profile)
    classes = np.unique(maps[2005][maps[2005] != NODATA])
    cov_stack = np.stack(list(covariates.values()), axis=-1)
    pool = sample_training_indices(maps[2005], maps[2010], cov_stack, max_samples, seed)
    train, test, train_blocks, test_blocks = spatial_split(pool, maps[2005].shape, block_size, .3, seed)
    terrain_valid = np.isfinite(cov_stack).all(axis=-1)
    del cov_stack
    if np.any(maps[2015].ravel()[train] == NODATA):
        raise ValueError("Refit labels missing in shared training rows; define a common training sample first")
    validation_eligible = terrain_valid & (maps[2010] != NODATA) & block_mask(maps[2010].shape, test_blocks, block_size)
    validation_indices = np.flatnonzero(validation_eligible)
    validation_mask = validation_eligible & (maps[2015] != NODATA)
    final_eligible = terrain_valid & (maps[2015] != NODATA)
    final_indices = np.flatnonzero(final_eligible)
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "train_indices.npy", train)
    np.save(output / "test_blocks.npy", test_blocks)
    np.save(output / "validation_indices.npy", validation_indices)
    np.save(output / "final_prediction_indices.npy", final_indices)
    (output / "candidates.json").write_text(json.dumps(specs, indent=2))
    audit = audit_labels(maps)
    (output / "label_audit.json").write_text(json.dumps(audit, indent=2))
    calibration_count = int((terrain_valid & (maps[2005] != NODATA) & (maps[2010] != NODATA)).sum())
    selection_counts = training_counts(maps[2005], maps[2010], train, calibration_count, min_change_pixels)
    refit_count = int((terrain_valid & (maps[2010] != NODATA) & (maps[2015] != NODATA)).sum())
    final_counts = training_counts(maps[2010], maps[2015], train, refit_count, min_change_pixels)
    if not selection_counts or not final_counts:
        raise ValueError("No supported transitions; increase sample size or reduce cutoff")
    phase_counts = {"selection": selection_counts, "refit": final_counts}
    for phase, counts in phase_counts.items():
        write_csv(output / f"{phase}_transition_counts.csv", [
            {"from_class": a, "to_class": b, "sampled_pixels": n} for (a, b), n in sorted(counts.items())])

    def matrix(spec, year, indices):
        previous_year = {2005: 2001, 2010: 2005, 2015: 2010}[year]
        return build_features(maps[year], covariates, classes, indices,
                              previous=maps[previous_year], history=spec["history"],
                              multiscale=spec["multiscale"], lag_years=year - previous_year)

    # Feature configurations are ordered so identical matrices can be reused.
    cached_config, X_train, X_valid = None, None, None
    validation_rows = []
    for spec in specs:
        LOGGER.info("VALIDATION candidate %s: fit 2005->2010; score 2010->2015", spec["name"])
        began = perf_counter()
        config = (spec["history"], spec["multiscale"])
        if config != cached_config:
            X_train, feature_names = matrix(spec, 2005, train)
            X_valid, valid_names = matrix(spec, 2010, validation_indices)
            assert feature_names == valid_names
            cached_config = config
        model = fit_candidate(spec, X_train, maps[2010].ravel()[train], seed, jobs)
        probabilities = predict_probabilities(model, X_valid)
        prediction, ledger = allocate_probabilities(maps[2010], validation_indices, probabilities,
                                                    model.classes_, selection_counts, len(train))
        metrics = evaluate(maps[2010], maps[2015], prediction, validation_mask)["summary"]
        validation_rows.append({"phase": "validation", "name": spec["name"], **metrics,
                                "features": len(feature_names), "seconds": perf_counter() - began})
        np.savez_compressed(output / f"validation_scores_{spec['name']}.npz",
                            probabilities=probabilities, classes=model.classes_)
        write_csv(output / f"validation_allocation_{spec['name']}.csv", ledger)
        LOGGER.info("%s validation FoM %.2f%%, accuracy %.2f%%", spec["name"],
                    metrics["figure_of_merit_pct"], metrics["overall_accuracy"] * 100)
        write_csv(output / "validation.csv", validation_rows)
    del X_train, X_valid, probabilities, prediction, model
    selected = choose_candidate(validation_rows)
    selected_spec = next(s for s in specs if s["name"] == selected["name"])
    scores = np.load(output / f"validation_scores_{selected['name']}.npz")
    allocation_rows = []
    for factor in (.5, .75, 1., 1.25):
        for threshold in (0., .1, .2, .3, .4, .5):
            prediction, _ = allocate_probabilities(maps[2010], validation_indices, scores["probabilities"],
                                                  scores["classes"], selection_counts, len(train), factor, threshold)
            metrics = evaluate(maps[2010], maps[2015], prediction, validation_mask)["summary"]
            allocation_rows.append({"phase": "validation", "name": f"factor_{factor}_threshold_{threshold}",
                                    "demand_factor": factor, "minimum_probability": threshold, **metrics})
    chosen_allocation = choose_candidate(allocation_rows)
    write_csv(output / "allocation_validation.csv", allocation_rows)
    selection = {"model": selected_spec, "selection_metric": "validation_change_figure_of_merit",
                 "validation_score": selected["figure_of_merit_pct"],
                 "allocation": {k: chosen_allocation[k] for k in ("demand_factor", "minimum_probability")},
                 "allocation_validation_score": chosen_allocation["figure_of_merit_pct"],
                 "selected_without_2020_labels": True}
    (output / "frozen_selection.json").write_text(json.dumps(selection, indent=2))
    LOGGER.info("FROZEN choice: %s; allocation %s", selected["name"], selection["allocation"])
    del scores, prediction
    cached_config, X_train, X_final = None, None, None
    # Refit every predeclared candidate for transparent ablation reporting; the
    # deployment choice remains the earlier frozen winner, not the best test row.
    final_metadata = {}
    for spec in specs:
        LOGGER.info("FINAL refit %s: 2010->2015; predict 2015->2020", spec["name"])
        began = perf_counter()
        config = (spec["history"], spec["multiscale"])
        if cached_config != config:
            X_train, feature_names = matrix(spec, 2010, train)
            X_final, final_names = matrix(spec, 2015, final_indices)
            assert feature_names == final_names
            cached_config = config
        model = fit_candidate(spec, X_train, maps[2015].ravel()[train], seed, jobs)
        probabilities = predict_probabilities(model, X_final)
        prediction, ledger = allocate_probabilities(maps[2015], final_indices, probabilities,
                                                    model.classes_, final_counts, len(train))
        write_map(output / f"{spec['name']}.tif", prediction, profile)
        write_csv(output / f"final_allocation_{spec['name']}.csv", ledger)
        joblib.dump({"model": model, "feature_names": feature_names, "classes": classes,
                     "spec": spec, "train_sample_sha256": sample_digest(train)},
                    output / f"{spec['name']}.joblib")
        final_metadata[spec["name"]] = {"features": len(feature_names), "seconds": perf_counter() - began}
        if spec["name"] == selected["name"]:
            gated, gated_ledger = allocate_probabilities(maps[2015], final_indices, probabilities,
                model.classes_, final_counts, len(train), **selection["allocation"])
            write_map(output / "selected_with_allocation_change.tif", gated, profile)
            write_csv(output / "selected_with_allocation_change_ledger.csv", gated_ledger)
            del gated
    del X_train, X_final, probabilities, prediction, model
    # Only now access final evaluation labels.
    observed, _ = read_raster(data_dir / "Guinea_LandCover_2020.tif", profile, categorical=True)
    evaluation_mask = final_eligible & (observed != NODATA)
    results = {"no_change": evaluate(maps[2015], observed, maps[2015], evaluation_mask)}
    display_predictions = {"no_change": maps[2015]}
    import rasterio
    for name in [s["name"] for s in specs] + ["selected_with_allocation_change"]:
        with rasterio.open(output / f"{name}.tif") as src:
            prediction = src.read(1)
        results[name] = evaluate(maps[2015], observed, prediction, evaluation_mask)
        if name in ("rf_baseline", selected["name"], "selected_with_allocation_change"):
            display_predictions[name] = prediction
    write_csv(output / "final_comparison.csv", [{"name": name, **r["summary"]} for name, r in results.items()])
    write_map(output / "evaluation_mask.tif", evaluation_mask.astype(np.uint8), profile, nodata=0)
    maps[2020] = observed
    (output / "label_audit_with_final_period.json").write_text(json.dumps(audit_labels(maps), indent=2))
    report = {"experiment": "historical_features_and_models", "selection": selection,
              "selection_fit_period": [2005, 2010], "validation_period": [2010, 2015],
              "refit_period": [2010, 2015], "final_comparison_period": [2015, 2020],
              "train_sample_sha256": sample_digest(train), "train_pixels": len(train),
              "train_blocks": len(train_blocks), "validation_blocks": len(test_blocks),
              "block_size": block_size, "seed": seed, "max_samples": max_samples,
              "classes": classes.tolist(), "validation": validation_rows, "results": results,
              "final_runtime": final_metadata, "label_audit": audit,
              "inputs": [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                         for p in [data_dir / f"Guinea_LandCover_{y}.tif" for y in maps]
                         + [data_dir / "Guinea_SRTM_DEM_Slope_Aspect.tif"]],
              "notes": ["All feature/model candidates share samples, blocks, support and fixed allocation within a phase.",
                        "Allocation modification is a separately selected experiment; it may change total predicted change.",
                        "Candidates and allocation grid are fixed before final labels; final results do not choose the winner.",
                        "2020 has been inspected in prior experiments; this is a retrospective comparison, not a pristine test.",
                        "Historical lag is 2001->2005 (four years) for initial fitting, five years thereafter; pressure trends are annualized.",
                        "No new historically unverified covariates or label cleaning were introduced.",
                        "The default baseline refit reproduces the controlled experiment on bundled data; initial validation uses an earlier training period.",
                        "MODIS label reversals are an audit signal, not independent proof of error or real change."]}
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    if make_plot:
        # Limit display to baseline, validation-selected model, and its allocation variant.
        display_predictions.pop("no_change")
        plot_comparison(maps[2015], observed, display_predictions, evaluation_mask, results,
                        output / "comparison.png", "Improvement experiments — 2015 to 2020 retrospective comparison")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=100000)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    report = run_improvements(args.data_dir, args.output_dir, max_samples=args.max_samples,
                              jobs=args.jobs, make_plot=not args.no_plot)
    print("Frozen selection:", report["selection"])
    for name, result in report["results"].items():
        m = result["summary"]
        print(f"{name:34s} accuracy={m['overall_accuracy']:.2%} FoM={m['figure_of_merit_pct']:.2f}% "
              f"changed={m['predicted_change_pixels']/m['pixels']:.2%}")


if __name__ == "__main__":
    from .improvements import main as canonical_main
    canonical_main()
