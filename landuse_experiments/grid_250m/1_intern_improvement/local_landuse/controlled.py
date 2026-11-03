"""Isolate binary vs multiclass RF scores with identical splitting/allocation.

Uses the original 100-pixel blocks / GroupShuffleSplit(0.3, seed=42), applied
once across the shared sample. The common row pool necessarily replaces the
legacy per-transition balanced samples. No class/sample weighting or random
five-fold CV is used. Both sides use exactly the same allocation function.
"""

import hashlib
import json
import logging
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np

from .comparison import DEFAULT_DATA, evaluate, plot_comparison, read_raster, write_csv, write_map
from .models import (NODATA, DestinationScorer, Settings, allocate_transitions, feature_stack,
                     sample_digest, sample_training_indices, spatial_split, train_multiclass,
                     train_shared_transitions, valid_features)

LOGGER = logging.getLogger(__name__)


def block_mask(shape, blocks, block_size=100):
    rows, cols = np.indices(shape)
    return np.isin((rows // block_size) * 1000 + cols // block_size, blocks)


def run_controlled_comparison(data_dir=DEFAULT_DATA, output_dir=None, years=(2010, 2015, 2020),
                              settings=None, max_samples=100000, min_change_pixels=200,
                              neighborhood=3, exclude_codes=(0, 255), static_predictors=None,
                              block_size=100, test_size=.3, make_plot=True):
    """Fit only shared training blocks; evaluate spatial and later-period maps.

    Shared demands are estimated ONLY from training rows (expanded to each
    prediction footprint), never from held-out destination labels. Raw counts,
    scaling factors, requested/allocated counts and sample identities are saved.
    No multiclass argmax map is used in this controlled experiment.
    """
    if len(years) != 3 or not years[0] < years[1] < years[2]:
        raise ValueError("Supply three strictly increasing years")
    if years[1] - years[0] != years[2] - years[1]:
        raise ValueError("Equal training and forecast intervals are required")
    settings = settings or Settings()
    data_dir = Path(data_dir).resolve()
    output = Path(output_dir or data_dir.parent / "outputs" / "controlled_comparison").resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory is not empty; choose a new directory: {output}")
    paths = [data_dir / f"Guinea_LandCover_{year}.tif" for year in years]
    predictor_paths = ([data_dir / "Guinea_SRTM_DEM_Slope_Aspect.tif"] if static_predictors is None
                       else [Path(p).resolve() for p in static_predictors])
    for path in paths + predictor_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    start, profile = read_raster(paths[0], categorical=True, exclude_codes=exclude_codes)
    if start.size > 5_000_000:
        raise ValueError("Local runner supports up to five million grid cells; crop/resample first")
    middle, _ = read_raster(paths[1], profile, categorical=True, exclude_codes=exclude_codes)
    covariates = {}
    for path in predictor_paths:
        layers, _ = read_raster(path, profile)
        if layers.keys() & covariates.keys():
            raise ValueError("Duplicate predictor names")
        covariates.update(layers)
    # Initial map is known before forecasting; held-out destination labels must
    # not determine the feature vocabulary or any training/model decision.
    classes = np.unique(start[start != NODATA])
    if not len(classes):
        raise ValueError("No valid initial land-cover classes")
    features, names = feature_stack(start, covariates, classes, neighborhood)
    pool = sample_training_indices(start, middle, features, max_samples, settings.random_state)
    train, test, train_blocks, test_blocks = spatial_split(
        pool, start.shape, block_size, test_size, settings.random_state)
    heldout = block_mask(start.shape, test_blocks, block_size)
    output.mkdir(parents=True, exist_ok=True)
    for name, array in {"shared_sample_indices": pool, "train_indices": train, "test_indices": test,
                        "train_blocks": train_blocks, "test_blocks": test_blocks}.items():
        np.save(output / f"{name}.npy", array)
    LOGGER.info("Shared split: %d train pixels in %d blocks; %d test pixels in %d blocks",
                len(train), len(train_blocks), len(test), len(test_blocks))
    began = perf_counter()
    multiclass, training_info = train_multiclass(start, middle, features, settings=settings,
                                               training_indices=train)
    multi_seconds = perf_counter() - began
    began = perf_counter()
    binary, _, training_rows = train_shared_transitions(
        start, middle, features, train, settings=settings, min_change_pixels=min_change_pixels)
    binary_seconds = perf_counter() - began
    if not binary:
        raise ValueError("No supported transitions; use a larger sample or lower the transition cutoff")
    # Shared model support: no transition is available to just one side.
    multi_scorers = {key: DestinationScorer(multiclass, key[1]) for key in binary}
    sample_counts = {(r["from_class"], r["to_class"]): r["sampled_positive_pixels"]
                     for r in training_rows if r["status"] == "trained"}
    fit_audit = {"same_training_rows": all(r["sample_sha256"] == training_info["sample_sha256"]
                                        and r["training_pixels"] == len(train)
                                        for r in training_rows if r["status"] == "trained"),
                 "disjoint_blocks": not bool(np.intersect1d(train_blocks, test_blocks).size),
                 "same_transition_support": set(binary) == set(multi_scorers)}
    if not all(fit_audit.values()):
        raise RuntimeError(f"Controlled comparison audit failed: {fit_audit}")
    common = {"feature_names": names, "landcover_classes": classes.tolist(),
              "neighborhood": neighborhood, "settings": asdict(settings),
              "forecast_interval_years": years[1] - years[0], "train_sample_sha256": sample_digest(train)}
    joblib.dump({"model": multiclass, **common}, output / "multiclass_model.joblib")
    joblib.dump({"models": binary, **common}, output / "binary_models.joblib")
    write_csv(output / "transition_training.csv", training_rows)
    model_sets = {"binary_rf_allocation": binary, "multiclass_rf_allocation": multi_scorers}
    report = {"experiment": "controlled_spatial_split_common_allocation", "years": list(years),
              **common, "training": training_info, "trained_transition_models": len(binary),
              "split": {"block_size_pixels": block_size, "test_fraction_of_blocks": test_size,
                        "sample_pixels": len(pool), "train_pixels": len(train), "test_pixels": len(test),
                        "train_block_count": len(train_blocks), "test_block_count": len(test_blocks),
                        "test_sample_sha256": sample_digest(test)},
              "audit": fit_audit, "fit_seconds": {"multiclass": multi_seconds, "binary": binary_seconds},
              "grid": {"height": start.shape[0], "width": start.shape[1], "crs": str(profile["crs"]),
                       "transform": list(profile["transform"])},
              "inputs": [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                         for p in paths + predictor_paths],
              "minimum_estimated_change_pixels": min_change_pixels,
              "notes": [
                  "Only model formulation differs between the two compared pipelines; both use allocation, not multiclass argmax.",
                  "Original 100x100 block GroupShuffleSplit(0.3) scheme, applied once to a shared uniform pool.",
                  "Shared pool and one-vs-rest binary targets replace legacy per-transition balanced A->B vs A->A samples, as agreed.",
                  "Equal features, row order, weights, RF settings, transition list, demand counts and allocator.",
                  "No extra random five-fold CV or hyperparameter selection using held-out labels.",
                  "Demand = training sample transition count * eligible forecast pixels / training sample size; equal horizons.",
                  "Spatial evaluation uses all eligible held-out-block pixels, and separately the held-out sample rows.",
                  "Temporal evaluation uses 2015 inputs for 2020 prediction; full-footprint scores include previously trained locations.",
                  "A separate temporal held-out-block view is also reported using the same full-footprint predictions.",
                  "The common local allocator retains valid-pixel and single-conversion safeguards; legacy scripts remain unchanged.",
                  "Local terrain and starting-map features, not the old cloud notebook's unavailable 10-layer predictor stack.",
                  "Map-label agreement, not field-verified accuracy; raw class codes are retained."]}
    all_summary = []

    def compare_scope(scope, baseline, predictor_stack, observed, footprint, duration, plot_title):
        scope_dir = output / scope
        scope_dir.mkdir()
        eligible = footprint & valid_features(baseline, predictor_stack)
        eligible_pixels = int(eligible.sum())
        if not eligible_pixels:
            raise ValueError(f"No valid prediction pixels in {scope}")
        masked_baseline = np.where(eligible, baseline, NODATA).astype(np.int32)
        # Demand is frozen once and passed unchanged to both allocation calls.
        demand = {key: int(round(count * eligible_pixels / len(train))) for key, count in sample_counts.items()}
        demand_rows = [{"from_class": key[0], "to_class": key[1], "training_sample_count": sample_counts[key],
                       "expanded_demand_pixels": value} for key, value in sorted(demand.items())]
        write_csv(scope_dir / "shared_demands.csv", demand_rows)
        predictions = {"no_change": masked_baseline}
        allocations = {}
        for name, models in model_sets.items():
            LOGGER.info("%s: allocating with %s", scope, name)
            predictions[name], allocations[name] = allocate_transitions(
                models, demand, masked_baseline, predictor_stack,
                calibration_years=years[1] - years[0], forecast_years=duration)
            write_csv(scope_dir / f"{name}_allocation.csv", allocations[name])
        if allocations["binary_rf_allocation"] != allocations["multiclass_rf_allocation"]:
            raise RuntimeError("Allocation demand/counts differed between models")
        # Invoke AFTER predictions, so temporal labels are not loaded beforehand.
        truth = observed() if callable(observed) else observed
        mask = eligible & (truth != NODATA)
        results = {name: evaluate(baseline, truth, pred, mask) for name, pred in predictions.items()}
        views = {}
        if scope == "spatial_holdout":
            sample_mask = np.zeros(start.shape, dtype=bool)
            sample_mask.ravel()[test] = True
            views["sampled_test_pixels"] = sample_mask & mask
        else:
            views["previously_heldout_blocks"] = heldout & mask
        subreports = {view: {name: evaluate(baseline, truth, pred, m) for name, pred in predictions.items()}
                      for view, m in views.items() if m.any()}
        write_map(scope_dir / "evaluation_mask.tif", mask.astype(np.uint8), profile, nodata=0)
        for name, pred in predictions.items():
            write_map(scope_dir / f"{name}.tif", pred, profile)
        write_csv(scope_dir / "comparison.csv", [{"model": name, **r["summary"]} for name, r in results.items()])
        write_csv(scope_dir / "per_class.csv", [{"model": name, **row} for name, r in results.items()
                                                for row in r["per_class"]])
        for name, r in results.items():
            all_summary.append({"evaluation": scope, "model": name, **r["summary"]})
        for view, reports in subreports.items():
            for name, r in reports.items():
                all_summary.append({"evaluation": f"{scope}/{view}", "model": name, **r["summary"]})
        if make_plot:
            plot_comparison(baseline, truth, predictions, mask, results, scope_dir / "comparison.png", plot_title)
        return {"prediction_eligible_pixels": eligible_pixels, "demand_expansion_factor": eligible_pixels / len(train),
                "identical_allocation_counts": True, "results": results, "additional_views": subreports}

    report["spatial_holdout"] = compare_scope(
        "spatial_holdout", start, features, middle, heldout, years[1] - years[0],
        f"Spatial holdout — {years[0]} to {years[1]} — identical demand and allocation")
    del features
    forecast_features, forecast_names = feature_stack(middle, covariates, classes, neighborhood)
    assert forecast_names == names
    report["temporal_holdout"] = compare_scope(
        "temporal_holdout", middle, forecast_features,
        lambda: read_raster(paths[2], profile, categorical=True, exclude_codes=exclude_codes)[0],
        np.ones(start.shape, dtype=bool), years[2] - years[1],
        f"Temporal holdout — {years[1]} to {years[2]} — identical demand and allocation")
    write_csv(output / "comparison.csv", all_summary)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    LOGGER.info("Saved controlled comparison to %s", output)
    return report
