"""Longer history and pooled-period fitting with original spatial validation.

Fit 2010->2015 on training blocks, select on 2010->2015 held-out blocks,
forecast 2015->2020 without using final labels. Optional earlier-period rows
come from the same training blocks. This differs explicitly from the earlier
2005->2010 fit / 2010->2015 temporal-validation protocol.
"""

import argparse
import hashlib
import json
import logging
from pathlib import Path

import joblib
import numpy as np
from scipy.ndimage import uniform_filter

from .advanced import ChangeModel, apply_decision, landscape_features, tune_decision
from .comparison import DEFAULT_DATA, evaluate, read_raster, write_csv, write_map
from .controlled import block_mask
from .improvements import allocate_probabilities, choose_candidate, training_counts
from .models import NODATA, sample_digest, sample_training_indices, spatial_split

LOGGER = logging.getLogger(__name__)
CANDIDATES = [
    dict(name="recent_two_stage_rf_control", family="rf", structure="two_stage", memory=False, pooled=False, trees=150, depth=16),
    dict(name="memory_two_stage_rf", family="rf", structure="two_stage", memory=True, pooled=False, trees=150, depth=16),
    dict(name="memory_action_rf", family="rf", structure="action", memory=True, pooled=False, trees=150, depth=16),
    dict(name="memory_joint_xgb", family="xgb", structure="joint", memory=True, pooled=False, trees=400, depth=4),
    dict(name="memory_two_stage_xgb", family="xgb", structure="two_stage", memory=True, pooled=False, trees=400, depth=4),
    dict(name="memory_action_xgb", family="xgb", structure="action", memory=True, pooled=False, trees=400, depth=4),
    dict(name="pooled_memory_joint_xgb", family="xgb", structure="joint", memory=True, pooled=True, trees=400, depth=4),
    dict(name="pooled_memory_two_stage_xgb", family="xgb", structure="two_stage", memory=True, pooled=True, trees=400, depth=4),
    dict(name="pooled_memory_action_rf", family="rf", structure="action", memory=True, pooled=True, trees=150, depth=16),
]


def memory_features(current, previous, older, covariates, classes, indices, pairs,
                    recent_lag=5, older_lag=5, memory=True):
    X, names = landscape_features(current, previous, covariates, classes, indices,
                                  lag_years=recent_lag)
    if not memory:
        return X, names
    extra = []

    def add(name, values):
        extra.append(np.asarray(values, dtype=np.float32).ravel()[indices])
        names.append(name)

    valid = (current != NODATA) & (previous != NODATA)
    older_valid = valid & (older != NODATA)
    recent_changed = valid & (current != previous)
    earlier_changed = older_valid & (previous != older)
    add("older_history_available", older_valid)
    add("older_interval_years", np.where(older_valid, older_lag, 0))
    add("changed_in_older_interval", earlier_changed)
    add("changed_in_both_intervals", earlier_changed & recent_changed)
    add("returned_to_older_class", older_valid & recent_changed & (current == older))
    add("stable_across_three_maps", older_valid & ~earlier_changed & ~recent_changed)
    add("observed_change_count", earlier_changed.astype(int) + recent_changed)
    add("observed_stable_duration_lower_bound", np.where(valid & ~recent_changed,
        recent_lag + np.where(older_valid & ~earlier_changed, older_lag, 0), 0))
    for c in classes:
        add(f"older_class_{int(c)}", older_valid & (older == c))
    # Pair vocabulary depends only on starting/past maps, frozen before labels.
    for size in (7, 21, 51):
        support = uniform_filter(valid.astype(np.float32), size, mode="constant")
        for a, b in pairs:
            indicator = (previous == a) & (current == b) & valid
            numerator = uniform_filter(indicator.astype(np.float32), size, mode="constant")
            rate = np.divide(numerator, support, out=np.zeros_like(numerator), where=support > 0) / recent_lag
            add(f"historical_{a}_to_{b}_annual_density_{size}", rate)
    return np.column_stack([X, *extra]).astype(np.float32), names


def run_memory(output_dir, data_dir=DEFAULT_DATA, candidates=None, max_samples=100000,
               block_size=100, jobs=4, seed=42, feature_provider=None, feature_provenance=None,
               expanded_samples=300000):
    output, data_dir = Path(output_dir), Path(data_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new/empty output directory")
    specs = CANDIDATES if candidates is None else candidates
    maps, profile = {}, None
    for year in (2001, 2005, 2010, 2015):
        maps[year], profile = read_raster(data_dir / f"Guinea_LandCover_{year}.tif", profile, categorical=True)
    covariates, _ = read_raster(data_dir / "Guinea_SRTM_DEM_Slope_Aspect.tif", profile)
    cov = np.stack(list(covariates.values()), axis=-1)
    pool = sample_training_indices(maps[2005], maps[2010], cov, max_samples, seed)
    train, _, train_blocks, test_blocks = spatial_split(pool, maps[2010].shape, block_size, .3, seed)
    terrain_valid = np.isfinite(cov).all(axis=-1)
    del cov
    if not np.all(maps[2015].ravel()[train] != NODATA):
        raise ValueError("Training rows missing calibration labels")
    expanded_eligible = (terrain_valid & block_mask(maps[2010].shape, train_blocks, block_size)
        & (maps[2005] != NODATA) & (maps[2010] != NODATA) & (maps[2015] != NODATA))
    extra_pool = np.setdiff1d(np.flatnonzero(expanded_eligible), train)
    extra = np.random.default_rng(seed).choice(extra_pool,
        min(len(extra_pool), max(0, expanded_samples-len(train))), replace=False)
    expanded = np.concatenate([train, extra])
    classes = np.unique(maps[2005][maps[2005] != NODATA])
    unique, counts = np.unique(np.column_stack([maps[2005].ravel()[train], maps[2010].ravel()[train]]), axis=0, return_counts=True)
    pairs = [(int(a), int(b)) for (a, b), n in zip(unique, counts) if a != b and n >= 30]
    validation = np.flatnonzero(terrain_valid & (maps[2010] != NODATA)
                                & block_mask(maps[2010].shape, test_blocks, block_size))
    final_indices = np.flatnonzero(terrain_valid & (maps[2015] != NODATA))
    output.mkdir(parents=True, exist_ok=True)
    for name, values in [("train_indices", train), ("expanded_train_indices", expanded),
                         ("train_blocks", train_blocks), ("test_blocks", test_blocks),
                         ("validation_indices", validation), ("final_indices", final_indices)]:
        np.save(output / f"{name}.npy", values)
    manifest = dict(candidates=specs, pairs=pairs, primary_sample_sha256=sample_digest(train),
        fit_period=[2010, 2015], spatial_validation_period=[2010, 2015], final_period=[2015, 2020],
        pooled_extra_period=[2005, 2010], future_labels_used_for_selection=False)
    manifest["extra_feature_provenance"] = feature_provenance
    manifest["expanded_sample_sha256"] = sample_digest(expanded)
    manifest["expanded_sample_pixels"] = len(expanded)
    manifest["input_history_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [data_dir / f"Guinea_LandCover_{year}.tif" for year in maps]
        + [data_dir / "Guinea_SRTM_DEM_Slope_Aspect.tif"]}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))

    def matrix(year, indices, spec):
        previous_year = {2005: 2001, 2010: 2005, 2015: 2010}[year]
        older_year = {2005: None, 2010: 2001, 2015: 2005}[year]
        older = maps[older_year] if older_year else np.full(maps[year].shape, NODATA)
        X, names = memory_features(maps[year], maps[previous_year], older, covariates, classes,
            indices, pairs, recent_lag=year-previous_year,
            older_lag=previous_year-older_year if older_year else 0, memory=spec["memory"])
        if spec.get("spectral"):
            if feature_provider is None:
                raise ValueError("Spectral candidate requires a dated feature provider")
            additional, additional_names = feature_provider(year, indices)
            if len(additional) != len(indices):
                raise ValueError("Additional features do not match requested pixel rows")
            X = np.column_stack([X, additional])
            names += additional_names
        return X, names

    rows, policies = [], {}
    cached = None
    X_train = X_valid = None
    for spec in specs:
        LOGGER.info("SPATIAL VALIDATION %s", spec["name"])
        config = (spec["memory"], spec["pooled"], bool(spec.get("spectral")), bool(spec.get("expanded")))
        if config != cached:
            X_train = X_valid = None
            used_train = expanded if spec.get("expanded") else train
            X_train, feature_names = matrix(2010, used_train, spec)
            X_valid, valid_names = matrix(2010, validation, spec)
            assert feature_names == valid_names
            source, target, previous = maps[2010].ravel()[used_train], maps[2015].ravel()[used_train], maps[2005].ravel()[used_train]
            if spec["pooled"]:
                earlier, earlier_names = matrix(2005, used_train, spec)
                assert earlier_names == feature_names
                X_train = np.concatenate([X_train, earlier])
                source = np.concatenate([source, maps[2005].ravel()[used_train]])
                target = np.concatenate([target, maps[2010].ravel()[used_train]])
                previous = np.concatenate([previous, maps[2001].ravel()[used_train]])
                del earlier
            cached = config
        full_spec = dict(spec, weight=1., geometry=True, multiscale=True)
        model = ChangeModel().fit(full_spec, X_train, source, target, classes, jobs, seed, previous=previous)
        valid_source = maps[2010].ravel()[validation]
        probabilities = model.predict(X_valid, valid_source, previous=maps[2005].ravel()[validation])
        observed = maps[2015].ravel()[validation]
        mask = observed != NODATA
        policy, best, sweep = tune_decision(probabilities[mask], classes, valid_source[mask], observed[mask])
        policies[spec["name"]] = policy
        rows.append(dict(best, name=spec["name"], training_rows=len(source), features=len(feature_names)))
        write_csv(output / "validation.csv", rows)
        write_csv(output / f"threshold_search_{spec['name']}.csv", sweep)
        prediction = np.full(maps[2010].shape, NODATA, dtype=np.int32)
        prediction.ravel()[validation] = apply_decision(probabilities, classes, valid_source, policy)
        write_map(output / f"validation_{spec['name']}.tif", prediction, profile)
        joblib.dump(dict(model=model, feature_names=feature_names, spec=spec, policy=policy,
                         actual_training_indices_sha256=sample_digest(used_train),
                         manifest=manifest), output / f"{spec['name']}.joblib")
        LOGGER.info("%s spatial FoM %.2f%%", spec["name"], best["figure_of_merit_pct"])
    del X_train, X_valid, model, probabilities
    selected = choose_candidate(rows)["name"]
    (output / "frozen_selection.json").write_text(json.dumps(dict(model=selected, policies=policies,
        basis="2010->2015 held-out spatial blocks", before_2020_labels=True), indent=2))
    LOGGER.info("FROZEN selection %s", selected)
    calibration_count = int((terrain_valid & (maps[2010] != NODATA) & (maps[2015] != NODATA)).sum())
    demands = training_counts(maps[2010], maps[2015], train, calibration_count)
    cached, X_final = None, None
    for spec in specs:
        LOGGER.info("FINAL FORECAST %s", spec["name"])
        config = (spec["memory"], bool(spec.get("spectral")))
        if cached != config:
            X_final = None
            X_final, names = matrix(2015, final_indices, spec)
            cached = config
        bundle = joblib.load(output / f"{spec['name']}.joblib")
        assert names == bundle["feature_names"]
        source = maps[2015].ravel()[final_indices]
        probabilities = bundle["model"].predict(X_final, source, previous=maps[2010].ravel()[final_indices])
        prediction = np.full(maps[2015].shape, NODATA, dtype=np.int32)
        prediction.ravel()[final_indices] = apply_decision(probabilities, classes, source, policies[spec["name"]])
        write_map(output / f"final_{spec['name']}.tif", prediction, profile)
        fixed, ledger = allocate_probabilities(maps[2015], final_indices, probabilities, classes, demands, len(train))
        write_map(output / f"final_{spec['name']}_fixed_demand.tif", fixed, profile)
        write_csv(output / f"allocation_{spec['name']}.csv", ledger)
    del X_final, probabilities, bundle
    observed, _ = read_raster(data_dir / "Guinea_LandCover_2020.tif", profile, categorical=True)
    mask = terrain_valid & (maps[2015] != NODATA) & (observed != NODATA)
    results = {}
    for spec in specs:
        for suffix in ("", "_fixed_demand"):
            name = spec["name"] + suffix
            pred, _ = read_raster(output / f"final_{name}.tif", profile, categorical=True)
            results[name] = evaluate(maps[2015], observed, pred, mask)
    report = dict(manifest=manifest, validation=rows, selected_model=selected,
        results=results, target_fom=40., target_met=results[selected]["summary"]["figure_of_merit_pct"] >= 40,
        evaluation_target_sha256=hashlib.sha256((data_dir / "Guinea_LandCover_2020.tif").read_bytes()).hexdigest(),
        notes=["Spatial validation protocol differs from the preceding earlier-period validation.",
               "Pooled candidates double training rows by adding an earlier period in the same blocks.",
               "2020 labels read only after all predictions; retrospective target was examined in earlier work."])
    (output / "report.json").write_text(json.dumps(report, indent=2))
    write_csv(output / "final_comparison.csv", [dict(name=n, **r["summary"]) for n, r in results.items()])
    write_map(output / "evaluation_mask.tif", mask.astype(np.uint8), profile, nodata=0)
    if data_dir.resolve() == DEFAULT_DATA.resolve():
        from .baseline_comparison import compare_with_original
        compare_with_original(output, data_dir)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    report = run_memory(args.output_dir)
    print("Selected:", report["selected_model"])
    for n, r in report["results"].items():
        if not n.endswith("_fixed_demand"):
            print(n, r["summary"]["figure_of_merit_pct"])


if __name__ == "__main__":
    main()
