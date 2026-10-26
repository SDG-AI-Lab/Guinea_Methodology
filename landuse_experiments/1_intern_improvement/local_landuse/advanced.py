"""Change-focused experiments; select on 2010->2015 before inspecting 2020.

The target remains destination-sensitive FoM on the full valid pixel mask.
No label smoothing, class merging, or final-period threshold optimization.
"""

import argparse
import hashlib
import json
import logging
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import rasterio
from scipy.ndimage import distance_transform_edt, label, uniform_filter
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from .comparison import DEFAULT_DATA, evaluate, plot_comparison, read_raster, write_csv, write_map
from .controlled import block_mask
from .improvements import allocate_probabilities, build_features, choose_candidate, training_counts
from .models import NODATA, sample_digest, sample_training_indices, spatial_split

LOGGER = logging.getLogger(__name__)
TARGET_FOM = 40.
CANDIDATES = [
    dict(name="rf_history_control", family="rf", structure="joint", geometry=False,
         multiscale=False, weight=1., trees=100, depth=20),
    dict(name="xgb_context", family="xgb", structure="joint", geometry=False,
         multiscale=True, weight=1., trees=300, depth=4),
    dict(name="xgb_context_weight4", family="xgb", structure="joint", geometry=False,
         multiscale=True, weight=4., trees=300, depth=4),
    dict(name="xgb_geometry", family="xgb", structure="joint", geometry=True,
         multiscale=True, weight=1., trees=300, depth=4),
    dict(name="xgb_geometry_weight4", family="xgb", structure="joint", geometry=True,
         multiscale=True, weight=4., trees=300, depth=4),
    dict(name="xgb_geometry_deep", family="xgb", structure="joint", geometry=True,
         multiscale=True, weight=1., trees=600, depth=6),
    dict(name="two_stage_xgb", family="xgb", structure="two_stage", geometry=True,
         multiscale=True, weight=1., trees=300, depth=4),
    dict(name="two_stage_xgb_weight4", family="xgb", structure="two_stage", geometry=True,
         multiscale=True, weight=4., trees=300, depth=4),
    dict(name="two_stage_rf", family="rf", structure="two_stage", geometry=True,
         multiscale=True, weight=1., trees=150, depth=16),
    dict(name="source_xgb", family="xgb", structure="source", geometry=True,
         multiscale=True, weight=1., trees=300, depth=3),
    dict(name="ensemble", structure="ensemble", members=["xgb_geometry", "two_stage_xgb"]),
    dict(name="xgb_more_samples", family="xgb", structure="joint", geometry=True,
         multiscale=True, weight=1., trees=300, depth=4, expanded=True),
]


def landscape_features(current, previous, covariates, classes, indices, *,
                       multiscale=True, geometry=True, lag_years=5):
    X, names = build_features(current, covariates, classes, indices, previous=previous,
                             history=True, multiscale=multiscale, lag_years=lag_years)
    if not geometry:
        return X, names
    extra = []

    def add(name, values):
        extra.append(np.asarray(values, dtype=np.float32).ravel()[indices])
        names.append(name)

    rr, cc = np.indices(current.shape)
    add("relative_row", rr / max(1, current.shape[0] - 1))
    add("relative_column", cc / max(1, current.shape[1] - 1))
    for tag, land in (("current", current), ("previous", previous)):
        area = np.zeros(land.shape, dtype=np.float32)
        interior = np.zeros_like(area)
        for c in classes:
            mask = land == c
            components, _ = label(mask)  # Four-connected patches; geography only.
            sizes = np.bincount(components.ravel())
            area[mask] = np.log1p(sizes[components[mask]])
            distance = distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1]
            interior[mask] = distance[mask]
            if tag == "current":
                nearest = (distance_transform_edt(~mask) if mask.any()
                           else np.full(land.shape, max(land.shape)))
                add(f"log_distance_to_class_{int(c)}", np.log1p(nearest))
        add(f"{tag}_log_patch_area_pixels", area)
        add(f"{tag}_log_distance_to_patch_edge", np.log1p(interior))
    for name, values in covariates.items():
        valid = np.isfinite(values)
        filled = np.where(valid, values, 0).astype(np.float32)
        for size in (3, 15):
            support = uniform_filter(valid.astype(np.float32), size, mode="constant")
            mean = np.divide(uniform_filter(filled, size, mode="constant"), support,
                             out=np.zeros_like(filled), where=support > 0)
            square = np.divide(uniform_filter(filled ** 2, size, mode="constant"), support,
                               out=np.zeros_like(filled), where=support > 0)
            add(f"{name}_mean_{size}", mean)
            add(f"{name}_std_{size}", np.sqrt(np.maximum(0, square - mean ** 2)))
    return np.column_stack([X, *extra]).astype(np.float32), names


class Classifier:
    """Encoded classifier with explicit support for constant targets."""
    def fit(self, spec, X, y, weights=None, jobs=4, seed=42):
        self.encoder = LabelEncoder().fit(y)
        self.classes_ = self.encoder.classes_
        self.model = None
        if len(self.classes_) == 1:
            return self
        params = dict(n_estimators=spec["trees"], max_depth=spec["depth"],
                      random_state=seed, n_jobs=jobs)
        if spec["family"] == "rf":
            self.model = RandomForestClassifier(**params, min_samples_split=5,
                                                min_samples_leaf=2)
            # sklearn's weighted bootstrap takes a different RNG path, even for
            # explicit all-one weights. Preserve the original unweighted draws.
            if weights is not None and np.all(np.asarray(weights) == 1):
                weights = None
        elif spec["family"] == "catboost":
            from catboost import CatBoostClassifier
            self.model = CatBoostClassifier(iterations=spec["trees"], depth=spec["depth"],
                learning_rate=.05, l2_leaf_reg=5, random_seed=seed, thread_count=jobs,
                loss_function="Logloss" if len(self.classes_) == 2 else "MultiClass",
                allow_writing_files=False, verbose=False)
        elif spec["family"] == "xgb":
            from xgboost import XGBClassifier
            objective = (dict(objective="binary:logistic", eval_metric="logloss")
                         if len(self.classes_) == 2 else
                         dict(objective="multi:softprob", num_class=len(self.classes_),
                              eval_metric="mlogloss"))
            self.model = XGBClassifier(**params, **objective, tree_method="hist",
                learning_rate=.05, min_child_weight=5, subsample=.8,
                colsample_bytree=.8, reg_lambda=5)
        else:
            raise ValueError(f"Unknown classifier family: {spec['family']}")
        self.model.fit(X, self.encoder.transform(y), sample_weight=weights)
        return self

    def probabilities(self, X, classes):
        raw = self.model.predict_proba(X) if self.model is not None else np.ones((len(X), 1))
        out = np.zeros((len(X), len(classes)), dtype=np.float32)
        for col, c in enumerate(self.classes_):
            if c in classes:
                out[:, np.flatnonzero(classes == c)[0]] = raw[:, col]
        return out


class ChangeModel:
    def fit(self, spec, X, source, target, classes, jobs=4, seed=42, previous=None, sample_weight=None):
        self.spec, self.classes_ = spec, np.asarray(classes)
        if not np.isin(target, self.classes_).all() or not np.isin(source, self.classes_).all():
            raise ValueError("Training classes absent from frozen vocabulary")
        changed = source != target
        weights = np.where(changed, spec["weight"], 1.)
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=float)
            if sample_weight.shape != source.shape or not np.isfinite(sample_weight).all() or np.any(sample_weight <= 0):
                raise ValueError("Sample weights must be aligned, finite and positive")
            weights *= sample_weight
        self.conditional = {}
        if spec["structure"] == "action":
            if previous is None or previous.shape != source.shape:
                raise ValueError("Action model requires aligned previous classes")
            action = np.searchsorted(self.classes_, target) + 2
            action[changed & (target == previous)] = 1
            action[~changed] = 0
            self.main = Classifier().fit(spec, X, action, weights, jobs, seed)
            return self
        self.main = Classifier().fit(spec, X, target, weights, jobs, seed)
        if spec["structure"] == "two_stage":
            self.change = Classifier().fit(spec, X, changed.astype(int), weights, jobs, seed)
            self.destination = (Classifier().fit(spec, X[changed], target[changed],
                                sample_weight[changed] if sample_weight is not None else None, jobs, seed)
                                if changed.any() else None)
        elif spec["structure"] == "source":
            for c in self.classes_:
                mask = source == c
                if mask.sum() >= 200 and changed[mask].sum() >= 20:
                    self.conditional[int(c)] = Classifier().fit(
                        spec, X[mask], target[mask], weights[mask], jobs, seed)
        return self

    def predict(self, X, source, batch_size=50000, previous=None):
        if self.spec["structure"] == "action" and (previous is None or previous.shape != source.shape):
            raise ValueError("Action model requires aligned previous classes")
        result = np.zeros((len(X), len(self.classes_)), dtype=np.float32)
        for begin in range(0, len(X), batch_size):
            xx, aa = X[begin:begin + batch_size], source[begin:begin + batch_size]
            stay_cols = np.searchsorted(self.classes_, aa)
            rr = np.arange(len(xx))
            if self.spec["structure"] == "action":
                tokens = self.main.probabilities(xx, np.arange(len(self.classes_) + 2))
                prob = tokens[:, 2:].copy()
                prob[rr, stay_cols] += tokens[:, 0]
                past = previous[begin:begin + len(xx)]
                past = np.where(np.isin(past, self.classes_), past, aa)
                prob[rr, np.searchsorted(self.classes_, past)] += tokens[:, 1]
                prob /= self.spec["weight"]
                prob[rr, stay_cols] *= self.spec["weight"]
                prob /= np.maximum(prob.sum(axis=1)[:, None], 1e-12)
            elif self.spec["structure"] == "two_stage" and self.destination is not None:
                risk = self.change.probabilities(xx, np.array([0, 1]))[:, 1]
                # Undo the known sampling cost, not an empirical calibration claim.
                risk = risk / np.maximum(risk + self.spec["weight"] * (1 - risk), 1e-12)
                prob = self.destination.probabilities(xx, self.classes_)
                prob[rr, stay_cols] = 0
                total = prob.sum(axis=1)
                risk = np.where(total > 0, risk, 0)
                prob *= np.divide(risk, total, out=np.zeros_like(risk), where=total > 0)[:, None]
                prob[rr, stay_cols] = 1 - risk
            else:
                prob = self.main.probabilities(xx, self.classes_)
                for c, model in self.conditional.items():
                    mask = aa == c
                    if mask.any():
                        prob[mask] = model.probabilities(xx[mask], self.classes_)
                prob /= self.spec["weight"]
                prob[rr, stay_cols] *= self.spec["weight"]
                total = prob.sum(axis=1)
                prob /= np.maximum(total[:, None], 1e-12)
                prob[total == 0, stay_cols[total == 0]] = 1
            result[begin:begin + len(xx)] = prob
        return result


def decision_scores(probabilities, classes, source, rule):
    probabilities = np.asarray(probabilities)
    if probabilities.shape != (len(source), len(classes)) or not np.isfinite(probabilities).all():
        raise ValueError("Invalid probability matrix")
    if not np.isin(source, classes).all():
        raise ValueError("Source class absent from frozen class vocabulary")
    other = probabilities.copy()
    rr, cc = np.arange(len(source)), np.searchsorted(classes, source)
    stay = other[rr, cc].copy()
    other[rr, cc] = -1
    destination = np.asarray(classes)[other.argmax(axis=1)]
    best = np.maximum(other.max(axis=1), 0)
    if rule == "odds":
        score = best / np.maximum(stay, 1e-6)
    elif rule == "joint":
        score = best
    elif rule == "risk":
        score = 1 - stay
    else:
        raise ValueError(f"Unknown decision rule: {rule}")
    score[(destination == source) | (best <= 0)] = -1
    return destination, score


def compact_metrics(source, observed, destination, change):
    actual = observed != source
    hit = int(np.sum(actual & change & (destination == observed)))
    false = int(np.sum(~actual & change))
    wrong = int(np.sum(actual & change & (destination != observed)))
    missed = int(np.sum(actual & ~change))
    total = int(actual.sum()) + false
    return dict(figure_of_merit_pct=100 * hit / total if total else 0.,
                overall_accuracy=float((hit + np.sum(~actual & ~change)) / len(source)),
                predicted_change_pixels=int(change.sum()), observed_change_pixels=int(actual.sum()),
                correct_change_pixels=hit, false_change_pixels=false,
                wrong_destination_pixels=wrong, missed_change_pixels=missed)


def tune_decision(probabilities, classes, source, observed):
    """Validation-only global search plus guarded source-specific thresholds."""
    rows, policies = [], {}
    for rule in ("odds", "joint", "risk"):
        thresholds = (np.geomspace(.01, 5., 65) if rule == "odds" else np.linspace(.01, .95, 65))
        dest, score = decision_scores(probabilities, classes, source, rule)
        candidates = []
        for threshold in thresholds:
            m = compact_metrics(source, observed, dest, score >= threshold)
            name = f"{rule}_{threshold:.8f}"
            candidates.append(dict(phase="validation", name=name, **m))
            policies[name] = dict(rule=rule, threshold=float(threshold), source_thresholds={})
        winner = choose_candidate(candidates)
        rows.extend(candidates)
        policy = dict(policies[winner["name"]])
        policy["source_thresholds"] = {}
        active = score >= policy["threshold"]
        actual = observed != source
        # Coordinate improvement on contributions to the SAME full-mask FoM.
        # Rare source classes retain the global threshold.
        for c in classes:
            mask = source == c
            if mask.sum() < 1000 or actual[mask].sum() < 100:
                continue
            outside = ~mask
            outside_hits = int(np.sum(active & actual & (dest == observed) & outside))
            outside_false = int(np.sum(active & ~actual & outside))
            best_value = compact_metrics(source, observed, dest, active)["figure_of_merit_pct"]
            best_threshold = policy["threshold"]
            for threshold in thresholds:
                local = score[mask] >= threshold
                hit = outside_hits + int(np.sum(local & actual[mask] & (dest[mask] == observed[mask])))
                false = outside_false + int(np.sum(local & ~actual[mask]))
                value = 100 * hit / max(1, int(actual.sum()) + false)
                if value > best_value + 1e-10:
                    best_value, best_threshold = value, float(threshold)
            policy["source_thresholds"][str(int(c))] = float(best_threshold)
            active[mask] = score[mask] >= best_threshold
        name = f"{rule}_source_thresholds"
        policies[name] = policy
        rows.append(dict(phase="validation", name=name, **compact_metrics(source, observed, dest, active)))
    best = choose_candidate(rows)
    return policies[best["name"]], best, rows


def apply_decision(probabilities, classes, source, policy):
    destination, score = decision_scores(probabilities, classes, source, policy["rule"])
    threshold = np.full(len(source), policy["threshold"], dtype=np.float64)
    for c, value in policy["source_thresholds"].items():
        threshold[source == int(c)] = value
    return np.where(score >= threshold, destination, source).astype(np.int32)


def run_advanced(data_dir=DEFAULT_DATA, output_dir=None, candidates=None, max_samples=100000,
                 expanded_samples=300000, block_size=100, jobs=4, seed=42, make_plot=True,
                 feature_provider=None, feature_provenance=None):
    data_dir, output = Path(data_dir).resolve(), Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new/empty output directory")
    specs = CANDIDATES if candidates is None else candidates
    names = [s["name"] for s in specs]
    if not names or len(set(names)) != len(names):
        raise ValueError("Candidates must have unique names")
    available = set()
    for spec in specs:
        if spec["structure"] == "ensemble" and not set(spec["members"]).issubset(available):
            raise ValueError("Ensemble members must appear first")
        available.add(spec["name"])
    maps, profile = {}, None
    for year in (2001, 2005, 2010, 2015):
        maps[year], profile = read_raster(data_dir / f"Guinea_LandCover_{year}.tif", profile, categorical=True)
    covariates, _ = read_raster(data_dir / "Guinea_SRTM_DEM_Slope_Aspect.tif", profile)
    cov_stack = np.stack(list(covariates.values()), axis=-1)
    valid_terrain = np.isfinite(cov_stack).all(axis=-1)
    pool = sample_training_indices(maps[2005], maps[2010], cov_stack, max_samples, seed)
    train, _, train_blocks, test_blocks = spatial_split(pool, maps[2005].shape, block_size, .3, seed)
    del cov_stack
    classes = np.unique(maps[2005][maps[2005] != NODATA])
    eligible_train = (valid_terrain & block_mask(maps[2005].shape, train_blocks, block_size)
                      & (maps[2005] != NODATA) & (maps[2010] != NODATA) & (maps[2015] != NODATA))
    if not eligible_train.ravel()[train].all():
        raise ValueError("Shared training rows lack labels; define a common training mask")
    # This separately labeled experiment expands sample count only within training blocks.
    extra_pool = np.setdiff1d(np.flatnonzero(eligible_train), train)
    extra = np.random.default_rng(seed).choice(extra_pool,
        min(len(extra_pool), max(0, expanded_samples - len(train))), replace=False)
    expanded = np.concatenate([train, extra])
    validation_indices = np.flatnonzero(valid_terrain & (maps[2010] != NODATA)
                                        & block_mask(maps[2010].shape, test_blocks, block_size))
    final_indices = np.flatnonzero(valid_terrain & (maps[2015] != NODATA))
    output.mkdir(parents=True, exist_ok=True)
    for name, array in [("train_indices", train), ("expanded_train_indices", expanded),
                        ("train_blocks", train_blocks), ("test_blocks", test_blocks),
                        ("validation_indices", validation_indices), ("final_indices", final_indices)]:
        np.save(output / f"{name}.npy", array)
    (output / "candidates.json").write_text(json.dumps(specs, indent=2))
    (output / "extra_feature_provenance.json").write_text(json.dumps(feature_provenance, indent=2))
    ensemble_members = {n for s in specs if s["structure"] == "ensemble" for n in s["members"]}
    choices, validation_rows, runtimes = {}, [], {}

    def matrix(spec, year, indices):
        earlier = {2005: 2001, 2010: 2005, 2015: 2010}[year]
        X, names = landscape_features(maps[year], maps[earlier], covariates, classes, indices,
            multiscale=spec["multiscale"], geometry=spec["geometry"], lag_years=year - earlier)
        if spec.get("spectral"):
            if feature_provider is None:
                raise ValueError("Satellite candidate requires a dated feature provider")
            extra, extra_names = feature_provider(year, indices)
            if len(extra) != len(indices):
                raise ValueError("Extra feature rows differ from the requested sample")
            X, names = np.column_stack([X, extra]), names + extra_names
        return X, names

    # Finish all validation and freeze model/policy before any final refit.
    for phase, fit_start, fit_end, forecast_start, prediction_indices in [
            ("validation", 2005, 2010, 2010, validation_indices),
            ("final", 2010, 2015, 2015, final_indices)]:
        calibration_pixels = int((valid_terrain & (maps[fit_start] != NODATA)
                                  & (maps[fit_end] != NODATA)).sum())
        counts = training_counts(maps[fit_start], maps[fit_end], train, calibration_pixels)
        write_csv(output / f"{phase}_transition_counts.csv", [
            dict(from_class=a, to_class=b, sample_count=n) for (a, b), n in sorted(counts.items())])
        source = maps[forecast_start].ravel()[prediction_indices]
        if not np.isin(source, classes).all():
            raise ValueError("Forecast source has an unseen class")
        config = None
        adaptation_cache = {}
        X_train = X_predict = None
        for spec in specs:
            name, began = spec["name"], perf_counter()
            LOGGER.info("%s %s", phase.upper(), name)
            if spec["structure"] == "ensemble":
                probabilities = np.zeros((len(source), len(classes)), dtype=np.float32)
                for member in spec["members"]:
                    probabilities += np.load(output / f"{phase}_probabilities_{member}.npy") / len(spec["members"])
                feature_names, used_train = ["mean_member_probabilities"], train
                config = None
            else:
                used_train = expanded if spec.get("expanded") else train
                next_config = (spec["geometry"], spec["multiscale"], bool(spec.get("expanded")), bool(spec.get("spectral")))
                if config != next_config:
                    X_train = X_predict = None
                    X_train, feature_names = matrix(spec, fit_start, used_train)
                    X_predict, predict_names = matrix(spec, forecast_start, prediction_indices)
                    assert feature_names == predict_names
                    config = next_config
                adaptation_weights, adaptation_report = None, None
                if spec.get("adapt"):
                    from .domain_adaptation import estimate_density_weights
                    if next_config not in adaptation_cache:
                        X_domain, domain_names = matrix(spec, forecast_start, used_train)
                        assert domain_names == feature_names
                        groups = (used_train // maps[fit_start].shape[1] // block_size) * 1000 + (used_train % maps[fit_start].shape[1] // block_size)
                        adaptation_cache[next_config] = estimate_density_weights(
                            X_train, X_domain, feature_names, groups, jobs, seed)
                        del X_domain
                    adaptation_weights, adaptation_report = adaptation_cache[next_config]
                    shrink = float(spec.get("adapt_shrink", 1.))
                    adaptation_weights = (1-shrink) + shrink*adaptation_weights
                    np.save(output / f"{phase}_{name}_sample_weights.npy", adaptation_weights)
                    (output / f"{phase}_{name}_adaptation.json").write_text(json.dumps(
                        dict(adaptation_report, shrink=shrink, fit_start=fit_start, forecast_start=forecast_start), indent=2))
                model = ChangeModel().fit(spec, X_train, maps[fit_start].ravel()[used_train],
                    maps[fit_end].ravel()[used_train], classes, jobs, seed,
                    previous=maps[{2005: 2001, 2010: 2005}[fit_start]].ravel()[used_train],
                    sample_weight=adaptation_weights)
                probabilities = model.predict(X_predict, source,
                    previous=maps[{2010: 2005, 2015: 2010}[forecast_start]].ravel()[prediction_indices])
                if phase == "final":
                    joblib.dump(dict(model=model, feature_names=feature_names,
                                     spec=spec, policy=choices[name], feature_provenance=feature_provenance,
                                     adaptation=adaptation_report,
                                     train_sample_sha256=sample_digest(used_train)), output / f"{name}.joblib")
                del model
            if name in ensemble_members:
                np.save(output / f"{phase}_probabilities_{name}.npy", probabilities)
            if phase == "validation":
                observed = maps[2015].ravel()[prediction_indices]
                mask = observed != NODATA
                policy, best, sweep = tune_decision(probabilities[mask], classes, source[mask], observed[mask])
                choices[name] = policy
                write_csv(output / f"threshold_search_{name}.csv", sweep)
                validation_rows.append(dict(phase="validation", name=name,
                    **{k: v for k, v in best.items() if k not in ("phase", "name")},
                    train_pixels=len(used_train), features=len(feature_names)))
                write_csv(output / "validation.csv", validation_rows)
                LOGGER.info("%s validation FoM %.2f%%", name, best["figure_of_merit_pct"])
            predicted = np.full(maps[forecast_start].shape, NODATA, dtype=np.int32)
            predicted.ravel()[prediction_indices] = apply_decision(probabilities, classes, source, choices[name])
            write_map(output / f"{phase}_{name}.tif", predicted, profile)
            # Same-demand view isolates score improvements from the new decision policy.
            fixed, ledger = allocate_probabilities(maps[forecast_start], prediction_indices,
                                                   probabilities, classes, counts, len(train))
            write_map(output / f"{phase}_{name}_fixed_demand.tif", fixed, profile)
            write_csv(output / f"{phase}_{name}_fixed_demand.csv", ledger)
            runtimes[f"{phase}_{name}"] = perf_counter() - began
            del probabilities, predicted, fixed
        del X_train, X_predict
        if phase == "validation":
            selected = choose_candidate(validation_rows)["name"]
            frozen = dict(selected_model=selected, policies=choices,
                          selected_validation_fom=choose_candidate(validation_rows)["figure_of_merit_pct"],
                          target_fom=TARGET_FOM, selected_before_final_labels=True)
            (output / "frozen_selection.json").write_text(json.dumps(frozen, indent=2))
            LOGGER.info("FROZEN selection: %s; FoM %.2f%%", selected, frozen["selected_validation_fom"])

    # Access the final target only after every forecast and policy has been saved.
    observed, _ = read_raster(data_dir / "Guinea_LandCover_2020.tif", profile, categorical=True)
    mask = np.zeros(observed.shape, dtype=bool)
    mask.ravel()[final_indices] = True
    mask &= observed != NODATA
    heldout_mask = mask & block_mask(mask.shape, test_blocks, block_size)
    results, heldout, predictions = {}, {}, {}
    for spec in specs:
        for suffix in ("", "_fixed_demand"):
            name = spec["name"] + suffix
            with rasterio.open(output / f"final_{name}.tif") as src:
                pred = src.read(1)
            results[name] = evaluate(maps[2015], observed, pred, mask)
            if heldout_mask.any():
                heldout[name] = evaluate(maps[2015], observed, pred, heldout_mask)
            if name in (selected, selected + "_fixed_demand"):
                predictions[name] = pred
    for name, result in [("no_change", evaluate(maps[2015], observed, maps[2015], mask))]:
        results[name] = result
    write_csv(output / "final_comparison.csv", [dict(name=n, **r["summary"]) for n, r in results.items()])
    write_csv(output / "final_heldout_blocks.csv", [dict(name=n, **r["summary"]) for n, r in heldout.items()])
    write_map(output / "evaluation_mask.tif", mask.astype(np.uint8), profile, nodata=0)
    final_fom = results[selected]["summary"]["figure_of_merit_pct"]
    report = dict(selection=frozen, validation=validation_rows, results=results,
        heldout_results=heldout, target_met_by_selected_model=final_fom >= TARGET_FOM,
        train_pixels=len(train), expanded_train_pixels=len(expanded), block_size=block_size,
        train_sample_sha256=sample_digest(train), expanded_sample_sha256=sample_digest(expanded),
        train_blocks=len(train_blocks), validation_blocks=len(test_blocks), classes=classes.tolist(),
        seed=seed, runtimes=runtimes, extra_feature_provenance=feature_provenance,
        periods=dict(initial_fit=[2005, 2010], validation=[2010, 2015],
                                                refit=[2010, 2015], final=[2015, 2020]),
        inputs=[dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                for p in [data_dir / f"Guinea_LandCover_{y}.tif" for y in (2001, 2005, 2010, 2015, 2020)]
                + [data_dir / "Guinea_SRTM_DEM_Slope_Aspect.tif"]],
        notes=["Target metric/mask/classes/labels unchanged. No destination-label edits or spatial aggregation.",
               "Model and thresholds selected on earlier validation, never on 2020 labels in this run.",
               "2020 was inspected in earlier experiments; retrospective evaluation is not pristine.",
               "Source-specific threshold search may overfit the one validation period; held-out period decides transfer.",
               "Two-stage destination fitting uses only changed rows of the shared pool; source experts use source subsets.",
               "The expanded-sample candidate changes training size explicitly within the same training blocks.",
               "Final full footprint includes training-block locations at a later date; separate held-out-block view saved.",
               "History lag is four years at initial fit and five years thereafter; rates are annualized.",
               "Weights emphasize changed rows; inverse weight adjustment is not empirical probability calibration.",
               "Default decision policies replace historical quotas; fixed-demand views retain previous allocation."])
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    if data_dir == DEFAULT_DATA.resolve():
        from .baseline_comparison import compare_with_original
        compare_with_original(output, data_dir)
    if make_plot:
        plot_comparison(maps[2015], observed, predictions, mask, results, output / "comparison.png",
                        "Validation-selected model: decision rule versus fixed demand")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    report = run_advanced(args.data_dir, args.output_dir, jobs=args.jobs)
    print("Selected:", report["selection"]["selected_model"])
    for name, result in report["results"].items():
        if not name.endswith("_fixed_demand"):
            print(name, result["summary"]["figure_of_merit_pct"])
    print("Selected model meets 40% target:", report["target_met_by_selected_model"])


if __name__ == "__main__":
    # Use canonical module identities in joblib, rather than __main__.ChangeModel.
    from .advanced import main as canonical_main
    canonical_main()
