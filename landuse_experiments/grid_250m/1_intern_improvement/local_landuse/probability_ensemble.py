"""Average fitted models' probability surfaces instead of choosing one of them.

The metric is bounded by false alarms, so what matters is how sharply the
ranking separates the pixels that flip from the ones that do not. Selecting a
single model per starting class keeps one ranking; averaging decorrelated
probability surfaces produces a new one. Members are chosen greedily with
replacement on 2010->2015 held-out spatial blocks, and both the membership and
the decision policy are frozen to disk before any 2020 label is read.
"""

import argparse
import hashlib
import json
import logging
from pathlib import Path

import joblib
import numpy as np

from .advanced import apply_decision, compact_metrics, tune_decision
from .comparison import DEFAULT_DATA, evaluate, read_raster, write_csv, write_map
from .improvements import choose_candidate
from .models import NODATA, sample_digest
from .temporal_memory import memory_features

LOGGER = logging.getLogger(__name__)


def build_provider(kind, phenology_dir, spectral_dir, reference, fire_dir=None):
    """Rebuild the dated feature provider a finished run was fitted with."""
    if kind not in ("phenology", "context", "spectral", "fused", "typicality", "fire"):
        raise ValueError(f"Unknown provider kind: {kind}")
    from .phenology_models import HistoricalPhenology
    from .spectral_models import HistoricalSpectra
    from .vegetation_context import VegetationContext
    if kind == "spectral":
        return HistoricalSpectra(spectral_dir, reference)
    phenology = HistoricalPhenology(phenology_dir, reference)
    if kind == "phenology":
        return phenology
    spectra = HistoricalSpectra(spectral_dir, reference) if kind in ("fused", "typicality", "fire") else None
    context = VegetationContext(phenology, spectra)
    if kind == "typicality":
        from .class_typicality import ClassTypicality
        return ClassTypicality(context)
    if kind == "fire":
        from .fire_models import FireContext, HistoricalFire
        return FireContext(context, HistoricalFire(fire_dir, reference))
    return context


def greedy_selection(probabilities, classes, source, observed, rounds=12):
    """Ensemble selection with replacement; each step must improve validation FoM."""
    names = sorted(probabilities)
    if not names:
        raise ValueError("Greedy selection needs at least one member surface")
    total = np.zeros_like(probabilities[names[0]])
    counts, history, best_value, best_policy = {}, [], -1., None
    for step in range(rounds):
        step_best = None
        for name in names:
            candidate = (total * len(history) + probabilities[name]) / (len(history) + 1)
            policy, row, _ = tune_decision(candidate, classes, source, observed)
            value = (row["figure_of_merit_pct"], row["overall_accuracy"])
            if step_best is None or value > step_best[0]:
                step_best = (value, name, candidate, policy, row)
        value, name, candidate, policy, row = step_best
        if value[0] <= best_value + 1e-9:
            LOGGER.info("Greedy selection stopped at %d members", len(history))
            break
        total, best_value, best_policy = candidate, value[0], policy
        counts[name] = counts.get(name, 0) + 1
        history.append(dict(step=step + 1, added=name, members=len(history) + 1,
                            figure_of_merit_pct=row["figure_of_merit_pct"],
                            overall_accuracy=row["overall_accuracy"]))
        LOGGER.info("Step %d added %s -> validation FoM %.2f%%", step + 1, name, best_value)
    if not history:
        raise ValueError("Greedy selection produced no members")
    weights = {name: count / sum(counts.values()) for name, count in counts.items()}
    return weights, best_policy, history, total


def member_probabilities(run, kind, indices, year, maps, covariates, classes, pairs,
                         phenology_dir, spectral_dir, reference, previous_year, fire_dir=None):
    """Rebuild one run's features once and predict every model fitted on them."""
    directory = Path(run)
    specs = json.loads((directory / "manifest.json").read_text())["candidates"]
    provider = None
    out, cached, X, names = {}, None, None, None
    source = maps[year].ravel()[indices]
    previous = maps[previous_year].ravel()[indices]
    for spec in specs:
        bundle_path = directory / f"{spec['name']}.joblib"
        if not bundle_path.exists():
            LOGGER.warning("Skipping %s: no saved bundle", spec["name"])
            continue
        config = (spec["memory"], bool(spec.get("spectral")))
        if config != cached:
            X = None
            older_year = {2010: 2001, 2015: 2005}[year]
            X, names = memory_features(maps[year], maps[previous_year], maps[older_year],
                covariates, classes, indices, pairs, recent_lag=year - previous_year,
                older_lag=previous_year - older_year, memory=spec["memory"])
            if spec.get("spectral"):
                if provider is None:
                    provider = build_provider(kind, phenology_dir, spectral_dir, reference, fire_dir)
                extra, extra_names = provider(year, indices)
                X, names = np.column_stack([X, extra]), names + extra_names
            cached = config
        bundle = joblib.load(bundle_path)
        if names != bundle["feature_names"]:
            raise ValueError(f"Rebuilt features differ from those {spec['name']} was fitted on")
        out[spec["name"]] = bundle["model"].predict(X, source, previous=previous)
        LOGGER.info("%s %d: predicted %s", directory.name, year, spec["name"])
        del bundle
    return out


def run_probability_ensemble(runs, output_dir, phenology_dir, spectral_dir, data_dir=DEFAULT_DATA,
                             rounds=12, seed=42, fire_dir=None):
    output, data = Path(output_dir), Path(data_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new/empty ensemble output directory")
    maps, profile = {}, None
    for year in (2001, 2005, 2010, 2015):
        maps[year], profile = read_raster(data / f"Guinea_LandCover_{year}.tif", profile, categorical=True)
    covariates, _ = read_raster(data / "Guinea_SRTM_DEM_Slope_Aspect.tif", profile)
    classes = np.unique(maps[2005][maps[2005] != NODATA])
    reference = data / "Guinea_LandCover_2010.tif"
    signature, pairs, validation, final_indices = None, None, None, None
    for run, _ in runs:
        manifest = json.loads((Path(run) / "manifest.json").read_text())
        current = dict(sample=manifest["primary_sample_sha256"], fit=manifest["fit_period"],
                       final=manifest["final_period"], history=manifest["input_history_sha256"])
        if signature is not None and current != signature:
            raise ValueError("Ensemble members must share the training sample and input history")
        signature, pairs = current, [tuple(p) for p in manifest["pairs"]]
        indices = (np.load(Path(run) / "validation_indices.npy"), np.load(Path(run) / "final_indices.npy"))
        if validation is not None and not (np.array_equal(validation, indices[0])
                                           and np.array_equal(final_indices, indices[1])):
            raise ValueError("Ensemble members must share validation and forecast rows")
        validation, final_indices = indices
    output.mkdir(parents=True, exist_ok=True)
    valid_source = maps[2010].ravel()[validation]
    observed = maps[2015].ravel()[validation]
    keep = observed != NODATA
    surfaces, provenance = {}, {}
    for run, kind in runs:
        for name, values in member_probabilities(run, kind, validation, 2010, maps, covariates,
                classes, pairs, phenology_dir, spectral_dir, reference, 2005, fire_dir).items():
            if name in surfaces:
                raise ValueError(f"Duplicate member name across runs: {name}")
            surfaces[name] = values[keep]
            provenance[name] = dict(run_directory=str(run), provider=kind)
    LOGGER.info("Rebuilt %d member probability surfaces", len(surfaces))
    singles = []
    for name, values in surfaces.items():
        row = tune_decision(values, classes, valid_source[keep], observed[keep])[1]
        singles.append(dict(row, phase="validation", name=name))
    write_csv(output / "member_validation.csv", singles)
    weights, policy, history, averaged = greedy_selection(surfaces, classes,
        valid_source[keep], observed[keep], rounds=rounds)
    decided = apply_decision(averaged, classes, valid_source[keep], policy)
    ensemble_row = dict(phase="validation", name="probability_ensemble",
        **compact_metrics(valid_source[keep], observed[keep], decided, decided != valid_source[keep]))
    best_single = choose_candidate(singles)
    selected = choose_candidate([ensemble_row, best_single])["name"]
    frozen = dict(weights=weights, policy=policy, greedy_history=history, members=provenance,
        best_single_member=best_single["name"], selected_model=selected,
        validation=[ensemble_row, best_single], protocol=signature,
        selection_basis="Greedy averaging with replacement on 2010->2015 held-out blocks; no 2020 label read")
    (output / "frozen_selection.json").write_text(json.dumps(frozen, indent=2, default=float))
    LOGGER.info("FROZEN %d distinct members, validation FoM %.2f%%",
                len(weights), ensemble_row["figure_of_merit_pct"])
    del surfaces, averaged
    # Only now build the forecast period and read the 2020 labels.
    source = maps[2015].ravel()[final_indices]
    total = np.zeros((len(final_indices), len(classes)), dtype=np.float32)
    single = None
    for run, kind in runs:
        for name, values in member_probabilities(run, kind, final_indices, 2015, maps, covariates,
                classes, pairs, phenology_dir, spectral_dir, reference, 2010, fire_dir).items():
            if name in weights:
                total += weights[name] * values
            if name == best_single["name"]:
                single = values
    prediction = np.full(maps[2015].shape, NODATA, dtype=np.int32)
    prediction.ravel()[final_indices] = apply_decision(total, classes, source, policy)
    write_map(output / "final_probability_ensemble.tif", prediction, profile)
    single_map = np.full(maps[2015].shape, NODATA, dtype=np.int32)
    single_map.ravel()[final_indices] = apply_decision(single, classes, source,
        json.loads((Path(provenance[best_single["name"]]["run_directory"]) / "frozen_selection.json")
                   .read_text())["policies"][best_single["name"]])
    write_map(output / f"final_{best_single['name']}.tif", single_map, profile)
    truth, _ = read_raster(data / "Guinea_LandCover_2020.tif", profile, categorical=True)
    mask = np.zeros(maps[2015].shape, dtype=bool)
    mask.ravel()[final_indices] = True
    mask &= truth != NODATA
    results = {"probability_ensemble": evaluate(maps[2015], truth, prediction, mask),
               best_single["name"]: evaluate(maps[2015], truth, single_map, mask)}
    report = dict(frozen, results=results, target_fom=40.,
        target_met=results[selected]["summary"]["figure_of_merit_pct"] >= 40,
        evaluation_target_sha256=hashlib.sha256((data / "Guinea_LandCover_2020.tif").read_bytes()).hexdigest(),
        notes=["Members keep their own training design; averaging is not an algorithm-only comparison.",
               "Weights are greedy selection counts on validation, never fitted to 2020.",
               "The single-model row uses that model's own frozen policy from its original run.",
               "Retrospective benchmark; 2020 was inspected in previous rounds."])
    write_map(output / "evaluation_mask.tif", mask.astype(np.uint8), profile, nodata=0)
    (output / "report.json").write_text(json.dumps(report, indent=2, default=float))
    if data.resolve() == DEFAULT_DATA.resolve():
        from .baseline_comparison import compare_with_original
        compare_with_original(output, data)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, metavar="DIRECTORY=KIND",
                        help="Finished run directory and the provider it was fitted with")
    parser.add_argument("--phenology-dir", required=True)
    parser.add_argument("--spectral-dir", required=True)
    parser.add_argument("--fire-dir")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rounds", type=int, default=12)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    runs = []
    for entry in args.run:
        directory, _, kind = entry.partition("=")
        if not kind:
            raise SystemExit("Each --run needs DIRECTORY=KIND")
        runs.append((directory, kind))
    report = run_probability_ensemble(runs, args.output_dir, args.phenology_dir, args.spectral_dir,
                                      rounds=args.rounds, fire_dir=args.fire_dir)
    print("Selected:", report["selected_model"])
    print("Final FoM:", report["results"][report["selected_model"]]["summary"]["figure_of_merit_pct"])


if __name__ == "__main__":
    main()
