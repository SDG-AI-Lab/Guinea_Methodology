"""Choose among fitted models by starting class, using spatial validation only."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .advanced import compact_metrics
from .comparison import DEFAULT_DATA, evaluate, read_raster, write_map
from .improvements import choose_candidate
from .models import NODATA, sample_digest


def select_source_models(source, observed, predictions, minimum_pixels=1000, minimum_changes=100):
    """Guarded class-wise swaps maximize aggregate FoM, not mean class FoM."""
    names = sorted(predictions)
    rows = [dict(phase="validation", name=name, **compact_metrics(source, observed,
        predictions[name], predictions[name] != source)) for name in names]
    default = choose_candidate(rows)["name"]
    classes = np.unique(source)
    counts = {}
    keys = ("correct_change_pixels", "missed_change_pixels", "false_change_pixels", "wrong_destination_pixels")
    for name in names:
        counts[name] = np.array([[compact_metrics(source[source == c], observed[source == c],
            predictions[name][source == c], predictions[name][source == c] != c)[key]
            for key in keys] for c in classes], dtype=np.int64)
    chosen = {int(c): default for c in classes}
    selected_counts = counts[default].copy()

    def score(total):
        hits, missed, false, wrong = total
        union = total.sum()
        observed_count = hits + missed + wrong
        return (100*hits/union if union else 0.,
                (len(source)-observed_count-false+hits)/len(source))

    for _ in range(5):
        updated = False
        for column, c in enumerate(classes):
            mask = source == c
            if mask.sum() < minimum_pixels or np.sum(mask & (source != observed)) < minimum_changes:
                continue
            total = selected_counts.sum(axis=0)
            best_name, best_score = chosen[int(c)], score(total)
            for name in names:
                proposed = score(total-selected_counts[column]+counts[name][column])
                if proposed > best_score:
                    best_name, best_score = name, proposed
            if best_name != chosen[int(c)]:
                chosen[int(c)] = best_name
                selected_counts[column] = counts[best_name][column]
                updated = True
        if not updated:
            break
    prediction = predictions[default].copy()
    for c, name in chosen.items():
        prediction[source == c] = predictions[name][source == c]
    metrics = dict(phase="validation", name="source_ensemble", **compact_metrics(
        source, observed, prediction, prediction != source))
    return dict(default_model=default, source_models=chosen,
        minimum_pixels=minimum_pixels, minimum_changes=minimum_changes), metrics, rows


def run_ensemble(run_directories, output_dir, data_dir=DEFAULT_DATA):
    output, data = Path(output_dir), Path(data_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new/empty ensemble output directory")
    source, profile = read_raster(data / "Guinea_LandCover_2010.tif", categorical=True)
    observed, _ = read_raster(data / "Guinea_LandCover_2015.tif", profile, categorical=True)
    reference, predictions, members = None, {}, {}
    for directory in map(Path, run_directories):
        report = json.loads((directory / "report.json").read_text())
        manifest = report["manifest"]
        validation = np.load(directory / "validation_indices.npy")
        signature = dict(sample=manifest["primary_sample_sha256"], validation=sample_digest(validation),
            fit=manifest["fit_period"], period=manifest["spatial_validation_period"],
            history=manifest["input_history_sha256"])
        if reference is not None and signature != reference:
            raise ValueError("Ensemble members must share training blocks, validation rows and input history")
        reference = signature
        validation = validation[observed.ravel()[validation] != NODATA]
        for row in report["validation"]:
            name = row["name"]
            if name in members:
                raise ValueError("Duplicate member name")
            values, _ = read_raster(directory / f"validation_{name}.tif", profile, categorical=True)
            predictions[name] = values.ravel()[validation]
            members[name] = dict(run_directory=str(directory), validation=row)
    if not members:
        raise ValueError("At least one completed run required")
    policy, ensemble_validation, rows = select_source_models(source.ravel()[validation], observed.ravel()[validation], predictions)
    for row in rows:
        if not np.isclose(row["figure_of_merit_pct"], members[row["name"]]["validation"]["figure_of_merit_pct"]):
            raise ValueError("Recomputed member validation FoM differs from saved report")
    global_row = next(row for row in rows if row["name"] == policy["default_model"])
    selected = choose_candidate([global_row, ensemble_validation])["name"]
    output.mkdir(parents=True, exist_ok=True)
    frozen = dict(policy=policy, selected_model=selected, members=members,
        validation=[global_row, ensemble_validation], protocol=reference,
        selection_basis="Starting-class swaps maximize complete spatial-validation FoM; no final labels enter selection")
    (output / "frozen_selection.json").write_text(json.dumps(frozen, indent=2))
    # Only now read the later starting map and the members' forecast rasters.
    current, _ = read_raster(data / "Guinea_LandCover_2015.tif", profile, categorical=True)
    default = policy["default_model"]
    base_dir = Path(members[default]["run_directory"])
    base, _ = read_raster(base_dir / f"final_{default}.tif", profile, categorical=True)
    prediction = base.copy()
    for name in sorted(set(policy["source_models"].values())):
        directory = Path(members[name]["run_directory"])
        values, _ = read_raster(directory / f"final_{name}.tif", profile, categorical=True)
        for c, chosen in policy["source_models"].items():
            if name == chosen:
                prediction[current == c] = values[current == c]
    write_map(output / "final_source_ensemble.tif", prediction, profile)
    final, _ = read_raster(data / "Guinea_LandCover_2020.tif", profile, categorical=True)
    indices = np.load(base_dir / "final_indices.npy")
    mask = np.zeros(current.shape, dtype=bool)
    mask.ravel()[indices] = True
    mask &= final != NODATA
    results = {default: evaluate(current, final, base, mask),
               "source_ensemble": evaluate(current, final, prediction, mask)}
    report = dict(frozen, results=results, target_fom=40.,
        target_met=results[selected]["summary"]["figure_of_merit_pct"] >= 40,
        evaluation_target_sha256=hashlib.sha256((data / "Guinea_LandCover_2020.tif").read_bytes()).hexdigest(),
        notes=["No class, evaluation-mask, resolution or metric changes.",
               "Sparse validation classes retain the global validation winner.",
               "Member models may have expanded or pooled training; provenance is retained per member.",
               "Ensemble uses threshold-selected member forecasts; no fixed-demand claim is made.",
               "Retrospective benchmark; 2020 was inspected in previous rounds."])
    write_map(output / "evaluation_mask.tif", mask.astype(np.uint8), profile, nodata=0)
    (output / "report.json").write_text(json.dumps(report, indent=2))
    if data.resolve() == DEFAULT_DATA.resolve():
        from .baseline_comparison import compare_with_original
        compare_with_original(output, data)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    report = run_ensemble(args.runs, args.output_dir)
    print("Selected:", report["selected_model"])
    print("Final FoM:", report["results"][report["selected_model"]]["summary"]["figure_of_merit_pct"])


if __name__ == "__main__":
    main()
