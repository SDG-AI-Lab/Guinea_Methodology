"""Summarize completed runs; select exclusively by earlier validation metrics."""

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio

from .comparison import DEFAULT_DATA, evaluate, read_raster, write_csv
from .improvements import choose_candidate


def summarize_runs(run_directories, output_dir, data_dir=DEFAULT_DATA):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records, reference_mask, sample_hash = {}, None, None
    for directory in map(Path, run_directories):
        report = json.loads((directory / "report.json").read_text())
        with rasterio.open(directory / "evaluation_mask.tif") as src:
            mask = src.read(1).astype(bool)
        if reference_mask is not None and not np.array_equal(mask, reference_mask):
            raise ValueError("Runs must use the same final evaluation mask")
        if sample_hash is not None and sample_hash != report["train_sample_sha256"]:
            raise ValueError("Runs must share the primary training pool")
        reference_mask, sample_hash = mask, report["train_sample_sha256"]
        for validation in report["validation"]:
            name = validation["name"]
            # A later run explicitly supersedes a corrected implementation's row.
            records[name] = dict(name=name, run_directory=str(directory), validation=validation,
                policy=report["selection"]["policies"][name],
                final=report["results"][name]["summary"],
                fixed_demand=report["results"][name + "_fixed_demand"]["summary"],
                final_heldout_blocks=report["heldout_results"].get(name, {}).get("summary"))
    selected = choose_candidate([r["validation"] for r in records.values()])["name"]
    final = records[selected]["final"]
    union = final["observed_change_pixels"] + final["false_change_pixels"]
    summary = dict(target_fom=40., selected_model=selected,
        selection_basis="Earlier 2010->2015 validation FoM; no final metric selects the model",
        selected_validation_fom=records[selected]["validation"]["figure_of_merit_pct"],
        selected_final_fom=final["figure_of_merit_pct"], target_met=final["figure_of_merit_pct"] >= 40,
        gap_to_target_percentage_points=max(0, 40 - final["figure_of_merit_pct"]),
        records=records, train_sample_sha256=sample_hash,
        evaluation_pixels=int(reference_mask.sum()),
        diagnostics=dict(
            correct_destination_change_precision=final["correct_change_pixels"] / max(1, final["predicted_change_pixels"]),
            correct_destination_change_recall=final["correct_change_pixels"] / max(1, final["observed_change_pixels"]),
            perfect_destinations_at_current_predicted_locations_fom=(100 *
                (final["correct_change_pixels"] + final["wrong_destination_pixels"]) / max(1, union)),
            interpretation="Oracle diagnostic only: repairing all wrong destinations at the current predicted locations still cannot fix false/missed changes. Not an achieved model score."),
        caveats=["2020 was examined previously; retrospective comparison, not a pristine test.",
                 "Threshold policies change quantity; fixed-demand scores are separately reported.",
                 "Source-specific thresholds are selected on one validation period and may overfit.",
                 "Map-label agreement is not independently validated real-world land-use change."])
    write_csv(output / "comparison.csv", [dict(name=n,
        validation_fom=r["validation"]["figure_of_merit_pct"],
        selected=n == selected, train_pixels=r["validation"]["train_pixels"],
        fixed_demand_fom=r["fixed_demand"]["figure_of_merit_pct"], **r["final"])
        for n, r in records.items()])
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    start, profile = read_raster(Path(data_dir) / "Guinea_LandCover_2015.tif", categorical=True)
    observed, _ = read_raster(Path(data_dir) / "Guinea_LandCover_2020.tif", profile, categorical=True)
    with rasterio.open(Path(records[selected]["run_directory"]) / f"final_{selected}.tif") as src:
        prediction = src.read(1)
    source_rows = []
    for c in np.unique(start[reference_mask]):
        mask = reference_mask & (start == c)
        source_rows.append(dict(source_class=int(c), **evaluate(start, observed, prediction, mask)["summary"]))
    write_csv(output / "selected_errors_by_source.csv", source_rows)
    transition_rows = []
    for a, b in np.unique(np.column_stack([start[reference_mask], observed[reference_mask]]), axis=0):
        if a == b:
            continue
        mask = reference_mask & (start == a) & (observed == b)
        transition_rows.append(dict(source_class=int(a), destination_class=int(b),
            observed_changes=int(mask.sum()), correct_changes=int(np.sum(mask & (prediction == b))),
            missed_changes=int(np.sum(mask & (prediction == a))),
            wrong_destination=int(np.sum(mask & (prediction != b) & (prediction != a)))))
    write_csv(output / "selected_errors_by_transition.csv", sorted(transition_rows,
        key=lambda r: -r["observed_changes"]))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True,
                        help="Completed directories in order; later corrected candidates replace earlier names")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    summary = summarize_runs(args.runs, args.output_dir)
    print("Selected model:", summary["selected_model"])
    print("Final FoM:", summary["selected_final_fom"])
    print("40% target met:", summary["target_met"])


if __name__ == "__main__":
    main()
