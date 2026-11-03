"""Compare completed memory/satellite runs under their shared spatial protocol."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio

from .comparison import write_csv
from .improvements import choose_candidate
from .models import sample_digest


def summarize_runs(run_directories, output_dir):
    records, reference, reference_mask = {}, None, None
    for directory in map(Path, run_directories):
        report = json.loads((directory / "report.json").read_text())
        manifest = report["manifest"]
        signature = {key: manifest[key] for key in (
            "primary_sample_sha256", "fit_period", "spatial_validation_period", "final_period")}
        signature["evaluation_target_sha256"] = report["evaluation_target_sha256"]
        signature["validation_indices_sha256"] = sample_digest(np.load(directory / "validation_indices.npy"))
        if sample_digest(np.load(directory / "train_indices.npy")) != signature["primary_sample_sha256"]:
            raise ValueError("Saved training sample differs from its manifest")
        with rasterio.open(directory / "evaluation_mask.tif") as src:
            mask = src.read(1).astype(bool)
            signature["grid"] = (str(src.crs), tuple(src.transform), src.shape)
        if reference is not None and (signature != reference or not np.array_equal(mask, reference_mask)):
            raise ValueError("Runs must share the sample, periods, target and complete evaluation grid/mask")
        reference, reference_mask = signature, mask
        for row in report["validation"]:
            name = row["name"]
            if name in records:
                raise ValueError(f"Duplicate model name: {name}")
            final = report["results"][name]["summary"]
            if final["pixels"] != int(mask.sum()):
                raise ValueError("Final metric count differs from the saved evaluation mask")
            records[name] = dict(name=name, run_directory=str(directory), validation=row,
                final=final, fixed_demand=report["results"][name+"_fixed_demand"]["summary"])
    if not records:
        raise ValueError("At least one completed run is required")
    selected = choose_candidate([r["validation"] for r in records.values()])["name"]
    final = records[selected]["final"]
    summary = dict(selected_model=selected, selected_final_fom=final["figure_of_merit_pct"],
        selected_validation_fom=records[selected]["validation"]["figure_of_merit_pct"],
        target_fom=40., target_met=final["figure_of_merit_pct"] >= 40,
        evaluation_pixels=int(reference_mask.sum()), records=records,
        selection_basis="Maximum 2010->2015 held-out spatial-block validation FoM; final scores do not select",
        protocol=reference,
        notes=["Fit and spatial validation both use 2010->2015, in disjoint original blocks.",
               "Final forecast is 2015->2020 on the unchanged full evaluation footprint.",
               "Expanded and pooled candidates have more training rows and are labeled separately.",
               "2020 has been examined in earlier experiments; this is a retrospective benchmark.",
               "Supplied map-label agreement does not establish accuracy against field-observed change."])
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    rows = [dict(name=name, selected=name == selected,
        validation_fom=record["validation"]["figure_of_merit_pct"],
        training_rows=record["validation"]["training_rows"],
        features=record["validation"]["features"],
        fixed_demand_fom=record["fixed_demand"]["figure_of_merit_pct"],
        predicted_change_pct=100*record["final"]["predicted_change_pixels"]/record["final"]["pixels"],
        **record["final"]) for name, record in records.items()]
    write_csv(output / "comparison.csv", rows)
    fig, ax = plt.subplots(figsize=(11, max(5, len(rows)*.36)))
    positions = np.arange(len(rows))
    ax.barh(positions-.16, [r["validation_fom"] for r in rows], height=.3,
            color="#9bbbd4", label="2010–2015 spatial validation")
    ax.barh(positions+.16, [r["figure_of_merit_pct"] for r in rows], height=.3,
            color=["#d78326" if r["selected"] else "#285e83" for r in rows], label="2015–2020 forecast")
    ax.axvline(40, color="#a83636", linestyle="--", label="40% target")
    ax.set_yticks(positions, [r["name"] + (" *" if r["selected"] else "") for r in rows], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Destination-sensitive Change FoM (%)")
    ax.set_title("Historical and satellite-feature experiments\n* Model chosen using spatial validation")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(.5, -.06), ncol=3)
    fig.tight_layout()
    fig.savefig(output / "fom_comparison.png", dpi=160)
    plt.close(fig)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = summarize_runs(args.runs, args.output_dir)
    print("Validation-selected model:", result["selected_model"])
    print("Final FoM:", result["selected_final_fom"])


if __name__ == "__main__":
    main()
