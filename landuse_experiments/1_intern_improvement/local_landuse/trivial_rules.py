"""Model-free 2015->2020 rules, scored on the same mask as every model.

The point of comparison is what a rule with no fitted model achieves. Each rule
reads land-cover labels up to 2015 only; the "most likely destination" counts come
from the 2010->2015 transition table. The three original-style baselines are read
from their saved maps and must reproduce their recorded FoM before any rule is
scored.
"""

import argparse
import hashlib
import json
import logging
from pathlib import Path

import numpy as np

from .advanced import compact_metrics
from .baseline_comparison import read_mask
from .comparison import DEFAULT_DATA, read_raster, write_csv
from .models import NODATA

LOGGER = logging.getLogger(__name__)
YEARS = (2001, 2005, 2010, 2015)
ENSEMBLE_RUN = "prob_ensemble_20260910"
BASELINES = ("Original-style transition RF (local reproduction)",
             "Original-style binary RF, shared samples/blocks",
             "Initial multiclass RF, shared samples/blocks",
             "probability_ensemble")


def neighbourhood_majority(grid, classes, include_self=True):
    """Modal class of the 3x3 window; edges use fewer neighbours and never wrap around."""
    counts = np.zeros((len(classes),) + grid.shape, dtype=np.int16)
    for index, code in enumerate(classes):
        member = (grid == code).astype(np.int16)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if not (dy or dx) and not include_self:
                    continue
                window = member[max(0, -dy):member.shape[0] - max(0, dy),
                                max(0, -dx):member.shape[1] - max(0, dx)]
                counts[index, max(0, dy):counts.shape[1] - max(0, -dy),
                       max(0, dx):counts.shape[2] - max(0, -dx)] += window
    return np.asarray(classes)[counts.argmax(axis=0)]


def modal_class(sequence):
    """Most frequent label per pixel across a (pixels, years) stack; ties take the lowest class."""
    values = np.unique(sequence)
    counts = np.stack([(sequence == value).sum(axis=1) for value in values])
    return values[counts.argmax(axis=0)]


def likely_destinations(start, end, classes):
    """Most common destination of each class among pixels that changed."""
    changed = start != end
    out = {}
    for code in classes:
        selected = changed & (start == code)
        if selected.any():
            values, counts = np.unique(end[selected], return_counts=True)
            out[int(code)] = int(values[counts.argmax()])
    return out


def build_rules(history, neighbour_majority, classes, neighbours_only=None):
    """Every rule as a predicted 2020 map, from labels up to 2015 only."""
    m01, m05, m10, m15 = (history[year] for year in YEARS)
    if neighbours_only is None:
        neighbours_only = neighbour_majority
    changed = m15 != m10
    twice = changed & (m10 != m05)
    modal = modal_class(np.stack([m01, m05, m10, m15], axis=1))
    destinations = likely_destinations(m10, m15, classes)
    forward = np.asarray([destinations.get(int(c), int(c)) for c in classes])[np.searchsorted(classes, m15)]
    return {
        "Every pixel takes the majority class of its eight 2015 neighbours": neighbours_only,
        "Every pixel takes the majority class of its 2015 3x3 window": neighbour_majority,
        "Previous changers take the majority class of their eight 2015 neighbours": np.where(changed, neighbours_only, m15),
        "Revert to the 2010 class if changed 2010->2015": np.where(changed, m10, m15),
        "Revert to the 2005 class if changed 2010->2015": np.where(changed, m05, m15),
        "Revert to the modal 2001-2015 class if changed 2010->2015": np.where(changed, modal, m15),
        "Continue: previous changers move to their class's most likely destination": np.where(changed, forward, m15),
        "Revert to the modal class if changed twice (2005->2010 and 2010->2015)": np.where(twice, modal, m15),
        "No change": m15.copy(),
    }


def run_trivial_rules(output_dir, data_dir=DEFAULT_DATA):
    output, data = Path(output_dir), Path(data_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new/empty output directory")
    outputs = data.parent / "outputs"
    maps, profile = {}, None
    for year in YEARS + (2020,):
        maps[year], profile = read_raster(data / f"Guinea_LandCover_{year}.tif", profile, categorical=True)
    mask = read_mask(outputs / ENSEMBLE_RUN / "evaluation_mask.tif", profile)
    if any(np.any(maps[year][mask] == NODATA) for year in maps):
        raise ValueError("Every evaluation pixel needs a label in every year used")
    history = {year: maps[year][mask] for year in YEARS}
    classes = np.unique(np.concatenate(list(history.values())))
    neighbour_majority = neighbourhood_majority(maps[2015], classes)[mask]
    neighbours_only = neighbourhood_majority(maps[2015], classes, include_self=False)[mask]
    start, observed = history[2015], maps[2020][mask]

    saved = {row["name"]: row["forecast_fom"] for row in json.loads(
        (outputs / ENSEMBLE_RUN / "comparison_with_original.json").read_text())["rows"]}
    rows = []
    for name in BASELINES:
        rows.append(dict(rule=name, role="baseline", figure_of_merit_pct=saved[name]))
    for name, predicted in build_rules(history, neighbour_majority, classes, neighbours_only).items():
        metrics = compact_metrics(start, observed, predicted, predicted != start)
        rows.append(dict(rule=name, role="trivial", **metrics))
        LOGGER.info("%-72s FoM %5.2f%%", name, metrics["figure_of_merit_pct"])
    best = max((r for r in rows if r["role"] == "trivial"), key=lambda r: r["figure_of_merit_pct"])
    output.mkdir(parents=True)
    # Baseline rows carry only their recorded FoM; pad every row to one column set.
    columns = list({key: None for row in rows for key in row})
    write_csv(output / "trivial_rules.csv", [{c: row.get(c) for c in columns} for row in rows])
    report = dict(evaluation_period=[2015, 2020], pixels=int(mask.sum()), rows=rows,
                  best_trivial_rule=best["rule"], best_trivial_fom=best["figure_of_merit_pct"],
                  beats_original_style_baselines=best["figure_of_merit_pct"] > saved[BASELINES[0]],
                  inputs={f"Guinea_LandCover_{y}.tif": hashlib.sha256(
                      (data / f"Guinea_LandCover_{y}.tif").read_bytes()).hexdigest() for y in maps},
                  notes=["Rules read labels up to 2015 only; 2020 is used for scoring.",
                         "Baseline rows are the saved original-style maps, re-scored on the identical mask.",
                         "A trivial rule scoring near a model is evidence about the metric, not about the model."])
    (output / "report.json").write_text(json.dumps(report, indent=2, default=float))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    report = run_trivial_rules(args.output_dir)
    print("Best trivial rule:", report["best_trivial_rule"], f"{report['best_trivial_fom']:.2f}%")


if __name__ == "__main__":
    main()
