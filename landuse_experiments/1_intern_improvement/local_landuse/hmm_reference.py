"""Temporally consistent reference labels with a hidden Markov model.

Coarse land-cover products flip labels between similar classes, and those flips
are scored as observed change. Following Abercrombie & Friedl (2016, IEEE TGRS
54:703), each pixel's label sequence is treated as noisy observations of a hidden
true class, and the most likely true sequence is recovered with Viterbi.

Parameters are estimated only from labels up to 2015. The whole sequence,
including 2020 and 2024, is then smoothed, because the purpose is to clean the
evaluation reference, not to feed a model. No model and no saved forecast map is
changed: they were built from the raw 2015 map and are re-scored against the
cleaned 2020 reference. Results are reported next to the raw-label scores, never
in place of them.
"""

import argparse
import hashlib
import json
import logging
from pathlib import Path

import numpy as np

from .advanced import compact_metrics
from .baseline_comparison import read_mask
from .comparison import DEFAULT_DATA, read_raster, write_csv, write_map
from .models import NODATA

LOGGER = logging.getLogger(__name__)
SEQUENCE_YEARS = (2001, 2005, 2010, 2015, 2020, 2024)
FIT_YEARS = (2001, 2005, 2010, 2015)
START, TARGET = SEQUENCE_YEARS.index(2015), SEQUENCE_YEARS.index(2020)
SENSITIVITY_ACCURACIES = (.95, .90, .85, .80, .75)
FLOOR = 1e-6
ENSEMBLE_RUN = "prob_ensemble_20260910"
FORECASTS = [  # (name in comparison_with_original.json, label, role, path under outputs/)
    ("Original-style transition RF (local reproduction)", "Original-style transition RF, local reproduction",
     "baseline", "local_comparison_2010_2015_2020/transition_rf_allocation.tif"),
    ("Original-style binary RF, shared samples/blocks", "Original-style binary RF, shared samples/blocks",
     "baseline", "controlled_2010_2015_2020/temporal_holdout/binary_rf_allocation.tif"),
    ("Initial multiclass RF, shared samples/blocks", "Initial multiclass RF, shared samples/blocks",
     "baseline", "controlled_2010_2015_2020/temporal_holdout/multiclass_rf_allocation.tif"),
    ("expanded_typicality_xgb", "Best single member (expanded_typicality_xgb)",
     "candidate", f"{ENSEMBLE_RUN}/final_expanded_typicality_xgb.tif"),
    ("probability_ensemble", "Probability-averaging ensemble", "selected",
     f"{ENSEMBLE_RUN}/final_probability_ensemble.tif"),
]


def swap_shape(sequences, counts, k):
    """Off-diagonal label-confusion shape, proportional to observed swaps between snapshots."""
    swaps = np.zeros((k, k))
    for t in range(sequences.shape[1] - 1):
        np.add.at(swaps, (sequences[:, t], sequences[:, t + 1]), counts)
    swaps = swaps + swaps.T + FLOOR
    np.fill_diagonal(swaps, 0)
    return swaps / swaps.sum(axis=1, keepdims=True)


def emission_matrix(accuracy, shape):
    """P(observed label | true class): `accuracy` on the diagonal, errors shaped like observed swaps."""
    emission = (1 - accuracy) * shape
    np.fill_diagonal(emission, accuracy)
    return emission


def forward_backward(sequences, prior, transition, emission):
    """Scaled forward-backward over many sequences at once."""
    rows, steps = sequences.shape
    alpha, scale = np.empty((rows, steps, len(prior))), np.empty((rows, steps))
    current = prior * emission[:, sequences[:, 0]].T
    for t in range(steps):
        if t:
            current = (alpha[:, t - 1] @ transition) * emission[:, sequences[:, t]].T
        scale[:, t] = current.sum(axis=1)
        alpha[:, t] = current / scale[:, t, None]
    beta = np.ones_like(alpha)
    for t in range(steps - 2, -1, -1):
        beta[:, t] = ((beta[:, t + 1] * emission[:, sequences[:, t + 1]].T) @ transition.T
                      / scale[:, t + 1, None])
    return alpha, beta, scale


def fit_parameters(sequences, counts, k, accuracy=None, iterations=300, tolerance=1e-9):
    """Baum-Welch for prior and transitions, plus one label accuracy unless it is fixed.

    The confusion shape stays fixed, and the accuracy is kept in [0.5, 1) so the hidden
    classes cannot swap identities with the observed ones.
    """
    weights = counts.astype(float)
    shape = swap_shape(sequences, counts, k)
    transition = np.full((k, k), FLOOR)
    for t in range(sequences.shape[1] - 1):
        np.add.at(transition, (sequences[:, t], sequences[:, t + 1]), weights)
    transition /= transition.sum(axis=1, keepdims=True)
    prior = np.bincount(sequences[:, 0], weights, minlength=k) + FLOOR
    prior /= prior.sum()
    current, previous = (.9 if accuracy is None else float(accuracy)), -np.inf
    for iteration in range(1, iterations + 1):
        emission = emission_matrix(current, shape)
        alpha, beta, scale = forward_backward(sequences, prior, transition, emission)
        log_likelihood = float(weights @ np.log(scale).sum(axis=1))
        gamma = alpha * beta
        prior = weights @ gamma[:, 0] + FLOOR
        prior /= prior.sum()
        expected = np.full((k, k), FLOOR)
        for t in range(sequences.shape[1] - 1):
            right = beta[:, t + 1] * emission[:, sequences[:, t + 1]].T / scale[:, t + 1, None]
            expected += transition * np.einsum("u,ui,uj->ij", weights, alpha[:, t], right)
        transition = expected / expected.sum(axis=1, keepdims=True)
        if accuracy is None:
            matched = np.take_along_axis(gamma, sequences[..., None], axis=2)[..., 0]
            current = float(np.clip(weights @ matched.sum(axis=1) / (weights.sum() * sequences.shape[1]),
                                    .5, 1 - FLOOR))
        if log_likelihood - previous < tolerance * abs(log_likelihood):
            break
        previous = log_likelihood
    return dict(prior=prior, transition=transition, shape=shape, accuracy=current,
                estimated_accuracy=accuracy is None, log_likelihood=log_likelihood, iterations=iteration)


def fit_from_stack(stack, classes, accuracy=None):
    """Fit on the FIT_YEARS columns only; later columns are never read."""
    codes = np.searchsorted(classes, stack[:, :len(FIT_YEARS)])
    unique, counts = np.unique(codes, axis=0, return_counts=True)
    return fit_parameters(unique, counts, len(classes), accuracy=accuracy)


def viterbi(sequences, prior, transition, emission):
    """Most likely hidden class sequence for each observed sequence."""
    with np.errstate(divide="ignore"):
        log_transition, log_emission, log_prior = np.log(transition), np.log(emission), np.log(prior)
    rows, steps = sequences.shape
    score = log_prior + log_emission[:, sequences[:, 0]].T
    back = np.empty((rows, steps, len(prior)), dtype=np.int16)
    for t in range(1, steps):
        candidates = score[:, :, None] + log_transition[None]
        back[:, t] = candidates.argmax(axis=1)
        score = candidates.max(axis=1) + log_emission[:, sequences[:, t]].T
    path = np.empty((rows, steps), dtype=np.int64)
    path[:, -1] = score.argmax(axis=1)
    for t in range(steps - 1, 0, -1):
        path[:, t - 1] = back[np.arange(rows), t, path[:, t]]
    return path


def smooth_labels(stack, classes, parameters):
    """Viterbi once per distinct label sequence, mapped back to every pixel."""
    if not np.isin(stack, classes).all():
        raise ValueError("Label stack contains classes outside the vocabulary")
    unique, inverse = np.unique(np.searchsorted(classes, stack), axis=0, return_inverse=True)
    emission = emission_matrix(parameters["accuracy"], parameters["shape"])
    return classes[viterbi(unique, parameters["prior"], parameters["transition"], emission)[inverse.ravel()]]


def score(label, role, setting, reference, source, observed, predicted):
    m = compact_metrics(source, observed, predicted, predicted != source)
    return dict(setting=setting, reference=reference, forecast=label, role=role, pixels=len(source), **m)


def run_hmm_reference(output_dir, data_dir=DEFAULT_DATA):
    output, data = Path(output_dir), Path(data_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new/empty output directory")
    outputs = data.parent / "outputs"
    maps, profile = {}, None
    for year in SEQUENCE_YEARS:
        maps[year], profile = read_raster(data / f"Guinea_LandCover_{year}.tif", profile, categorical=True)
    mask = read_mask(outputs / ENSEMBLE_RUN / "evaluation_mask.tif", profile)
    if any(np.any(maps[year][mask] == NODATA) for year in SEQUENCE_YEARS):
        raise ValueError("Every evaluation pixel needs a label in every sequence year")
    stack = np.stack([maps[year][mask] for year in SEQUENCE_YEARS], axis=1)
    classes = np.unique(stack)
    output.mkdir(parents=True)

    settings = {"estimated": fit_from_stack(stack, classes)}
    for accuracy in SENSITIVITY_ACCURACIES:
        settings[f"accuracy_{accuracy:.2f}"] = fit_from_stack(stack, classes, accuracy)
    frozen = {name: dict(accuracy=p["accuracy"], estimated_accuracy=p["estimated_accuracy"],
                         log_likelihood=p["log_likelihood"], iterations=p["iterations"],
                         prior=p["prior"].tolist(), transition=p["transition"].tolist(),
                         confusion_shape=p["shape"].tolist())
              for name, p in settings.items()}
    (output / "parameters.json").write_text(json.dumps(dict(
        classes=classes.tolist(), fit_years=FIT_YEARS, sequence_years=SEQUENCE_YEARS,
        primary_setting="estimated", primary_reference="stable_start", settings=frozen), indent=2))
    LOGGER.info("FROZEN parameters from %s labels; estimated label accuracy %.3f",
                FIT_YEARS, settings["estimated"]["accuracy"])

    saved = {row["name"]: row["forecast_fom"] for row in
             json.loads((outputs / ENSEMBLE_RUN / "comparison_with_original.json").read_text())["rows"]}
    raw_start, raw_target = stack[:, START], stack[:, TARGET]
    forecasts = [(label, role, read_raster(outputs / path, profile, categorical=True)[0][mask])
                 for _, label, role, path in FORECASTS]
    for (name, label, _, _), (_, _, predicted) in zip(FORECASTS, forecasts):
        fom = compact_metrics(raw_start, raw_target, predicted, predicted != raw_start)["figure_of_merit_pct"]
        if not np.isclose(fom, saved[name]):
            raise ValueError(f"{label} no longer reproduces its saved raw-label FoM")
    reversion = np.where(stack[:, 2] != raw_start, stack[:, 2], raw_start)
    forecasts.append(("Reversion rule: 2010->2015 changers return to 2010 (diagnostic)", "diagnostic", reversion))

    rows = [score(label, role, "raw", "raw_labels", raw_start, raw_target, predicted)
            for label, role, predicted in forecasts]
    diagnostics = {}
    raw_change = raw_target != raw_start
    reverts = raw_change & (stack[:, SEQUENCE_YEARS.index(2024)] == raw_start)
    for name, parameters in settings.items():
        smoothed = smooth_labels(stack, classes, parameters)
        stable = smoothed[:, START] == raw_start
        clean_change = stable & (smoothed[:, TARGET] != raw_start)
        diagnostics[name] = dict(
            label_accuracy=parameters["accuracy"],
            pixels_with_any_relabelled_year_pct=100 * float(np.mean((smoothed != stack).any(axis=1))),
            noisy_start_pixels=int((~stable).sum()),
            raw_observed_change=int(raw_change.sum()),
            stable_start_reference_change=int(clean_change.sum()),
            raw_changes_reverting_by_2024=int(reverts.sum()),
            reverting_changes_removed_pct=100 * float(np.mean(~clean_change[reverts])),
            persistent_changes_removed_pct=100 * float(np.mean(~clean_change[raw_change & ~reverts])))
        for label, role, predicted in forecasts:
            rows.append(score(label, role, name, "stable_start", raw_start[stable],
                              smoothed[stable, TARGET], predicted[stable]))
            rows.append(score(label, role, name, "full_mask", raw_start, smoothed[:, TARGET], predicted))
        if name == "estimated":
            for year in (2015, 2020):
                grid = np.full(maps[year].shape, NODATA, dtype=np.int32)
                grid[mask] = smoothed[:, SEQUENCE_YEARS.index(year)]
                write_map(output / f"hmm_reference_{year}.tif", grid, profile)
        LOGGER.info("%s: accuracy %.3f, stable-start reference change %d (raw %d)", name,
                    parameters["accuracy"], clean_change.sum(), raw_change.sum())

    original = {(r["setting"], r["reference"]): r["figure_of_merit_pct"] for r in rows
                if r["forecast"] == FORECASTS[0][1]}
    for row in rows:
        row["delta_fom_vs_original"] = row["figure_of_merit_pct"] - original[(row["setting"], row["reference"])]
    write_csv(output / "hmm_reference_scores.csv", rows)
    report = dict(
        method="Viterbi smoothing of MCD12Q1 label sequences (Abercrombie & Friedl 2016)",
        target_fom=40., parameters_file="parameters.json", diagnostics=diagnostics, rows=rows,
        inputs={f"Guinea_LandCover_{y}.tif": hashlib.sha256((data / f"Guinea_LandCover_{y}.tif").read_bytes()
                                                             ).hexdigest() for y in SEQUENCE_YEARS},
        notes=["Parameters use 2001/2005/2010/2015 labels only; smoothing then uses the full 2001-2024 sequence.",
               "No model or forecast map was refitted or altered; only the reference changes.",
               "stable_start keeps pixels whose raw 2015 label survives smoothing, so model inputs and reference start agree.",
               "full_mask keeps all pixels with raw 2015 as the start and smoothed 2020 as the target.",
               "Snapshots are 4-5 years apart; a single homogeneous transition matrix is assumed.",
               "A different reference definition: never a replacement for the raw-label result."])
    (output / "report.json").write_text(json.dumps(report, indent=2, default=float))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    report = run_hmm_reference(args.output_dir)
    for row in report["rows"]:
        if row["setting"] in ("raw", "estimated") and row["reference"] in ("raw_labels", "stable_start"):
            print(f"{row['reference']:<13} {row['forecast']:<66} FoM {row['figure_of_merit_pct']:6.2f}%")


if __name__ == "__main__":
    main()
