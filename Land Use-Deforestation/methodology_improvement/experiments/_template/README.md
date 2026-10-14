# <Experiment title>

- **Owner:**
- **Backlog item:** B__
- **Module:** Land Use / Deforestation / Plastic Waste
- **Started:** YYYY-MM-DD

## Hypothesis

What change, and what improvement you expect from it. Write this before
running anything.

## Baseline

The metric, the data split and the number, copied from `../../BASELINES.md`.

## Change

What differs from the production method: code, parameters, data or imagery.

## Data

Training, tuning and held-out data, including where the held-out data comes
from and why it is independent of the training data.

## Result

| Method   | Metric | Value |
|----------|--------|-------|
| Baseline |        |       |
| Proposed |        |       |

Cost: runtime, Earth Engine quota, licensed imagery.

## Verdict

Accepted / rejected / inconclusive, and why. Copy one line into
`../../DECISIONS.md`.

## Files

- `run.py` or `analysis.ipynb`: the experiment
- `results/`: summary JSON or CSV (committed)
- `outputs/`: large intermediates (ignored by git)
