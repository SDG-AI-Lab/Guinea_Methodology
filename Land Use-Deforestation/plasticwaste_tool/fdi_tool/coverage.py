"""
River corridor coverage: how much of the corridor network a river run
actually assessed (feature 006; specs/006-plastic-waste-views/contracts/
toolchain.md §1; research R6; doc 70 K36).

One implementation, used by the river job (stored on each run as
`summary_statistics["coverage"]`, with each tile's outcome handed to the
publish step) and by `plasticwaste_api.evidence.upstream` on the January
2024 study file, so the length the map states and the length the evidence
states cannot drift apart.

A corridor tile's outcome, in this order:

- `below_water_floor` -- not assessed: too little water at 10-20 m for a
  threshold to mean anything;
- `no_water_mask`     -- a missing or NaN FDI threshold: no water was found,
  so no threshold exists (fifteen study tiles once passed as assessed with
  one);
- `implausible`       -- a missing NDVI threshold, or one outside [-1, 1],
  which is algebraically impossible (nodata or scene edge; four study tiles
  had values up to 55.7);
- `valid`             -- otherwise.

Rows are deduplicated by AOI first (the last one wins), because `--resume`
appends rather than replaces. `coverage_fraction` is None when no length is
known: 0 of 0 km is not 0 %.
"""

from __future__ import annotations

import math

OUTCOMES = ("valid", "below_water_floor", "no_water_mask", "implausible")


def _missing(value) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def outcome(result: dict) -> str:
    """The outcome of one corridor tile's run result."""
    if not result.get("assessed"):
        return "below_water_floor"
    if _missing(result.get("threshold_fdi")):
        return "no_water_mask"
    ndvi = result.get("threshold_ndvi")
    if _missing(ndvi) or abs(ndvi) > 1.0:
        return "implausible"
    return "valid"


def river_coverage(results: list[dict], corridors: dict[str, dict]) -> dict:
    """`{km_assessed, km_total, coverage_fraction, tiles_by_outcome,
    outcome_by_tile, method}` for the corridor tiles of one river run.

    `results` are the per-tile run results (`aoi`, `assessed`,
    `threshold_fdi`, `threshold_ndvi`, ...); `corridors` the corridor
    definitions of `river_aois.json` keyed by AOI (`length_km`). A tile's
    length is its corridor's, else the one its result carries."""
    rows = {row["aoi"]: row for row in results}
    outcome_by_tile = {aoi: outcome(row) for aoi, row in rows.items()}
    lengths = {aoi: float((corridors.get(aoi) or {}).get("length_km") or rows[aoi].get("length_km") or 0.0)
               for aoi in rows}
    km_total = sum(lengths.values())
    km_assessed = sum(lengths[aoi] for aoi, said in outcome_by_tile.items() if said == "valid")
    return {
        "km_assessed": km_assessed,
        "km_total": km_total,
        "coverage_fraction": (km_assessed / km_total) if km_total else None,
        "tiles_by_outcome": {said: sum(1 for value in outcome_by_tile.values() if value == said)
                             for said in OUTCOMES},
        "outcome_by_tile": outcome_by_tile,
        "method": "river_coverage",
    }
