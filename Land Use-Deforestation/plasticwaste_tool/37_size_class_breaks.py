#!/usr/bin/env python
"""
The floating-debris size classes: three classes by Fisher-Jenks natural breaks
(feature 006 FR-013a; research R7; doc 70 S46;
specs/006-plastic-waste-views/contracts/toolchain.md §2).

    python 37_size_class_breaks.py --dry-run          # print, write nothing
    python 37_size_class_breaks.py                    # write SIZE_CLASS_CONFIG into config.py
    python 37_size_class_breaks.py --accept-counts    # after the plastic lead's note on the counts

WHAT IT DOES

Reads the patch areas (m²) of the reference national run, January 2024, in
Conakry, Konkouré and Fatala -- 30 patches per the methodology (26, 0, 4);
the "mangrove" box's offshore patches are excluded (doc 70 D17). If that run's
output is not on disk it re-runs `pipeline.run_national("2024-01-01",
"2024-02-01", region_keys=["conakry", "konkoure", "fatala"])` (needs
GEE_SA_KEY), says so, and keeps the output. It stops if the counts differ
from the methodology's, unless --accept-counts records that the plastic lead
has looked at the difference.

Three classes by `mapclassify.FisherJenks(areas, k=3)`, Fisher's exact
optimisation of Jenks natural breaks; the two upper bounds below the maximum
are the breaks, each rounded to the nearest whole pixel of the 20 m grid
(`pixel_m2 = download_scale_m²`, 400 m²), since every area is a whole number
of pixels. It stops, writing nothing, if rounding makes the breaks equal or
puts the first at or below the smallest patch: a class would be empty by
construction, and that is the plastic lead's decision, not a silent merge.

Output: `SIZE_CLASS_CONFIG = {breaks_m2: [b1, b2], method: "fisher-jenks", k:
3, reference, n, pixel_m2, computed_at}` between the markers in config.py, a
change reviewed in the pull request. The publish step then classes patches
small `area_m2 < b1`, medium `b1 <= area_m2 < b2`, large `area_m2 >= b2`.
"""

from __future__ import annotations

import argparse
import json
import pprint
import re
import sys
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "config.py"
REFERENCE_PATH = HERE / "runs" / "size_classes" / "national_2024-01_patches.geojson"
REFERENCE = {
    "start": "2024-01-01", "end": "2024-02-01",
    # The methodology's patch counts of the reference run (doc 70 S46, S76).
    "counts": {"conakry": 26, "konkoure": 0, "fatala": 4},
    "label": "national 2024-01, conakry+konkoure+fatala",
}
START, END = "# --- SIZE_CLASS_CONFIG", "# --- end SIZE_CLASS_CONFIG ---"


class BreaksRefused(ValueError):
    """The breaks would leave a class empty by construction."""


def compute_breaks(areas, pixel_m2: int = 400, k: int = 3) -> list[int]:
    """[b1, b2] in m²: Fisher-Jenks upper bounds below the maximum, rounded to
    the nearest multiple of `pixel_m2`."""
    import mapclassify

    values = sorted(float(area) for area in areas)
    if len(set(values)) < k:
        raise BreaksRefused(f"{len(set(values))} distinct areas cannot make {k} classes")
    bins = list(mapclassify.FisherJenks(values, k=k).bins)
    breaks = [int(round(bound / pixel_m2)) * pixel_m2 for bound in bins[: k - 1]]
    if len(set(breaks)) < len(breaks):
        raise BreaksRefused(f"rounding to {pixel_m2} m² makes the breaks equal: {breaks}")
    if breaks[0] <= values[0]:
        raise BreaksRefused(f"the first break {breaks[0]} m² is not above the smallest patch ({values[0]} m²)")
    return breaks


def config_block(breaks, n: int, pixel_m2: int = 400, computed_at: str | None = None) -> str:
    config = {"breaks_m2": list(breaks), "method": "fisher-jenks", "k": 3, "reference": REFERENCE["label"],
              "n": n, "pixel_m2": pixel_m2, "computed_at": computed_at or date.today().isoformat()}
    return (f"{START} (written by 37_size_class_breaks.py; do not edit by hand) ---\n"
            "# Floating-debris size classes (feature 006, research R7; doc 70 S46): small\n"
            "# area_m2 < b1, medium b1 <= area_m2 < b2, large area_m2 >= b2. The publish\n"
            "# step classifies patches with it and stores it on the version.\n"
            f"SIZE_CLASS_CONFIG: Dict = {pprint.pformat(config, sort_dicts=False, width=100)}\n"
            f"{END}")


def write_config(block: str, path: Path = CONFIG_PATH) -> None:
    """Replace the block between the markers (or add it at the end)."""
    source = path.read_text(encoding="utf-8")
    pattern = re.compile(re.escape(START) + r".*?" + re.escape(END), re.S)
    updated = pattern.sub(lambda _: block, source) if pattern.search(source) else f"{source.rstrip()}\n\n{block}\n"
    path.write_text(updated, encoding="utf-8")


def areas_by_region(path: Path) -> dict[str, list[float]]:
    features = json.loads(path.read_text(encoding="utf-8")).get("features", [])
    result: dict[str, list[float]] = {}
    for feature in features:
        properties = feature.get("properties") or {}
        result.setdefault(properties["region"], []).append(float(properties["area_m2"]))
    return result


def check_counts(by_region: dict[str, list[float]], accept: bool) -> list[str]:
    """The differences from the methodology's counts; an error without --accept-counts."""
    differences = [f"{region}: {len(by_region.get(region, []))} patches, methodology {expected}"
                   for region, expected in REFERENCE["counts"].items() if len(by_region.get(region, [])) != expected]
    if differences and not accept:
        raise SystemExit("patch counts differ from the methodology's; ask the plastic lead, then re-run with "
                         "--accept-counts:\n  " + "\n  ".join(differences))
    return differences


def run_reference(path: Path) -> None:
    """Re-run the reference national run (Earth Engine; needs GEE_SA_KEY)."""
    from plasticwaste_tool.fdi_tool import pipeline

    print(f"reference output not found at {path}; re-running run_national (Earth Engine)", file=sys.stderr)
    result = pipeline.run_national(REFERENCE["start"], REFERENCE["end"], region_keys=list(REFERENCE["counts"]))
    vectors = result["hotspot_vectors"]
    if vectors is None or vectors.empty:
        raise SystemExit("the reference run produced no patches")
    path.parent.mkdir(parents=True, exist_ok=True)
    vectors[["region", "area_m2", "geometry"]].to_file(path, driver="GeoJSON")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dry-run", action="store_true", help="print the configuration, write nothing")
    parser.add_argument("--accept-counts", action="store_true",
                        help="the plastic lead has accepted counts that differ from the methodology's")
    parser.add_argument("--reference", type=Path, default=REFERENCE_PATH)
    args = parser.parse_args(argv)

    if not args.reference.is_file():
        run_reference(args.reference)
    by_region = areas_by_region(args.reference)
    differences = check_counts(by_region, args.accept_counts)
    areas = [area for region in REFERENCE["counts"] for area in by_region.get(region, [])]
    try:
        from plasticwaste_tool import config
        pixel_m2 = int(config.S2_CONFIG["download_scale_m"]) ** 2
    except ImportError:
        pixel_m2 = 400
    try:
        breaks = compute_breaks(areas, pixel_m2=pixel_m2)
    except BreaksRefused as refused:
        print(f"refused, nothing written: {refused}", file=sys.stderr)
        return 1
    block = config_block(breaks, n=len(areas), pixel_m2=pixel_m2)
    if differences:
        print("accepted count differences:\n  " + "\n  ".join(differences))
    print(block)
    if not args.dry_run:
        write_config(block)
        print(f"written to {CONFIG_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
