"""Relative-action candidates declared before reading the advanced final results."""

import argparse
import logging

from .advanced import run_advanced


CANDIDATES = [
    dict(name="action_rf", family="rf", structure="action", geometry=True,
         multiscale=True, weight=1., trees=150, depth=16),
    dict(name="action_xgb", family="xgb", structure="action", geometry=True,
         multiscale=True, weight=1., trees=300, depth=4),
    dict(name="action_xgb_deep", family="xgb", structure="action", geometry=True,
         multiscale=True, weight=1., trees=600, depth=6),
    dict(name="action_xgb_more_samples", family="xgb", structure="action", geometry=True,
         multiscale=True, weight=1., trees=300, depth=4, expanded=True),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    report = run_advanced(output_dir=args.output_dir, candidates=CANDIDATES)
    print("Selected:", report["selection"]["selected_model"])
    for name, result in report["results"].items():
        if not name.endswith("_fixed_demand"):
            print(name, result["summary"]["figure_of_merit_pct"])


if __name__ == "__main__":
    main()
