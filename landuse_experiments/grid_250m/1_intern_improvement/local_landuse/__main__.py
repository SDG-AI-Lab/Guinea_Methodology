"""Usage: python -m local_landuse --output-dir outputs/run_01"""

import argparse
import logging

from .comparison import DEFAULT_DATA, run_comparison
from .controlled import run_controlled_comparison
from .models import Settings


def main():
    parser = argparse.ArgumentParser(description="Compare next-class RF, binary transition/allocation "
                                     "and persistence using local maps; no cloud credentials needed.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA))
    parser.add_argument("--output-dir", required=True, help="New or empty output directory")
    parser.add_argument("--years", nargs=3, type=int, default=[2010, 2015, 2020],
                        metavar=("TRAIN_START", "FORECAST_START", "TEST_YEAR"))
    parser.add_argument("--trees", type=int, default=100)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=100000)
    parser.add_argument("--experiment", choices=("controlled", "direct"), default="controlled",
                        help="controlled: shared spatial split and allocation (default); direct: earlier argmax experiment")
    parser.add_argument("--block-size", type=int, default=100)
    parser.add_argument("--test-size", type=float, default=.3)
    parser.add_argument("--sampling", choices=("shared", "legacy_balanced"), default="shared",
                        help="shared: identical rows for every RF (default); legacy_balanced: earlier comparison")
    parser.add_argument("--min-change-pixels", type=int, default=200)
    parser.add_argument("--neighborhood", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exclude-codes", nargs="*", type=int, default=[0, 255])
    parser.add_argument("--static-predictor", action="append", dest="static_predictors",
                        help="Aligned static raster (repeatable); replaces default terrain input")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    try:
        if args.experiment == "controlled" and args.sampling != "shared":
            parser.error("The controlled experiment requires --sampling shared")
        run = run_controlled_comparison if args.experiment == "controlled" else run_comparison
        extra = ({"block_size": args.block_size, "test_size": args.test_size}
                 if args.experiment == "controlled" else {"sampling": args.sampling})
        report = run(
            data_dir=args.data_dir, output_dir=args.output_dir, years=args.years,
            settings=Settings(n_estimators=args.trees, n_jobs=args.jobs, random_state=args.seed),
            max_samples=args.max_samples, min_change_pixels=args.min_change_pixels,
            neighborhood=args.neighborhood, exclude_codes=args.exclude_codes,
            static_predictors=args.static_predictors, make_plot=not args.no_plot,
            **extra)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    scopes = ({name: report[name]["results"] for name in ("spatial_holdout", "temporal_holdout")}
              if args.experiment == "controlled" else {"direct_temporal": report["results"]})
    for scope, results in scopes.items():
        print(f"\n{scope}\nModel                         Accuracy   Macro F1   Change FoM   Changed pixels")
        for name, result in results.items():
            m = result["summary"]
            print(f"{name:29s} {m['overall_accuracy']:8.2%} {m['macro_f1']:10.3f} "
                  f"{m['figure_of_merit_pct']:10.2f}% {m['predicted_change_pixels']:15,d}")


if __name__ == "__main__":
    main()
