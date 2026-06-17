"""SSL pretrain-then-probe classification on a real labeled dataset (UCI HAR).

Download and extract UCI HAR first (the script reads the extracted directory):

    uv run python scripts/classify_dataset.py --har "data/UCI HAR Dataset" --seeds 5
"""

import argparse

from chronojepa.data import load_har
from chronojepa.eval import format_ssl_classification_table, run_ssl_classification


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="SSL classification on a labeled dataset")
    parser.add_argument("--har", required=True, help="path to the extracted UCI HAR Dataset dir")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--out", default="results/har_classification.json")
    args = parser.parse_args(argv)

    x_train, y_train, x_test, y_test = load_har(args.har)
    aggregate = run_ssl_classification(
        x_train,
        y_train,
        x_test,
        y_test,
        seeds=tuple(range(args.seeds)),
        steps=args.steps,
        d_model=args.d_model,
        results_path=args.out,
    )
    print(format_ssl_classification_table(aggregate))
    print(f"\nsaved: {args.out}")
    return aggregate


if __name__ == "__main__":
    main()
