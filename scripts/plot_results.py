"""Plot a placement comparison from its results JSON.

Requires the optional plotting extra:

    uv sync --extra plot
    uv run python scripts/plot_results.py results/placement_comparison.json
"""

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Plot the placement comparison")
    parser.add_argument("results", help="path to a placement_comparison.json")
    parser.add_argument("--out", default="results/placement_comparison.png")
    args = parser.parse_args(argv)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise SystemExit("matplotlib is required: uv sync --extra plot") from error

    data = json.loads(Path(args.results).read_text())
    placements = list(data)
    metrics = ["across_time_variance", "effective_rank", "forecast_mse"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 3.5))
    for axis, metric in zip(axes, metrics, strict=True):
        axis.bar(placements, [data[p][metric] for p in placements])
        axis.set_title(metric)
    fig.suptitle("SIGReg placement comparison")
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
