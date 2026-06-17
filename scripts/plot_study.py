"""Plot the architecture x placement factorial study into publication figures.

Requires the plotting extra:

    uv sync --extra plot
    uv run python scripts/plot_study.py results/architecture_study.json --outdir results/figures
"""

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Plot the architecture study")
    parser.add_argument("results", help="path to architecture_study.json")
    parser.add_argument("--outdir", default="results/figures")
    args = parser.parse_args(argv)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise SystemExit("matplotlib is required: uv sync --extra plot") from error

    data = json.loads(Path(args.results).read_text())
    configs = list(data)  # e.g. positional|pooled, ...
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    def _bar(
        metrics: list[str], title: str, ylabel: str, filename: str, chance: float | None = None
    ) -> None:
        fig, ax = plt.subplots(figsize=(1.6 * len(configs) + 2, 4))
        width = 0.8 / len(metrics)
        positions = range(len(configs))
        for i, metric in enumerate(metrics):
            means = [data[c][metric]["mean"] for c in configs]
            stds = [data[c][metric]["std"] for c in configs]
            offset = (i - (len(metrics) - 1) / 2) * width
            ax.bar(
                [p + offset for p in positions], means, width, yerr=stds, capsize=3, label=metric
            )
        if chance is not None:
            ax.axhline(chance, linestyle="--", color="gray", linewidth=1, label="chance")
        ax.set_xticks(list(positions))
        ax.set_xticklabels(configs, rotation=20, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(outdir / filename, dpi=150)
        plt.close(fig)

    _bar(
        ["across_time_variance"],
        "Collapse diagnostic (placement controls it)",
        "across-time variance",
        "collapse.png",
    )
    _bar(
        ["halfswap_token", "halfswap_pooled"],
        "Order recovery: position-free + pooled loses order",
        "halfswap AUROC-style accuracy",
        "halfswap.png",
        chance=0.5,
    )
    _bar(
        ["trend_token", "trend_pooled"],
        "Trend classification by feature",
        "accuracy",
        "trend.png",
        chance=0.5,
    )
    print(f"wrote figures to {outdir}/ : collapse.png, halfswap.png, trend.png")


if __name__ == "__main__":
    main()
