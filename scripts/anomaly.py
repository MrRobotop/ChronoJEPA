"""Compare placements on injected-anomaly detection AUROC (Mahalanobis on token features).

Synthetic by default:

    uv run python scripts/anomaly.py

On PEMS:

    uv run python scripts/anomaly.py --pems data/pems08.npz --seeds 3
"""

import argparse

import numpy as np

from chronojepa.data import load_pems
from chronojepa.eval import format_anomaly_table, run_anomaly_comparison


def _synthetic_series() -> np.ndarray:
    rng = np.random.default_rng(0)
    t = np.linspace(0.0, 8.0, 2400)[:, None]
    columns = [
        np.sin(2.0 * np.pi * 3.0 * t) + 0.3 * rng.standard_normal((2400, 1)),
        np.sin(2.0 * np.pi * 7.0 * t + 1.0) + 0.3 * rng.standard_normal((2400, 1)),
        np.cos(2.0 * np.pi * 2.0 * t) + 0.3 * rng.standard_normal((2400, 1)),
    ]
    return np.concatenate(columns, axis=1).astype(np.float32)


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Anomaly-detection placement comparison")
    parser.add_argument("--pems", help="path to a PEMS .npz; uses synthetic data if omitted")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--strength", type=float, default=4.0)
    parser.add_argument("--out", default="results/anomaly_comparison.json")
    args = parser.parse_args(argv)

    series = load_pems(args.pems) if args.pems else _synthetic_series()
    aggregate = run_anomaly_comparison(
        series,
        seeds=tuple(range(args.seeds)),
        steps=args.steps,
        strength=args.strength,
        results_path=args.out,
    )
    print(format_anomaly_table(aggregate))
    print(f"\nsaved: {args.out}")
    return aggregate


if __name__ == "__main__":
    main()
