"""Forecasting-vs-horizon sweep: does dual help once short-horizon persistence breaks down?

Synthetic by default:

    uv run python scripts/horizon_sweep.py

On PEMS:

    uv run python scripts/horizon_sweep.py --pems data/pems08.npz --seeds 3
"""

import argparse

import numpy as np

from chronojepa.data import load_ett, load_pems
from chronojepa.eval import format_horizon_table, run_horizon_sweep


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
    parser = argparse.ArgumentParser(description="Forecasting-vs-horizon placement sweep")
    parser.add_argument("--pems", help="path to a PEMS .npz")
    parser.add_argument("--ett", help="path to an ETT .csv")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--horizons", default="3,6,12,24,48")
    parser.add_argument("--out", default="results/horizon_sweep.json")
    args = parser.parse_args(argv)

    if args.pems:
        series = load_pems(args.pems)
    elif args.ett:
        series = load_ett(args.ett)
    else:
        series = _synthetic_series()
    horizons = tuple(int(h) for h in args.horizons.split(","))
    aggregate = run_horizon_sweep(
        series,
        horizons=horizons,
        seeds=tuple(range(args.seeds)),
        steps=args.steps,
        results_path=args.out,
    )
    print(format_horizon_table(aggregate))
    print(f"\nsaved: {args.out}")
    return aggregate


if __name__ == "__main__":
    main()
