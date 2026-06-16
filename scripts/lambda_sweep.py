"""Lambda sweep: how the SIGReg-to-invariance balance trades collapse against downstream,
and whether final SIGReg loss selects models without labels.

Synthetic by default:

    uv run python scripts/lambda_sweep.py

On PEMS:

    uv run python scripts/lambda_sweep.py --pems data/pems08.npz --seeds 3
"""

import argparse

import numpy as np

from chronojepa.data import load_pems
from chronojepa.eval import format_lambda_table, run_lambda_sweep


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
    parser = argparse.ArgumentParser(description="Lambda sweep with label-free selection")
    parser.add_argument("--pems", help="path to a PEMS .npz; uses synthetic data if omitted")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--lambdas", default="0.1,0.3,0.5,0.7,0.9")
    parser.add_argument("--out", default="results/lambda_sweep.json")
    args = parser.parse_args(argv)

    series = load_pems(args.pems) if args.pems else _synthetic_series()
    lambdas = tuple(float(x) for x in args.lambdas.split(","))
    out = run_lambda_sweep(
        series,
        lambdas=lambdas,
        seeds=tuple(range(args.seeds)),
        steps=args.steps,
        results_path=args.out,
    )
    print(format_lambda_table(out))
    print(f"\nsaved: {args.out}")
    return out


if __name__ == "__main__":
    main()
