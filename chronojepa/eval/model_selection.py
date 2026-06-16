"""Label-free model selection: rank runs by SIGReg loss, correlate with the metric.

LeJEPA reports that the final SIGReg training loss rank-correlates with downstream
performance, which would let us select checkpoints and architectures without labels. This
utility measures that correlation on time series. It is pure analysis: it reads metrics
that were logged elsewhere and never retrains.
"""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr


def label_free_model_selection(
    names: Sequence[str],
    sigreg_losses: Sequence[float],
    downstream_metrics: Sequence[float],
    *,
    lower_is_better: bool = True,
) -> dict[str, Any]:
    """Rank runs by final SIGReg loss and correlate with the labeled downstream metric.

    Returns the Spearman correlation, the label-free pick (lowest SIGReg loss), the
    label-based pick (best downstream metric), and whether they agree.
    """
    names = list(names)
    sigreg = np.asarray(sigreg_losses, dtype=float)
    downstream = np.asarray(downstream_metrics, dtype=float)
    if len(names) < 3:
        raise ValueError("need at least three runs to estimate a rank correlation")

    rho, pvalue = spearmanr(sigreg, downstream)
    label_free = int(np.argmin(sigreg))
    label_based = int(np.argmin(downstream) if lower_is_better else np.argmax(downstream))
    return {
        "spearman": float(rho),
        "pvalue": float(pvalue),
        "label_free_pick": names[label_free],
        "label_based_pick": names[label_based],
        "agree": names[label_free] == names[label_based],
    }


def selection_report(result: dict[str, Any]) -> str:
    """Render the selection result as a short human-readable report."""
    return (
        f"Spearman correlation (sigreg loss vs downstream): {result['spearman']:.3f} "
        f"(p={result['pvalue']:.3f})\n"
        f"label-free pick (lowest sigreg loss): {result['label_free_pick']}\n"
        f"label-based pick (best downstream):   {result['label_based_pick']}\n"
        f"picks agree: {result['agree']}"
    )


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """Thin CLI: read a JSON list of runs and print the selection report."""
    parser = argparse.ArgumentParser(description="Label-free model selection by SIGReg loss")
    parser.add_argument("runs", help="JSON list of {name, sigreg_loss, downstream_metric}")
    parser.add_argument(
        "--higher-is-better",
        action="store_true",
        help="set when a larger downstream metric is better (for example accuracy)",
    )
    args = parser.parse_args(argv)

    runs = json.loads(Path(args.runs).read_text())
    result = label_free_model_selection(
        [r["name"] for r in runs],
        [r["sigreg_loss"] for r in runs],
        [r["downstream_metric"] for r in runs],
        lower_is_better=not args.higher_is_better,
    )
    print(selection_report(result))
    return result


if __name__ == "__main__":
    main()
