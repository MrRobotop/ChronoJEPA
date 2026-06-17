"""Lightweight run logging. W&B is optional and online logging is opt-in."""

from typing import Any


class RunLogger:
    """Collects metrics in memory and prints periodically.

    This is the default logger, so a run needs no external account. It keeps a history
    of logged metrics that callers and tests can inspect.
    """

    def __init__(self, print_every: int = 10) -> None:
        self.print_every = print_every
        self.history: list[dict[str, Any]] = []

    def log(self, metrics: dict[str, Any], step: int) -> None:
        record = {"step": step, **metrics}
        self.history.append(record)
        if self.print_every and step % self.print_every == 0:
            terms = " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            print(f"step {step}: {terms}")

    def finish(self) -> None:
        pass


class WandbLogger:
    """Opt-in Weights and Biases logger. Imports wandb lazily and defaults to offline."""

    def __init__(
        self,
        project: str = "chronojepa",
        mode: str = "offline",
        config: dict | None = None,
        name: str | None = None,
        group: str | None = None,
        tags: list[str] | None = None,
    ) -> None:
        import wandb  # noqa: PLC0415  (lazy: keeps wandb an optional dependency)

        self.run = wandb.init(
            project=project, mode=mode, config=config, name=name, group=group, tags=tags
        )
        self.history: list[dict[str, Any]] = []

    def log(self, metrics: dict[str, Any], step: int) -> None:
        self.history.append({"step": step, **metrics})
        self.run.log(metrics, step=step)

    def finish(self) -> None:
        self.run.finish()
