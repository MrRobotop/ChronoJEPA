"""Smoke test: the package imports and exposes a version string."""

import chronojepa


def test_package_exposes_version() -> None:
    assert isinstance(chronojepa.__version__, str)
    assert chronojepa.__version__
