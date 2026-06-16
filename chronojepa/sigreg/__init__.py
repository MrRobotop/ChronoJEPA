"""SIGReg objective: univariate normality tests, random slicing, and placement variants."""

from .placements import PooledSIGReg
from .slicing import random_directions, sliced_sigreg
from .univariate import epps_pulley_statistic

__all__ = [
    "PooledSIGReg",
    "epps_pulley_statistic",
    "random_directions",
    "sliced_sigreg",
]
