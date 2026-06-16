"""SIGReg objective: univariate normality tests, random slicing, and placement variants."""

from .placements import DualSIGReg, PooledSIGReg, StructuredSIGReg, make_sigreg
from .slicing import random_directions, sliced_sigreg
from .univariate import epps_pulley_statistic

__all__ = [
    "DualSIGReg",
    "PooledSIGReg",
    "StructuredSIGReg",
    "epps_pulley_statistic",
    "make_sigreg",
    "random_directions",
    "sliced_sigreg",
]
