"""Encoders (PatchTST, TCN), an optional MLP predictor, and RevIN.

Shared encoder contract: a forward call maps ``(B, C, T)`` to ``(tokens, pooled)`` where
``tokens`` is ``(B, L, D)`` indexed by time and ``pooled`` is ``(B, D)``.
"""

from .patchtst import PatchTSTEncoder
from .revin import RevIN
from .tcn import TCNEncoder

__all__ = ["PatchTSTEncoder", "RevIN", "TCNEncoder"]
