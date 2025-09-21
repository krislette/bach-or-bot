"""
Factorization module for MusicLIME explainability.

This module provides different factorization strategies for decomposing
audio signals into interpretable components for explanation.
"""

from .base import Factorization
from .temporal import TimeOnlyFactorization
from .source_separation import OpenunmixFactorization

__all__ = ["Factorization", "TimeOnlyFactorization", "OpenunmixFactorization"]
