"""
LIME module for MusicLIME explainability.

This module provides LIME-based explanation capabilities for multimodal
music classification models using both audio and textual features.
"""

from .base import LimeBase
from .explainer import LimeMusicExplainer

__all__ = ["LimeBase", "LimeMusicExplainer"]
