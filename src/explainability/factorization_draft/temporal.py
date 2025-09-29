"""
Temporal factorization module for MusicLIME explainability.

This module contains the TimeOnlyFactorization class that handles temporal perturbations
of audio signals by segmenting audio into time-based components for explanation.
"""

import numpy as np
from typing import List, Optional, Dict, Any, Union
from .base import Factorization


class TimeOnlyFactorization(Factorization):
    def __init__(
        self,
        input: Union[str, np.ndarray],
        target_sr: int = 22050,
        temporal_segmentation_params: Optional[Union[Dict[str, Any], int]] = None,
        composition_fn: Optional[callable] = None,
    ):
        # Initialize the base class FIRST
        super().__init__(
            input=input,
            target_sr=target_sr,
            temporal_segmentation_params=temporal_segmentation_params,
            composition_fn=composition_fn,
        )

        try:
            # Prefer the readable names returned by the method
            self._components_names = self.get_component_names()
        except Exception:
            # Fallback to generic names if anything goes wrong
            self._components_names = [
                f"segment_{i}" for i in range(len(self.temporal_segments))
            ]

        # Now initialize subclass-specific attributes
        self.segment_duration = None
        if (
            temporal_segmentation_params
            and isinstance(temporal_segmentation_params, dict)
            and "segment_duration" in temporal_segmentation_params
        ):
            self.segment_duration = temporal_segmentation_params["segment_duration"]

    def retrieve_components(
        self, selection_order: Optional[List[int]] = None
    ) -> np.ndarray:
        """Retrieve selected temporal segments."""
        if selection_order is None:
            selection_order = list(range(len(self.temporal_segments)))

        # Create silent audio
        perturbed_audio = np.zeros(self.explained_length)

        # Add selected segments
        for segment_idx in selection_order:
            if 0 <= segment_idx < len(self.temporal_segments):
                start, end = self.temporal_segments[segment_idx]
                segment_length = end - start

                # Ensure we don't exceed bounds
                if start < len(self._original_mix) and segment_length > 0:
                    actual_end = min(end, len(self._original_mix))
                    actual_length = actual_end - start
                    perturbed_audio[start : start + actual_length] = self._original_mix[
                        start:actual_end
                    ]

        return perturbed_audio

    def get_component_names(self) -> List[str]:
        """Get names for temporal segments."""
        names = []
        for i, (start, end) in enumerate(self.temporal_segments):
            start_time = start / self.target_sr
            end_time = end / self.target_sr
            names.append(f"segment_{i}_{start_time:.1f}s-{end_time:.1f}s")
        return names
