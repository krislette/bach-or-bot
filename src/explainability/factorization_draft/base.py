# factorization/base.py
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple
import numpy as np
import warnings


class Factorization(ABC):
    """Abstract base for factorizing inputs (audio, text, multimodal)."""

    def __init__(self, input_data: Any):
        self.input_data = input_data
        self.components = None  # to be set by subclass

    @abstractmethod
    def factorize(self) -> List[Any]:
        """Split input_data into interpretable components."""
        pass

    @abstractmethod
    def reconstruct(self, mask: np.ndarray) -> Any:
        """
        Reconstruct the input given a binary mask.
        Mask shape should match number of components.
        """
        pass

    def num_components(self) -> int:
        if self.components is None:
            raise ValueError("factorize() must be called before num_components().")
        return len(self.components)


class TemporalFactorization(Factorization):
    """
    Simple factorizer that splits audio into temporal segments.
    """

    def __init__(self, input_data: np.ndarray, sr: int, segment_duration: float = 1.0):
        super().__init__(input_data)
        self.sr = sr
        self.segment_duration = segment_duration

    def factorize(self) -> List[np.ndarray]:
        samples_per_segment = int(self.sr * self.segment_duration)
        self.components = [
            self.input_data[i : i + samples_per_segment]
            for i in range(0, len(self.input_data), samples_per_segment)
        ]
        return self.components

    def reconstruct(self, mask: np.ndarray) -> np.ndarray:
        if self.components is None:
            raise ValueError("Must call factorize() before reconstruct().")
        if len(mask) != len(self.components):
            raise ValueError("Mask length must equal number of components.")

        # Rebuild by zeroing out masked segments
        reconstructed = np.zeros_like(self.input_data)
        idx = 0
        for keep, comp in zip(mask, self.components):
            length = len(comp)
            if keep:
                reconstructed[idx : idx + length] = comp
            idx += length
        return reconstructed
