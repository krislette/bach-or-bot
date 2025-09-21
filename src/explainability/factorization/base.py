"""
Base classes for audio factorization in MusicLIME explainability framework.

This module provides the foundational classes for decomposing audio into
interpretable components for explanation purposes.
"""

import warnings
import numpy as np
import librosa
from abc import ABC, abstractmethod
from typing import Optional, Union, List, Tuple, Dict, Any


def default_composition_fn(x: np.ndarray) -> np.ndarray:
    """
    Default composition function that returns input as-is.

    Parameters
    ----------
    x : np.ndarray
        Input audio array

    Returns
    -------
    np.ndarray
        Unmodified input array
    """
    return x


def load_audio(audio_path: str, target_sr: int) -> np.ndarray:
    """
    Load audio file and resample to target sample rate.

    Parameters
    ----------
    audio_path : str
        Path to audio file
    target_sr : int
        Target sample rate for resampling

    Returns
    -------
    np.ndarray
        Loaded and resampled audio waveform
    """
    waveform, _ = librosa.load(audio_path, mono=True, sr=target_sr)
    return waveform


def compute_segments(
    signal: np.ndarray,
    sr: int,
    temporal_segmentation_params: Optional[Union[Dict[str, Any], int]] = None,
) -> Tuple[List[Tuple[int, int]], int]:
    """
    Compute temporal segments for audio signal.

    Parameters
    ----------
    signal : np.ndarray
        Input audio signal
    sr : int
        Sample rate of the signal
    temporal_segmentation_params : dict or int, optional
        Segmentation parameters. If int, treated as number of segments.
        If dict, should contain 'type' and relevant parameters.

    Returns
    -------
    segments : List[Tuple[int, int]]
        List of (start_sample, end_sample) tuples defining segments
    explained_length : int
        Total length of audio that will be explained
    """
    audio_length = len(signal)
    explained_length = audio_length

    if temporal_segmentation_params is None:
        # Default: 1 segment per second, max 10 segments
        n_temporal_segments_default = min(audio_length // sr, 10)
        temporal_segmentation_params = {
            "type": "fixed_length",
            "n_temporal_segments": n_temporal_segments_default,
        }
    elif isinstance(temporal_segmentation_params, int):
        temporal_segmentation_params = {
            "type": "fixed_length",
            "n_temporal_segments": temporal_segmentation_params,
        }

    segmentation_type = temporal_segmentation_params["type"]
    assert segmentation_type in [
        "fixed_length",
        "manual",
    ], f"Segmentation type must be 'fixed_length' or 'manual', got {segmentation_type}"

    segments = []

    if segmentation_type == "fixed_length":
        n_temporal_segments = temporal_segmentation_params["n_temporal_segments"]
        samples_per_segment = audio_length // n_temporal_segments

        explained_length = samples_per_segment * n_temporal_segments
        if explained_length < audio_length:
            warnings.warn(f"Last {audio_length - explained_length} samples are ignored")

        for s in range(n_temporal_segments):
            segment_start = s * samples_per_segment
            segment_end = segment_start + samples_per_segment
            segments.append((segment_start, segment_end))

    elif segmentation_type == "manual":
        segments = temporal_segmentation_params["manual_segments"]
        explained_length = segments[-1][1]  # end of last segment

    return segments, explained_length


class Factorization(ABC):
    """
    Abstract base class for audio factorization methods.

    This class provides the framework for decomposing audio into interpretable
    components for explainability. Subclasses implement specific factorization
    approaches (e.g., temporal segmentation, source separation).

    Parameters
    ----------
    input : str or np.ndarray
        Audio input - either file path or audio array
    target_sr : int
        Target sample rate for processing
    temporal_segmentation_params : dict or int, optional
        Parameters for temporal segmentation
    composition_fn : callable, optional
        Function to compose selected components back to audio
    """

    def __init__(
        self,
        input: Union[str, np.ndarray],
        target_sr: int,
        temporal_segmentation_params: Optional[Union[Dict[str, Any], int]] = None,
        composition_fn: Optional[callable] = None,
    ):
        self._audio_path = None
        self.target_sr = target_sr

        # Handle input - either file path or audio array
        if isinstance(input, str):
            self._audio_path = input
            input = load_audio(input, target_sr)
        self._original_mix = input

        if composition_fn is None:
            composition_fn = default_composition_fn
        self._composition_fn = composition_fn

        # Initialize component storage
        self.original_components = []
        self.components = []
        self._components_names = []

        # Compute temporal segmentation
        self.temporal_segments, self.explained_length = compute_segments(
            self._original_mix, self.target_sr, temporal_segmentation_params
        )

    def compose_model_input(self, components: Optional[List[int]] = None) -> np.ndarray:
        """
        Compose audio from selected components.

        Parameters
        ----------
        components : List[int], optional
            Indices of components to include. If None, uses all components.

        Returns
        -------
        np.ndarray
            Composed audio signal
        """
        return self._composition_fn(self.retrieve_components(components))

    def get_number_components(self) -> int:
        """
        Get the total number of components.

        Returns
        -------
        int
            Number of available components
        """
        return len(self._components_names)

    @abstractmethod
    def retrieve_components(
        self, selection_order: Optional[List[int]] = None
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Retrieve selected components.

        Parameters
        ----------
        selection_order : List[int], optional
            Indices of components to retrieve

        Returns
        -------
        np.ndarray or List[np.ndarray]
            Selected audio components
        """
        raise NotImplementedError("Subclasses must implement retrieve_components")

    def get_ordered_component_names(self) -> List[str]:
        """
        Get ordered list of component names.

        Returns
        -------
        List[str]
            Component names (e.g., instrument names, time segment labels)

        Raises
        ------
        Exception
            If components were not named
        """
        if len(self._components_names) == 0:
            raise Exception("Components were not named.")
        return self._components_names
