"""
Temporal factorization module for MusicLIME explainability.

This module contains the TimeOnlyFactorization class that handles temporal perturbations
of audio signals by segmenting audio into time-based components for explanation.
"""

import numpy as np
import librosa
from typing import List, Tuple, Optional
from .base import Factorization


class TimeOnlyFactorization(Factorization):
    """
    Factorization class that segments audio into temporal components only.

    This factorization divides the audio signal into time-based segments of equal
    duration, allowing LIME to perturb individual time segments to understand
    their contribution to the model's prediction.

    Attributes:
        segment_duration (float): Duration of each temporal segment in seconds
        hop_length (int): Hop length for audio processing
        sr (int): Sample rate of the audio
        n_segments (int): Number of temporal segments created from the audio
    """

    def __init__(self, segment_duration: float = 1.0, hop_length: int = 512):
        """
        Initialize the TimeOnlyFactorization.

        Args:
            segment_duration (float): Duration of each temporal segment in seconds.
                                    Default is 1.0 second.
            hop_length (int): Hop length for audio processing. Default is 512.
        """
        super().__init__()
        self.segment_duration = segment_duration
        self.hop_length = hop_length
        self.sr = None
        self.n_segments = None
        self._segment_boundaries = None

    def factorize(
        self, audio_path: str, sr: Optional[int] = None
    ) -> Tuple[np.ndarray, List[dict]]:
        """
        Factorize audio into temporal segments.

        This method loads the audio file and segments it into equal-duration
        time segments that can be independently perturbed during explanation.

        Args:
            audio_path (str): Path to the audio file to factorize
            sr (Optional[int]): Target sample rate. If None, uses librosa default

        Returns:
            Tuple[np.ndarray, List[dict]]: A tuple containing:
                - audio (np.ndarray): The loaded audio signal with shape (n_samples,)
                - segments (List[dict]): List of segment metadata, each containing:
                    - 'start_time': Start time of segment in seconds
                    - 'end_time': End time of segment in seconds
                    - 'start_sample': Start sample index
                    - 'end_sample': End sample index
                    - 'segment_id': Unique identifier for the segment

        Raises:
            FileNotFoundError: If the audio file cannot be found
            ValueError: If the audio file cannot be loaded or is empty
        """
        # Load audio file
        try:
            audio, self.sr = librosa.load(audio_path, sr=sr)
        except Exception as e:
            raise FileNotFoundError(f"Could not load audio file {audio_path}: {str(e)}")

        if len(audio) == 0:
            raise ValueError(f"Loaded audio file {audio_path} is empty")

        # Calculate segment parameters
        samples_per_segment = int(self.segment_duration * self.sr)
        total_samples = len(audio)
        self.n_segments = int(np.ceil(total_samples / samples_per_segment))

        # Create segment metadata
        segments = []
        self._segment_boundaries = []

        for i in range(self.n_segments):
            start_sample = i * samples_per_segment
            end_sample = min((i + 1) * samples_per_segment, total_samples)
            start_time = start_sample / self.sr
            end_time = end_sample / self.sr

            segment_info = {
                "start_time": start_time,
                "end_time": end_time,
                "start_sample": start_sample,
                "end_sample": end_sample,
                "segment_id": i,
                "type": "temporal",
            }

            segments.append(segment_info)
            self._segment_boundaries.append((start_sample, end_sample))

        return audio, segments

    def perturb(
        self, audio: np.ndarray, segments: List[dict], active_segments: List[int]
    ) -> np.ndarray:
        """
        Create a perturbed version of the audio with only active segments.

        This method creates a new audio signal where only the segments specified
        in active_segments contain the original audio, while inactive segments
        are replaced with silence.

        Args:
            audio (np.ndarray): Original audio signal with shape (n_samples,)
            segments (List[dict]): List of segment metadata from factorize()
            active_segments (List[int]): List of segment IDs that should remain active

        Returns:
            np.ndarray: Perturbed audio signal with same shape as input, where
                       inactive segments are replaced with silence

        Raises:
            ValueError: If segment IDs in active_segments are invalid
        """
        if len(active_segments) == 0:
            # Return silence if no segments are active
            return np.zeros_like(audio)

        # Validate active_segments
        max_segment_id = len(segments) - 1
        invalid_segments = [
            seg_id
            for seg_id in active_segments
            if seg_id < 0 or seg_id > max_segment_id
        ]
        if invalid_segments:
            raise ValueError(
                f"Invalid segment IDs: {invalid_segments}. "
                f"Valid range is 0 to {max_segment_id}"
            )

        # Create perturbed audio (start with silence)
        perturbed_audio = np.zeros_like(audio)

        # Activate only the specified segments
        for segment_id in active_segments:
            segment = segments[segment_id]
            start_sample = segment["start_sample"]
            end_sample = segment["end_sample"]
            perturbed_audio[start_sample:end_sample] = audio[start_sample:end_sample]

        return perturbed_audio

    def get_segment_info(self, segment_id: int, segments: List[dict]) -> dict:
        """
        Get detailed information about a specific segment.

        Args:
            segment_id (int): ID of the segment to get information for
            segments (List[dict]): List of segment metadata from factorize()

        Returns:
            dict: Detailed segment information including timing and sample indices

        Raises:
            ValueError: If segment_id is invalid
        """
        if segment_id < 0 or segment_id >= len(segments):
            raise ValueError(
                f"Invalid segment_id {segment_id}. "
                f"Valid range is 0 to {len(segments) - 1}"
            )

        return segments[segment_id].copy()

    def get_total_duration(self) -> Optional[float]:
        """
        Get the total duration of the audio in seconds.

        Returns:
            Optional[float]: Total duration in seconds, or None if no audio has been processed
        """
        if self.sr is None or self.n_segments is None:
            return None

        # Calculate from last segment boundary
        if self._segment_boundaries:
            last_boundary = self._segment_boundaries[-1]
            return last_boundary[1] / self.sr

        return None

    def __str__(self) -> str:
        """String representation of the factorization."""
        if self.n_segments is None:
            return f"TimeOnlyFactorization(segment_duration={self.segment_duration}s, not fitted)"

        return (
            f"TimeOnlyFactorization(segment_duration={self.segment_duration}s, "
            f"n_segments={self.n_segments}, total_duration={self.get_total_duration():.2f}s)"
        )

    def __repr__(self) -> str:
        """Detailed representation of the factorization."""
        return self.__str__()
