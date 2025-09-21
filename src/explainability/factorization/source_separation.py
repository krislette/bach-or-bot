"""
Source separation factorization module for MusicLIME explainability.

This module contains the OpenunmixFactorization class that uses OpenUnmix
for source separation to create instrument-based components for explanation.
"""

import numpy as np
import librosa
import torch
import openunmix
from typing import List, Tuple, Optional, Dict
from .base import Factorization


class OpenunmixFactorization(Factorization):
    """
    Factorization class that separates audio into instrument sources using OpenUnmix.

    This factorization uses the OpenUnmix deep learning model to separate a music
    track into individual instrument sources (vocals, drums, bass, other), allowing
    LIME to perturb individual instruments to understand their contribution.
    """

    def __init__(
        self,
        model_name: str = "umxhq",
        device: Optional[str] = None,
        chunk_duration: float = 10.0,
    ):
        """
        Initialize the OpenunmixFactorization.

        Parameters
        ----------
        model_name : str, default="umxhq"
            OpenUnmix model variant to use ("umxhq" for high quality)
        device : Optional[str], default=None
            Device to run inference on. If None, auto-detects CUDA availability
        chunk_duration : float, default=10.0
            Duration in seconds for processing chunks (for memory efficiency)
        """
        super().__init__()
        self.model_name = model_name
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.chunk_duration = chunk_duration
        self.model = None
        self.sr = None
        self.component_names = ["vocals", "drums", "bass", "other"]
        self._separated_sources = None

    def _load_model(self):
        """
        Load the OpenUnmix model.

        Raises
        ------
        RuntimeError
            If model fails to load
        """
        try:
            self.model = openunmix.load_model(self.model_name, device=self.device)
            self.model.eval()
        except Exception as e:
            raise RuntimeError(
                f"Failed to load OpenUnmix model {self.model_name}: {str(e)}"
            )

    def factorize(
        self, audio_path: str, sr: Optional[int] = None
    ) -> Tuple[np.ndarray, List[dict]]:
        """
        Factorize audio into separated instrument sources.

        Parameters
        ----------
        audio_path : str
            Path to the audio file to factorize
        sr : Optional[int], default=None
            Target sample rate. If None, uses 44100 Hz (OpenUnmix default)

        Returns
        -------
        Tuple[np.ndarray, List[dict]]
            A tuple containing:
            - audio: The original mixed audio signal with shape (n_samples,)
            - segments: List of source metadata, each containing:
                - 'source_name': Name of the separated source
                - 'source_id': Unique identifier for the source
                - 'type': Always 'source_separation'

        Raises
        ------
        FileNotFoundError
            If the audio file cannot be found
        ValueError
            If the audio file cannot be loaded or is empty
        RuntimeError
            If source separation fails
        """
        # Load model if not already loaded
        if self.model is None:
            self._load_model()

        # Set sample rate (OpenUnmix works best at 44100 Hz)
        target_sr = sr if sr is not None else 44100

        # Load audio file
        try:
            audio, self.sr = librosa.load(audio_path, sr=target_sr)
        except Exception as e:
            raise FileNotFoundError(f"Could not load audio file {audio_path}: {str(e)}")

        if len(audio) == 0:
            raise ValueError(f"Loaded audio file {audio_path} is empty")

        # Convert mono to stereo if necessary (OpenUnmix expects stereo)
        if audio.ndim == 1:
            audio_stereo = np.stack([audio, audio], axis=0)  # (2, n_samples)
        else:
            audio_stereo = audio.T  # Ensure (2, n_samples) format

        # Perform source separation
        try:
            separated_sources = self._separate_sources(audio_stereo)
            self._separated_sources = separated_sources
        except Exception as e:
            raise RuntimeError(f"Source separation failed: {str(e)}")

        # Create segment metadata for each source
        segments = []
        for i, source_name in enumerate(self.component_names):
            segment_info = {
                "source_name": source_name,
                "source_id": i,
                "type": "source_separation",
            }
            segments.append(segment_info)

        # Return mono version of original audio for consistency
        return audio, segments

    def _separate_sources(self, audio_stereo: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Perform source separation on stereo audio.

        Parameters
        ----------
        audio_stereo : np.ndarray
            Stereo audio with shape (2, n_samples)

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary mapping source names to separated audio arrays

        Raises
        ------
        RuntimeError
            If separation process fails
        """
        try:
            # Convert to torch tensor
            audio_tensor = (
                torch.from_numpy(audio_stereo).float().unsqueeze(0)
            )  # (1, 2, n_samples)
            audio_tensor = audio_tensor.to(self.device)

            # Process in chunks if audio is long
            n_samples = audio_tensor.shape[-1]
            chunk_samples = int(self.chunk_duration * self.sr)

            if n_samples <= chunk_samples:
                # Process entire audio at once
                with torch.no_grad():
                    separated = self.model(audio_tensor)
            else:
                # Process in chunks and concatenate
                separated_chunks = []
                n_chunks = int(np.ceil(n_samples / chunk_samples))

                for i in range(n_chunks):
                    start_idx = i * chunk_samples
                    end_idx = min((i + 1) * chunk_samples, n_samples)
                    chunk = audio_tensor[:, :, start_idx:end_idx]

                    with torch.no_grad():
                        chunk_separated = self.model(chunk)
                    separated_chunks.append(chunk_separated)

                # Concatenate chunks
                separated = torch.cat(separated_chunks, dim=-1)

            # Convert back to numpy and organize by source
            separated_np = separated.cpu().numpy()  # (1, n_sources, 2, n_samples)
            separated_sources = {}

            for i, source_name in enumerate(self.component_names):
                # Convert stereo to mono by averaging channels
                stereo_source = separated_np[0, i, :, :]  # (2, n_samples)
                mono_source = np.mean(stereo_source, axis=0)  # (n_samples,)
                separated_sources[source_name] = mono_source

            return separated_sources

        except Exception as e:
            raise RuntimeError(f"OpenUnmix inference failed: {str(e)}")

    def perturb(
        self, audio: np.ndarray, segments: List[dict], active_segments: List[int]
    ) -> np.ndarray:
        """
        Create a perturbed version of audio with only active sources.

        Parameters
        ----------
        audio : np.ndarray
            Original mixed audio signal with shape (n_samples,)
        segments : List[dict]
            List of source metadata from factorize()
        active_segments : List[int]
            List of source IDs that should remain active

        Returns
        -------
        np.ndarray
            Perturbed audio signal created by mixing only active sources

        Raises
        ------
        ValueError
            If segment IDs in active_segments are invalid
        RuntimeError
            If separated sources are not available
        """
        if self._separated_sources is None:
            raise RuntimeError("No separated sources available. Run factorize() first.")

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

        # Mix only the active sources
        perturbed_audio = np.zeros_like(audio)

        for segment_id in active_segments:
            source_name = segments[segment_id]["source_name"]
            source_audio = self._separated_sources[source_name]

            # Ensure same length as original audio
            min_length = min(len(perturbed_audio), len(source_audio))
            perturbed_audio[:min_length] += source_audio[:min_length]

        return perturbed_audio

    def get_source_info(self, source_id: int, segments: List[dict]) -> dict:
        """
        Get detailed information about a specific source.

        Parameters
        ----------
        source_id : int
            ID of the source to get information for
        segments : List[dict]
            List of source metadata from factorize()

        Returns
        -------
        dict
            Detailed source information

        Raises
        ------
        ValueError
            If source_id is invalid
        """
        if source_id < 0 or source_id >= len(segments):
            raise ValueError(
                f"Invalid source_id {source_id}. "
                f"Valid range is 0 to {len(segments) - 1}"
            )

        return segments[source_id].copy()

    def get_separated_source(self, source_name: str) -> Optional[np.ndarray]:
        """
        Get the separated audio for a specific source.

        Parameters
        ----------
        source_name : str
            Name of the source ('vocals', 'drums', 'bass', 'other')

        Returns
        -------
        Optional[np.ndarray]
            Separated source audio, or None if not available

        Raises
        ------
        ValueError
            If source_name is invalid
        """
        if source_name not in self.component_names:
            raise ValueError(
                f"Invalid source_name '{source_name}'. "
                f"Valid options: {self.component_names}"
            )

        if self._separated_sources is None:
            return None

        return self._separated_sources[source_name].copy()

    def get_all_separated_sources(self) -> Optional[Dict[str, np.ndarray]]:
        """
        Get all separated sources.

        Returns
        -------
        Optional[Dict[str, np.ndarray]]
            Dictionary mapping source names to audio arrays, or None if not available
        """
        if self._separated_sources is None:
            return None

        return {name: audio.copy() for name, audio in self._separated_sources.items()}

    def __str__(self) -> str:
        """String representation of the factorization."""
        status = "fitted" if self._separated_sources is not None else "not fitted"
        return f"OpenunmixFactorization(model={self.model_name}, device={self.device}, {status})"

    def __repr__(self) -> str:
        """Detailed representation of the factorization."""
        return self.__str__()
