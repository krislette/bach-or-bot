"""
Fixed source separation factorization that works properly with LIME.
"""

import numpy as np
import librosa
import torch
import openunmix
from typing import List, Optional, Union, Dict, Any
from .base import Factorization


class OpenunmixFactorization(Factorization):
    """
    Fixed OpenUnmix factorization that properly integrates with LIME.

    This version follows the original MusicLIME pattern more closely.
    """

    def __init__(
        self,
        input: Union[str, np.ndarray],
        target_sr: int = 44100,
        temporal_segmentation_params: Optional[Union[Dict[str, Any], int]] = None,
        composition_fn: Optional[callable] = None,
        model_name: str = "umxhq",
        device: Optional[str] = None,
    ):
        # Initialize base class first
        super().__init__(
            input=input,
            target_sr=target_sr,
            temporal_segmentation_params=temporal_segmentation_params,
            composition_fn=composition_fn,
        )

        self.model_name = model_name
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = None

        # Initialize components using source separation
        self.original_components, self._components_names = self.initialize_components()
        self.prepare_components(0, len(self._original_mix))

    def initialize_components(self):
        """
        Initialize separated sources using OpenUnmix.

        This method follows the original MusicLIME pattern.
        """
        # Load model if not already loaded
        if self.model is None:
            try:
                self.model = openunmix.load_model(self.model_name, device=self.device)
                self.model.eval()
            except Exception as e:
                raise RuntimeError(f"Failed to load OpenUnmix model: {e}")

        # Prepare audio for OpenUnmix (needs stereo)
        waveform = self._original_mix
        if waveform.ndim == 1:
            # Convert mono to stereo
            waveform_stereo = np.stack([waveform, waveform], axis=0)
        else:
            waveform_stereo = waveform.T if waveform.shape[0] != 2 else waveform

        # Perform source separation
        try:
            # Convert to tensor and add batch dimension
            audio_tensor = torch.from_numpy(waveform_stereo).float().unsqueeze(0)
            audio_tensor = audio_tensor.to(self.device)

            with torch.no_grad():
                separated = self.model(audio_tensor)

            # Convert back to numpy and extract mono sources
            separated_np = separated.cpu().numpy()  # (1, n_sources, 2, n_samples)

            original_components = []
            component_names = ["vocals", "drums", "bass", "other"]

            for i in range(separated_np.shape[1]):
                # Convert stereo to mono by averaging channels
                stereo_source = separated_np[0, i, :, :]  # (2, n_samples)
                mono_source = np.mean(stereo_source, axis=0)  # (n_samples,)

                # Resample if necessary
                if self.target_sr != 44100:
                    mono_source = librosa.resample(
                        mono_source, orig_sr=44100, target_sr=self.target_sr
                    )

                original_components.append(mono_source)

            return original_components, component_names

        except Exception as e:
            raise RuntimeError(f"Source separation failed: {e}")

    def prepare_components(self, start_sample: int, y_length: int):
        """
        Prepare components with temporal segmentation.

        This creates the final component list that LIME will use.
        """
        # Reset components to original sources
        self.components = [
            comp[start_sample : start_sample + y_length]
            for comp in self.original_components
        ]

        # Create temporally segmented components
        component_names = []
        temporary_components = []

        for s, (segment_start, segment_end) in enumerate(self.temporal_segments):
            for co in range(len(self.original_components)):
                # Create component that's only active in this time segment
                current_component = np.zeros(self.explained_length, dtype=np.float32)
                current_component[segment_start:segment_end] = self.components[co][
                    segment_start:segment_end
                ]

                temporary_components.append(current_component)
                component_names.append(f"{self._components_names[co]}_t{s}")

        self.components = temporary_components
        self._components_names = component_names

    def retrieve_components(
        self, selection_order: Optional[List[int]] = None
    ) -> np.ndarray:
        """
        Retrieve and compose selected components.

        This is the method LIME calls to create perturbed audio.
        """
        if selection_order is None:
            # Return original mix if no selection
            return self._original_mix

        if len(selection_order) == 0:
            # Return silence if no components selected
            return np.zeros_like(self._original_mix[: self.explained_length])

        # Sum selected components
        composed_audio = np.zeros(self.explained_length, dtype=np.float32)

        for component_idx in selection_order:
            if 0 <= component_idx < len(self.components):
                composed_audio += self.components[component_idx]

        return composed_audio

    def compose_model_input(self, components: Optional[List[int]] = None) -> np.ndarray:
        """
        Compose model input from selected components.

        This is what gets passed to your classifier.
        """
        composed = self.retrieve_components(components)
        return self._composition_fn(composed)

    def get_separated_source(self, source_name: str) -> Optional[np.ndarray]:
        """Get a specific separated source by name."""
        source_names = ["vocals", "drums", "bass", "other"]
        if source_name not in source_names:
            return None

        source_idx = source_names.index(source_name)
        if source_idx < len(self.original_components):
            return self.original_components[source_idx].copy()
        return None

    def get_all_separated_sources(self) -> Dict[str, np.ndarray]:
        """Get all separated sources as a dictionary."""
        source_names = ["vocals", "drums", "bass", "other"]
        sources = {}

        for i, name in enumerate(source_names):
            if i < len(self.original_components):
                sources[name] = self.original_components[i].copy()

        return sources

    def __str__(self) -> str:
        return f"OpenunmixFactorization(model={self.model_name}, n_components={self.get_number_components()})"
