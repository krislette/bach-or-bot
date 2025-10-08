import numpy as np
import torch
from openunmix import predict


class OpenUnmixFactorization:
    def __init__(self, audio, temporal_segmentation_params=10, composition_fn=None):
        print(f"[MusicLIME] Initializing OpenUnmix factorization...")
        self.audio = audio
        self.target_sr = 44100

        print(
            f"[MusicLIME] Computing {temporal_segmentation_params} temporal segments..."
        )
        self.temporal_segments = self._compute_segments(
            audio, temporal_segmentation_params
        )

        # Initialize source separation
        print("[MusicLIME] Separating audio sources...")
        self.original_components, self.component_names = self._separate_sources()
        print(f"[MusicLIME] Found components: {self.component_names}")

        print("[MusicLIME] Preparing temporal-source combinations...")
        self._prepare_temporal_components()
        print(f"[MusicLIME] Created {len(self.components)} total components")

    def _compute_segments(self, signal, n_segments):
        audio_length = len(signal)
        samples_per_segment = audio_length // n_segments

        segments = []
        for i in range(n_segments):
            start = i * samples_per_segment
            end = start + samples_per_segment
            segments.append((start, end))
        return segments

    def _separate_sources(self):
        waveform = np.expand_dims(self.audio, axis=1)
        prediction = predict.separate(torch.as_tensor(waveform).float(), rate=44100)

        components = [prediction[key][0].mean(dim=0).numpy() for key in prediction]
        names = list(prediction.keys())
        return components, names

    def _prepare_temporal_components(self):
        # Create temporal-source combinations
        self.components = []
        self.final_component_names = []

        for s, (start, end) in enumerate(self.temporal_segments):
            for c, component in enumerate(self.original_components):
                temp_component = np.zeros_like(self.audio)
                temp_component[start:end] = component[start:end]
                self.components.append(temp_component)
                self.final_component_names.append(f"{self.component_names[c]}_T{s}")

    def get_number_components(self):
        return len(self.components)

    def get_ordered_component_names(self):
        return self.final_component_names

    def compose_model_input(self, component_indices):
        if len(component_indices) == 0:
            return np.zeros_like(self.audio)

        selected_components = [self.components[i] for i in component_indices]
        return sum(selected_components)
