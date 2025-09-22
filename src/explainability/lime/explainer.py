"""MusicLIME explainer implementation."""

import numpy as np
from typing import List, Dict, Callable, Optional, Any
import warnings

from .base import LimeBase
from ..factorization.base import Factorization
from ..factorization.temporal import TimeOnlyFactorization
from ..factorization.lyrics import LyricsFactorization


class LimeMusicExplainer:
    def __init__(
        self,
        audio_factorization: Factorization,
        kernel_width: float = 0.25,
        verbose: bool = False,
        random_state: Optional[int] = None,
    ):
        self.audio_factorization = audio_factorization
        self.lime_base = LimeBase(
            kernel_width=kernel_width, verbose=verbose, random_state=random_state
        )
        self.verbose = verbose
        self.random_state = random_state

    def _create_perturbation_function(
        self,
        classifier_fn: Callable,
        audio_factorizer: Factorization,
        lyrics_factorizer: "LyricsFactorization",
    ) -> Callable:
        """Create perturbation function that handles both modalities correctly."""

        def perturbation_fn(binary_mask: np.ndarray) -> float:
            # Split mask for audio and lyrics
            n_audio_components = audio_factorizer.get_number_components()
            n_lyrics_components = lyrics_factorizer.get_number_components()

            audio_mask = binary_mask[:n_audio_components]
            lyrics_mask = binary_mask[
                n_audio_components : n_audio_components + n_lyrics_components
            ]

            # Get active indices
            active_audio_indices = [i for i, active in enumerate(audio_mask) if active]
            active_lyrics_indices = [
                i for i, active in enumerate(lyrics_mask) if active
            ]

            # Create perturbed audio
            perturbed_audio = audio_factorizer.compose_model_input(active_audio_indices)

            # Create perturbed lyrics
            perturbed_lyrics = lyrics_factorizer.compose_model_input(
                active_lyrics_indices
            )

            # Get prediction
            try:
                prediction = classifier_fn(perturbed_audio, perturbed_lyrics)
                return prediction
            except Exception as e:
                if self.verbose:
                    warnings.warn(f"Classifier error: {str(e)}")
                return 0.5  # Neutral prediction

        return perturbation_fn

    def explain_instance(
        self,
        audio_path: str,
        lyrics_lines: List[str],
        classifier_fn: Callable,
        n_samples: int = 1000,
        sr: Optional[int] = None,
        distance_metric: str = "cosine",
        alpha: float = 1.0,
    ) -> Dict[str, Any]:
        try:
            # Initialize audio factorization with proper parameters
            if isinstance(self.audio_factorization, TimeOnlyFactorization):
                # Reinitialize with actual audio file
                audio_factorizer = TimeOnlyFactorization(
                    input=audio_path,
                    target_sr=sr or 22050,
                    temporal_segmentation_params={"n_temporal_segments": 10},
                )
            else:
                # For source separation, use the existing initialization
                audio_factorizer = self.audio_factorization

            # Initialize lyrics factorization
            lyrics_factorizer = LyricsFactorization(lyrics_lines)

            # Create proper perturbation function
            perturbation_fn = self._create_perturbation_function(
                classifier_fn, audio_factorizer, lyrics_factorizer
            )

            # Get total number of components
            n_audio_components = audio_factorizer.get_number_components()
            n_lyrics_components = lyrics_factorizer.get_number_components()
            total_components = n_audio_components + n_lyrics_components

            # Generate explanation
            feature_importance, lime_metadata = self.lime_base.explain_instance(
                instance_id=audio_path,
                classifier_fn=lambda x: x,  # Dummy, real logic in perturbation_fn
                perturbation_fn=perturbation_fn,
                n_features=total_components,
                n_samples=n_samples,
                distance_metric=distance_metric,
                alpha=alpha,
            )

            # Process results
            explanation = self._process_explanation_results(
                feature_importance, lime_metadata, audio_factorizer, lyrics_factorizer
            )

            return explanation

        except Exception as e:
            raise RuntimeError(f"Explanation failed: {str(e)}")

    def _process_explanation_results(
        self, feature_importance, lime_metadata, audio_factorizer, lyrics_factorizer
    ):
        # Your existing processing logic here, but fixed to use the factorizers
        n_audio = audio_factorizer.get_number_components()
        n_lyrics = lyrics_factorizer.get_number_components()

        # ... rest of your processing logic
        return {
            "audio_explanations": [],
            "lyrics_explanations": [],
            "all_explanations": [],
            "summary": {
                "n_audio_components": n_audio,
                "n_lyrics_components": n_lyrics,
                "total_components": n_audio + n_lyrics,
            },
            "lime_metadata": lime_metadata,
        }
