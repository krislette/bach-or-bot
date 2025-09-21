"""
MusicLIME explainer implementation.

This module contains the LimeMusicExplainer class that provides explanations
for multimodal music classification models using both audio and lyrics.
"""

import numpy as np
from typing import List, Dict, Callable, Optional, Any
import warnings

from .base import LimeBase
from ..factorization.base import Factorization


class LimeMusicExplainer:
    """
    LIME explainer specifically designed for multimodal music classification.

    This explainer can generate explanations for models that use both audio features
    and lyrical content by perturbing audio components and text lines independently
    or jointly.
    """

    def __init__(
        self,
        audio_factorization: Factorization,
        kernel_width: float = 0.25,
        verbose: bool = False,
        random_state: Optional[int] = None,
    ):
        """
        Initialize the MusicLIME explainer.

        Parameters
        ----------
        audio_factorization : Factorization
            Factorization strategy for audio (temporal or source separation)
        kernel_width : float, default=0.25
            Width parameter for the kernel function
        verbose : bool, default=False
            Whether to print progress information
        random_state : Optional[int], default=None
            Random seed for reproducibility
        """
        self.audio_factorization = audio_factorization
        self.lime_base = LimeBase(
            kernel_width=kernel_width, verbose=verbose, random_state=random_state
        )
        self.verbose = verbose
        self.random_state = random_state

        # Will be set during factorization
        self._audio_segments = None
        self._original_audio = None
        self._lyrics_lines = None
        self._factorized_data = None

    def factorize_inputs(
        self, audio_path: str, lyrics_lines: List[str], sr: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Factorize audio and lyrics into interpretable components.

        Parameters
        ----------
        audio_path : str
            Path to the audio file
        lyrics_lines : List[str]
            List of lyric lines (preprocessed)
        sr : Optional[int], default=None
            Target sample rate for audio

        Returns
        -------
        Dict[str, Any]
            Dictionary containing factorized components and metadata

        Raises
        ------
        ValueError
            If inputs are invalid or factorization fails
        """
        if not lyrics_lines:
            raise ValueError("lyrics_lines cannot be empty")

        try:
            # Factorize audio
            if self.verbose:
                print("Factorizing audio...")

            self._original_audio, self._audio_segments = (
                self.audio_factorization.factorize(audio_path, sr=sr)
            )

            # Store lyrics lines
            self._lyrics_lines = lyrics_lines

            # Create combined component information
            n_audio_components = len(self._audio_segments)
            n_lyrics_components = len(self._lyrics_lines)
            total_components = n_audio_components + n_lyrics_components

            # Build component metadata
            components = []

            # Audio components
            for i, segment in enumerate(self._audio_segments):
                component = {
                    "id": i,
                    "type": "audio",
                    "modality": "audio",
                    "name": self._get_audio_component_name(segment, i),
                    "metadata": segment,
                }
                components.append(component)

            # Lyrics components
            for i, line in enumerate(self._lyrics_lines):
                component = {
                    "id": n_audio_components + i,
                    "type": "lyrics",
                    "modality": "text",
                    "name": f"lyrics_line_{i}",
                    "content": line,
                    "metadata": {"line_id": i, "line_text": line},
                }
                components.append(component)

            self._factorized_data = {
                "audio_path": audio_path,
                "components": components,
                "n_audio_components": n_audio_components,
                "n_lyrics_components": n_lyrics_components,
                "total_components": total_components,
                "audio_factorization_type": type(self.audio_factorization).__name__,
            }

            if self.verbose:
                print(
                    f"Factorization complete: {n_audio_components} audio + {n_lyrics_components} lyrics = {total_components} total components"
                )

            return self._factorized_data

        except Exception as e:
            raise ValueError(f"Failed to factorize inputs: {str(e)}")

    def _get_audio_component_name(self, segment: Dict[str, Any], index: int) -> str:
        """
        Generate a descriptive name for an audio component.

        Parameters
        ----------
        segment : Dict[str, Any]
            Audio segment metadata
        index : int
            Component index

        Returns
        -------
        str
            Descriptive component name
        """
        if segment.get("type") == "temporal":
            start_time = segment.get("start_time", 0)
            end_time = segment.get("end_time", start_time + 1)
            return f"audio_{start_time:.1f}s-{end_time:.1f}s"
        elif segment.get("type") == "source_separation":
            source_name = segment.get("source_name", f"source_{index}")
            return f"audio_{source_name}"
        else:
            return f"audio_component_{index}"

    def _create_perturbation_function(self, classifier_fn: Callable) -> Callable:
        """
        Create perturbation function for LIME that handles both audio and lyrics.

        Parameters
        ----------
        classifier_fn : Callable
            Classifier function that takes (audio, lyrics_lines) and returns predictions

        Returns
        -------
        Callable
            Perturbation function for LIME
        """

        def perturbation_fn(binary_mask: np.ndarray) -> Any:
            """
            Generate perturbed input based on binary mask.

            Parameters
            ----------
            binary_mask : np.ndarray
                Binary mask indicating which components are active

            Returns
            -------
            Any
                Classifier prediction for perturbed input
            """
            if self._factorized_data is None:
                raise RuntimeError(
                    "No factorized data available. Call factorize_inputs() first."
                )

            n_audio = self._factorized_data["n_audio_components"]
            n_lyrics = self._factorized_data["n_lyrics_components"]

            # Split mask into audio and lyrics components
            audio_mask = binary_mask[:n_audio]
            lyrics_mask = (
                binary_mask[n_audio : n_audio + n_lyrics]
                if n_lyrics > 0
                else np.array([])
            )

            # Perturb audio
            active_audio_segments = np.where(audio_mask)[0].tolist()
            if len(active_audio_segments) > 0:
                perturbed_audio = self.audio_factorization.perturb(
                    self._original_audio, self._audio_segments, active_audio_segments
                )
            else:
                # No audio components active - use silence
                perturbed_audio = np.zeros_like(self._original_audio)

            # Perturb lyrics
            if n_lyrics > 0:
                active_lyrics_indices = np.where(lyrics_mask)[0].tolist()
                perturbed_lyrics = [
                    self._lyrics_lines[i] if i in active_lyrics_indices else ""
                    for i in range(n_lyrics)
                ]
                # Remove empty lines
                perturbed_lyrics = [line for line in perturbed_lyrics if line.strip()]
            else:
                perturbed_lyrics = []

            # Get classifier prediction
            try:
                prediction = classifier_fn(perturbed_audio, perturbed_lyrics)
                return prediction
            except Exception as e:
                if self.verbose:
                    warnings.warn(f"Classifier error for perturbation: {str(e)}")
                # Return neutral prediction in case of error
                return 0.5  # Assuming binary classification

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
        """
        Generate explanation for a music instance with both audio and lyrics.

        Parameters
        ----------
        audio_path : str
            Path to the audio file to explain
        lyrics_lines : List[str]
            List of lyric lines (preprocessed)
        classifier_fn : Callable
            Function that takes (audio, lyrics_lines) and returns prediction
        n_samples : int, default=1000
            Number of perturbed samples to generate for LIME
        sr : Optional[int], default=None
            Target sample rate for audio
        distance_metric : str, default='cosine'
            Distance metric for LIME neighborhood weighting
        alpha : float, default=1.0
            Regularization strength for local linear model

        Returns
        -------
        Dict[str, Any]
            Comprehensive explanation containing feature importance and metadata

        Raises
        ------
        ValueError
            If inputs are invalid
        RuntimeError
            If explanation generation fails
        """
        try:
            # Factorize inputs
            factorized_data = self.factorize_inputs(audio_path, lyrics_lines, sr)

            if self.verbose:
                print(f"Starting explanation with {n_samples} samples...")

            # Create perturbation function
            perturbation_fn = self._create_perturbation_function(classifier_fn)

            # Generate explanation using LIME
            feature_importance, lime_metadata = self.lime_base.explain_instance(
                instance_id=audio_path,
                classifier_fn=lambda data: perturbation_fn(
                    np.ones(factorized_data["total_components"])
                ),  # Original instance
                perturbation_fn=perturbation_fn,
                n_features=factorized_data["total_components"],
                n_samples=n_samples,
                distance_metric=distance_metric,
                alpha=alpha,
            )

            # Process and organize results
            explanation = self._process_explanation_results(
                feature_importance, lime_metadata, factorized_data
            )

            if self.verbose:
                print("Explanation completed successfully")

            return explanation

        except Exception as e:
            raise RuntimeError(f"Failed to generate explanation: {str(e)}")

    def _process_explanation_results(
        self,
        feature_importance: np.ndarray,
        lime_metadata: Dict[str, Any],
        factorized_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Process and organize explanation results into interpretable format.

        Parameters
        ----------
        feature_importance : np.ndarray
            Raw feature importance scores from LIME
        lime_metadata : Dict[str, Any]
            Metadata from LIME explanation
        factorized_data : Dict[str, Any]
            Factorized component information

        Returns
        -------
        Dict[str, Any]
            Organized explanation results
        """
        components = factorized_data["components"]
        n_audio = factorized_data["n_audio_components"]
        n_lyrics = factorized_data["n_lyrics_components"]

        # Separate audio and lyrics importance
        audio_importance = feature_importance[:n_audio] if n_audio > 0 else np.array([])
        lyrics_importance = (
            feature_importance[n_audio:] if n_lyrics > 0 else np.array([])
        )

        # Create detailed component explanations
        audio_explanations = []
        for i, importance in enumerate(audio_importance):
            component = components[i]
            audio_explanations.append(
                {
                    "component_id": component["id"],
                    "name": component["name"],
                    "importance": float(importance),
                    "type": component["type"],
                    "metadata": component["metadata"],
                }
            )

        lyrics_explanations = []
        for i, importance in enumerate(lyrics_importance):
            component = components[n_audio + i]
            lyrics_explanations.append(
                {
                    "component_id": component["id"],
                    "name": component["name"],
                    "importance": float(importance),
                    "content": component["content"],
                    "type": component["type"],
                    "metadata": component["metadata"],
                }
            )

        # Sort by absolute importance
        audio_explanations.sort(key=lambda x: abs(x["importance"]), reverse=True)
        lyrics_explanations.sort(key=lambda x: abs(x["importance"]), reverse=True)

        # Combine all explanations
        all_explanations = audio_explanations + lyrics_explanations
        all_explanations.sort(key=lambda x: abs(x["importance"]), reverse=True)

        return {
            "audio_explanations": audio_explanations,
            "lyrics_explanations": lyrics_explanations,
            "all_explanations": all_explanations,
            "summary": {
                "n_audio_components": n_audio,
                "n_lyrics_components": n_lyrics,
                "total_components": len(all_explanations),
                "most_important_audio": (
                    audio_explanations[0] if audio_explanations else None
                ),
                "most_important_lyrics": (
                    lyrics_explanations[0] if lyrics_explanations else None
                ),
                "most_important_overall": (
                    all_explanations[0] if all_explanations else None
                ),
                "audio_factorization_type": factorized_data["audio_factorization_type"],
            },
            "lime_metadata": lime_metadata,
            "factorization_info": {
                "audio_segments": len(factorized_data.get("components", [])),
                "lyrics_lines": n_lyrics,
                "factorization_type": factorized_data["audio_factorization_type"],
            },
        }

    def get_top_features(
        self,
        explanation: Dict[str, Any],
        n_features: int = 5,
        modality: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get top N most important features from explanation.

        Parameters
        ----------
        explanation : Dict[str, Any]
            Explanation dictionary from explain_instance()
        n_features : int, default=5
            Number of top features to return
        modality : Optional[str], default=None
            Filter by modality ('audio', 'lyrics', or None for both)

        Returns
        -------
        List[Dict[str, Any]]
            List of top feature explanations
        """
        if modality == "audio":
            features = explanation["audio_explanations"]
        elif modality == "lyrics":
            features = explanation["lyrics_explanations"]
        else:
            features = explanation["all_explanations"]

        return features[:n_features]

    def __str__(self) -> str:
        """String representation of the explainer."""
        factorization_type = type(self.audio_factorization).__name__
        return f"LimeMusicExplainer(factorization={factorization_type})"

    def __repr__(self) -> str:
        """Detailed representation of the explainer."""
        return self.__str__()
