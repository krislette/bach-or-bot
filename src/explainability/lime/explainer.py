"""Corrected MusicLIME explainer implementation."""

import numpy as np
from sklearn.metrics import pairwise_distances
from typing import List, Dict, Callable, Optional, Any
import warnings

from .base import LimeBase
from ..factorization.base import Factorization
from ..factorization.lyrics import LyricsFactorization


class LimeMusicExplainer:
    """
    Corrected MusicLIME explainer that follows the original paper architecture.

    This implementation creates unified perturbations for both audio and lyrics
    simultaneously, which is the core of MusicLIME.
    """

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
        self.random_state = random_state if random_state else 42
        np.random.seed(self.random_state)

    def explain_instance(
        self,
        audio_path: str,
        lyrics_lines: List[str],
        classifier_fn: Callable,
        n_samples: int = 1000,
        batch_size: int = 32,
        distance_metric: str = "cosine",
        modality: str = "both",
    ) -> Dict[str, Any]:
        """
        Generate MusicLIME explanation using unified perturbations.

        This is the corrected version that follows the original MusicLIME logic.
        """
        try:
            # Store original lyrics for perturbation
            self._original_lyrics = lyrics_lines

            # Create lyrics factorization
            lyrics_factorizer = LyricsFactorization(lyrics_lines)

            # Get component counts
            n_audio_components = self.audio_factorization.get_number_components()
            n_lyrics_components = lyrics_factorizer.get_number_components()

            if modality == "both":
                total_components = n_audio_components + n_lyrics_components
            elif modality == "audio":
                total_components = n_audio_components
            elif modality == "lyrics":
                total_components = n_lyrics_components
            else:
                raise ValueError("modality must be 'both', 'audio', or 'lyrics'")

            # Generate unified perturbation matrix
            data = np.random.randint(0, 2, (n_samples, total_components))
            data[0, :] = 1  # First sample is original (all features active)

            # Get predictions for all perturbations
            labels = self._get_perturbation_predictions(
                data,
                n_audio_components,
                n_lyrics_components,
                classifier_fn,
                batch_size,
                modality,
            )

            # Compute distances
            distances = pairwise_distances(
                data, data[0].reshape(1, -1), metric=distance_metric
            ).ravel()

            # Generate LIME explanation
            feature_importance, lime_metadata = self.lime_base.explain_instance(
                instance_id=audio_path,
                classifier_fn=lambda x: x,  # Dummy, we already have predictions
                perturbation_fn=lambda x: x,  # Dummy, we already have perturbations
                n_features=total_components,
                n_samples=n_samples,
                distance_metric=distance_metric,
            )

            # Override with our data
            from sklearn.linear_model import Ridge

            weights = self.lime_base.kernel_fn(distances)
            ridge = Ridge(alpha=1.0, fit_intercept=True)
            ridge.fit(data, labels, sample_weight=weights)
            feature_importance = ridge.coef_

            # Process results
            explanation = self._process_explanation_results(
                feature_importance,
                lime_metadata,
                n_audio_components,
                n_lyrics_components,
                lyrics_lines,
                modality,
            )

            return explanation

        except Exception as e:
            raise RuntimeError(f"Explanation failed: {str(e)}")

    def _get_perturbation_predictions(
        self,
        data,
        n_audio_components,
        n_lyrics_components,
        classifier_fn,
        batch_size,
        modality,
    ):
        """Get predictions for all perturbations using batch processing."""

        labels = []
        batch_lyrics = []
        batch_audios = []

        for row in data:
            if modality == "both":
                # Split perturbation between audio and lyrics
                audio_mask = row[:n_audio_components]
                lyrics_mask = row[n_audio_components:]

                # Get active audio components
                active_audio = np.where(audio_mask != 0)[0].tolist()
                perturbed_audio = self.audio_factorization.compose_model_input(
                    active_audio
                )

                # Get active lyrics lines
                inactive_lyrics = np.where(lyrics_mask == 0)[0].tolist()
                lyrics_factorizer = LyricsFactorization(
                    [
                        line
                        for i, line in enumerate(self._original_lyrics)
                        if i not in inactive_lyrics
                    ]
                )
                perturbed_lyrics = lyrics_factorizer.lyrics_lines

            elif modality == "audio":
                active_audio = np.where(row != 0)[0].tolist()
                perturbed_audio = self.audio_factorization.compose_model_input(
                    active_audio
                )
                perturbed_lyrics = self._original_lyrics

            elif modality == "lyrics":
                # Use all audio
                all_audio = list(range(n_audio_components))
                perturbed_audio = self.audio_factorization.compose_model_input(
                    all_audio
                )

                # Perturb lyrics
                inactive_lyrics = np.where(row == 0)[0].tolist()
                perturbed_lyrics = [
                    line
                    for i, line in enumerate(self._original_lyrics)
                    if i not in inactive_lyrics
                ]

            batch_audios.append(perturbed_audio)
            batch_lyrics.append(perturbed_lyrics)

            # Process batch when full
            if len(batch_audios) == batch_size:
                preds = classifier_fn(batch_lyrics, np.array(batch_audios))
                labels.extend(preds)
                batch_audios = []
                batch_lyrics = []

        # Process remaining batch
        if len(batch_audios) > 0:
            preds = classifier_fn(batch_lyrics, np.array(batch_audios))
            labels.extend(preds)

        return np.array(labels)

    def _process_explanation_results(
        self,
        feature_importance,
        lime_metadata,
        n_audio_components,
        n_lyrics_components,
        lyrics_lines,
        modality,
    ):
        """Process LIME results into explanation format."""

        # Split importance scores
        if modality == "both":
            audio_importances = feature_importance[:n_audio_components]
            lyrics_importances = feature_importance[n_audio_components:]
        elif modality == "audio":
            audio_importances = feature_importance
            lyrics_importances = []
        elif modality == "lyrics":
            audio_importances = []
            lyrics_importances = feature_importance

        # Create explanations
        audio_explanations = []
        for i, importance in enumerate(audio_importances):
            audio_explanations.append(
                {
                    "component_id": i,
                    "importance": float(importance),
                    "name": f"audio_segment_{i}",
                    "type": "audio",
                }
            )

        lyrics_explanations = []
        for i, importance in enumerate(lyrics_importances):
            lyrics_explanations.append(
                {
                    "component_id": n_audio_components + i,
                    "importance": float(importance),
                    "name": f"lyrics: {lyrics_lines[i] if i < len(lyrics_lines) else f'line_{i}'}",
                    "type": "lyrics",
                }
            )

        # Combine and sort
        all_explanations = audio_explanations + lyrics_explanations
        all_explanations.sort(key=lambda x: abs(x["importance"]), reverse=True)

        return {
            "audio_explanations": audio_explanations,
            "lyrics_explanations": lyrics_explanations,
            "all_explanations": all_explanations,
            "summary": {
                "n_audio_components": len(audio_importances),
                "n_lyrics_components": len(lyrics_importances),
                "total_components": len(audio_importances) + len(lyrics_importances),
            },
            "lime_metadata": lime_metadata,
        }
