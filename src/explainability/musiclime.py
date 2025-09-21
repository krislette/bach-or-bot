"""
Multimodal explanation class for MusicLIME results.

This module contains the MultimodalExplanation class that stores, organizes,
and provides utilities for working with explanations from multimodal music models.
"""

import numpy as np
from typing import List, Dict, Optional, Union, Any
import json
from pathlib import Path


class MultimodalExplanation:
    """
    Container class for multimodal music explanations.

    This class stores and organizes explanation results from MusicLIME,
    providing methods for accessing, filtering, and exporting explanations
    across both audio and textual modalities.
    """

    def __init__(
        self,
        explanation_data: Dict[str, Any],
        instance_info: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize multimodal explanation container.

        Parameters
        ----------
        explanation_data : Dict[str, Any]
            Raw explanation data from LimeMusicExplainer
        instance_info : Optional[Dict[str, Any]], default=None
            Additional information about the explained instance
        """
        self.explanation_data = explanation_data
        self.instance_info = instance_info or {}

        # Extract key components for easy access
        self.audio_explanations = explanation_data.get("audio_explanations", [])
        self.lyrics_explanations = explanation_data.get("lyrics_explanations", [])
        self.all_explanations = explanation_data.get("all_explanations", [])
        self.summary = explanation_data.get("summary", {})
        self.lime_metadata = explanation_data.get("lime_metadata", {})
        self.factorization_info = explanation_data.get("factorization_info", {})

        # Derived properties
        self.n_audio_components = self.summary.get("n_audio_components", 0)
        self.n_lyrics_components = self.summary.get("n_lyrics_components", 0)
        self.total_components = self.summary.get("total_components", 0)

    def get_audio_importance_scores(self, normalize: bool = False) -> np.ndarray:
        """
        Get importance scores for audio components.

        Parameters
        ----------
        normalize : bool, default=False
            Whether to normalize scores to [0, 1] range

        Returns
        -------
        np.ndarray
            Array of audio component importance scores
        """
        scores = np.array([exp["importance"] for exp in self.audio_explanations])

        if normalize and len(scores) > 0:
            min_score, max_score = scores.min(), scores.max()
            if max_score != min_score:
                scores = (scores - min_score) / (max_score - min_score)

        return scores

    def get_lyrics_importance_scores(self, normalize: bool = False) -> np.ndarray:
        """
        Get importance scores for lyrics components.

        Parameters
        ----------
        normalize : bool, default=False
            Whether to normalize scores to [0, 1] range

        Returns
        -------
        np.ndarray
            Array of lyrics component importance scores
        """
        scores = np.array([exp["importance"] for exp in self.lyrics_explanations])

        if normalize and len(scores) > 0:
            min_score, max_score = scores.min(), scores.max()
            if max_score != min_score:
                scores = (scores - min_score) / (max_score - min_score)

        return scores

    def get_top_audio_components(
        self, n: int = 5, by_absolute: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get top N most important audio components.

        Parameters
        ----------
        n : int, default=5
            Number of components to return
        by_absolute : bool, default=True
            Whether to rank by absolute importance values

        Returns
        -------
        List[Dict[str, Any]]
            List of top audio component explanations
        """
        explanations = self.audio_explanations.copy()

        if by_absolute:
            explanations.sort(key=lambda x: abs(x["importance"]), reverse=True)
        else:
            explanations.sort(key=lambda x: x["importance"], reverse=True)

        return explanations[:n]

    def get_top_lyrics_components(
        self, n: int = 5, by_absolute: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get top N most important lyrics components.

        Parameters
        ----------
        n : int, default=5
            Number of components to return
        by_absolute : bool, default=True
            Whether to rank by absolute importance values

        Returns
        -------
        List[Dict[str, Any]]
            List of top lyrics component explanations
        """
        explanations = self.lyrics_explanations.copy()

        if by_absolute:
            explanations.sort(key=lambda x: abs(x["importance"]), reverse=True)
        else:
            explanations.sort(key=lambda x: x["importance"], reverse=True)

        return explanations[:n]

    def get_positive_components(
        self, modality: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get components with positive importance (supporting the prediction).

        Parameters
        ----------
        modality : Optional[str], default=None
            Filter by modality ('audio', 'lyrics', or None for both)

        Returns
        -------
        List[Dict[str, Any]]
            List of components with positive importance scores
        """
        if modality == "audio":
            explanations = self.audio_explanations
        elif modality == "lyrics":
            explanations = self.lyrics_explanations
        else:
            explanations = self.all_explanations

        positive = [exp for exp in explanations if exp["importance"] > 0]
        positive.sort(key=lambda x: x["importance"], reverse=True)

        return positive

    def get_negative_components(
        self, modality: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get components with negative importance (opposing the prediction).

        Parameters
        ----------
        modality : Optional[str], default=None
            Filter by modality ('audio', 'lyrics', or None for both)

        Returns
        -------
        List[Dict[str, Any]]
            List of components with negative importance scores
        """
        if modality == "audio":
            explanations = self.audio_explanations
        elif modality == "lyrics":
            explanations = self.lyrics_explanations
        else:
            explanations = self.all_explanations

        negative = [exp for exp in explanations if exp["importance"] < 0]
        negative.sort(key=lambda x: x["importance"])  # Most negative first

        return negative

    def get_modality_contribution(self) -> Dict[str, float]:
        """
        Get overall contribution of each modality to the prediction.

        Returns
        -------
        Dict[str, float]
            Dictionary with total positive importance per modality
        """
        audio_positive = sum(
            exp["importance"]
            for exp in self.audio_explanations
            if exp["importance"] > 0
        )
        lyrics_positive = sum(
            exp["importance"]
            for exp in self.lyrics_explanations
            if exp["importance"] > 0
        )

        total_positive = audio_positive + lyrics_positive

        if total_positive > 0:
            return {
                "audio": audio_positive / total_positive,
                "lyrics": lyrics_positive / total_positive,
            }
        else:
            return {"audio": 0.0, "lyrics": 0.0}

    def get_component_by_id(self, component_id: int) -> Optional[Dict[str, Any]]:
        """
        Get component explanation by its ID.

        Parameters
        ----------
        component_id : int
            ID of the component to retrieve

        Returns
        -------
        Optional[Dict[str, Any]]
            Component explanation or None if not found
        """
        for exp in self.all_explanations:
            if exp["component_id"] == component_id:
                return exp
        return None

    def filter_components(
        self,
        importance_threshold: float = 0.0,
        modality: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Filter components based on various criteria.

        Parameters
        ----------
        importance_threshold : float, default=0.0
            Minimum absolute importance threshold
        modality : Optional[str], default=None
            Filter by modality ('audio', 'lyrics', or None for both)
        top_k : Optional[int], default=None
            Return only top K components by absolute importance

        Returns
        -------
        List[Dict[str, Any]]
            Filtered list of component explanations
        """
        # Select base explanations
        if modality == "audio":
            explanations = self.audio_explanations
        elif modality == "lyrics":
            explanations = self.lyrics_explanations
        else:
            explanations = self.all_explanations

        # Filter by importance threshold
        filtered = [
            exp
            for exp in explanations
            if abs(exp["importance"]) >= importance_threshold
        ]

        # Sort by absolute importance
        filtered.sort(key=lambda x: abs(x["importance"]), reverse=True)

        # Apply top-k limit
        if top_k is not None:
            filtered = filtered[:top_k]

        return filtered

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert explanation to dictionary format.

        Returns
        -------
        Dict[str, Any]
            Complete explanation data as dictionary
        """
        return {
            "explanation_data": self.explanation_data,
            "instance_info": self.instance_info,
            "derived_stats": {
                "modality_contributions": self.get_modality_contribution(),
                "n_positive_components": len(self.get_positive_components()),
                "n_negative_components": len(self.get_negative_components()),
                "audio_score_stats": self._get_score_stats(
                    self.get_audio_importance_scores()
                ),
                "lyrics_score_stats": self._get_score_stats(
                    self.get_lyrics_importance_scores()
                ),
            },
        }

    def _get_score_stats(self, scores: np.ndarray) -> Dict[str, float]:
        """
        Get statistical summary of importance scores.

        Parameters
        ----------
        scores : np.ndarray
            Array of importance scores

        Returns
        -------
        Dict[str, float]
            Statistical summary
        """
        if len(scores) == 0:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

        return {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
        }

    def save_to_json(self, filepath: Union[str, Path]) -> None:
        """
        Save explanation to JSON file.

        Parameters
        ----------
        filepath : Union[str, Path]
            Path where to save the JSON file

        Raises
        ------
        IOError
            If file cannot be written
        """
        filepath = Path(filepath)

        try:
            data = self.to_dict()
            # Convert numpy arrays to lists for JSON serialization
            data = self._prepare_for_json(data)

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            raise IOError(f"Failed to save explanation to {filepath}: {str(e)}")

    def _prepare_for_json(self, obj: Any) -> Any:
        """
        Prepare object for JSON serialization by converting numpy types.

        Parameters
        ----------
        obj : Any
            Object to prepare

        Returns
        -------
        Any
            JSON-serializable object
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.number):
            return obj.item()
        elif isinstance(obj, dict):
            return {key: self._prepare_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._prepare_for_json(item) for item in obj]
        else:
            return obj

    @classmethod
    def load_from_json(cls, filepath: Union[str, Path]) -> "MultimodalExplanation":
        """
        Load explanation from JSON file.

        Parameters
        ----------
        filepath : Union[str, Path]
            Path to the JSON file

        Returns
        -------
        MultimodalExplanation
            Loaded explanation object

        Raises
        ------
        IOError
            If file cannot be read or parsed
        """
        filepath = Path(filepath)

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            return cls(
                explanation_data=data["explanation_data"],
                instance_info=data.get("instance_info", {}),
            )

        except Exception as e:
            raise IOError(f"Failed to load explanation from {filepath}: {str(e)}")

    def get_summary_text(self) -> str:
        """
        Get human-readable summary of the explanation.

        Returns
        -------
        str
            Text summary of the explanation
        """
        lines = []
        lines.append("=== MusicLIME Explanation Summary ===")
        lines.append(f"Total components: {self.total_components}")
        lines.append(f"Audio components: {self.n_audio_components}")
        lines.append(f"Lyrics components: {self.n_lyrics_components}")
        lines.append("")

        # Modality contributions
        contributions = self.get_modality_contribution()
        lines.append("Modality contributions:")
        lines.append(f"  Audio: {contributions['audio']:.1%}")
        lines.append(f"  Lyrics: {contributions['lyrics']:.1%}")
        lines.append("")

        # Top components overall
        lines.append("Top 3 most important components:")
        top_components = self.all_explanations[:3]
        for i, comp in enumerate(top_components, 1):
            modality = "🎵" if comp["type"] == "audio" else "📝"
            lines.append(f"  {i}. {modality} {comp['name']}: {comp['importance']:.3f}")

        # LIME quality metrics
        if self.lime_metadata.get("local_r2_score") is not None:
            lines.append("")
            lines.append(
                f"Local model R² score: {self.lime_metadata['local_r2_score']:.3f}"
            )

        return "\n".join(lines)

    def __str__(self) -> str:
        """String representation of the explanation."""
        return f"MultimodalExplanation({self.n_audio_components} audio, {self.n_lyrics_components} lyrics)"

    def __repr__(self) -> str:
        """Detailed representation of the explanation."""
        return self.__str__()

    def __len__(self) -> int:
        """Return total number of components."""
        return self.total_components
