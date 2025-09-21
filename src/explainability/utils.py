"""
Utility functions for MusicLIME explainability module.

This module contains helper functions for data processing, validation,
and common operations used across the explainability components.
"""

import numpy as np
import librosa
from typing import List, Dict, Tuple, Optional, Union, Any, Callable
from pathlib import Path
import warnings


def validate_audio_file(audio_path: Union[str, Path]) -> bool:
    """
    Validate that an audio file exists and can be loaded.

    Parameters
    ----------
    audio_path : Union[str, Path]
        Path to the audio file

    Returns
    -------
    bool
        True if file is valid, False otherwise
    """
    try:
        audio_path = Path(audio_path)
        if not audio_path.exists():
            return False

        # Try to load a small portion to validate format
        audio, sr = librosa.load(audio_path, duration=1.0, sr=None)
        return len(audio) > 0
    except Exception:
        return False


def validate_lyrics_lines(lyrics_lines: List[str]) -> Tuple[bool, str]:
    """
    Validate lyrics lines for MusicLIME processing.

    Parameters
    ----------
    lyrics_lines : List[str]
        List of lyric lines to validate

    Returns
    -------
    Tuple[bool, str]
        (is_valid, error_message) - error_message is empty if valid
    """
    if not isinstance(lyrics_lines, list):
        return False, "lyrics_lines must be a list"

    if len(lyrics_lines) == 0:
        return False, "lyrics_lines cannot be empty"

    if not all(isinstance(line, str) for line in lyrics_lines):
        return False, "All lyrics lines must be strings"

    # Check for reasonable content
    non_empty_lines = [line.strip() for line in lyrics_lines if line.strip()]
    if len(non_empty_lines) == 0:
        return False, "All lyrics lines are empty"

    return True, ""


def normalize_importance_scores(
    scores: np.ndarray, method: str = "minmax"
) -> np.ndarray:
    """
    Normalize importance scores to a standard range.

    Parameters
    ----------
    scores : np.ndarray
        Raw importance scores
    method : str, default='minmax'
        Normalization method ('minmax', 'zscore', 'absolute')

    Returns
    -------
    np.ndarray
        Normalized importance scores

    Raises
    ------
    ValueError
        If method is not supported
    """
    if len(scores) == 0:
        return scores

    scores = scores.copy()

    if method == "minmax":
        min_score, max_score = scores.min(), scores.max()
        if max_score != min_score:
            scores = (scores - min_score) / (max_score - min_score)
    elif method == "zscore":
        mean_score, std_score = scores.mean(), scores.std()
        if std_score != 0:
            scores = (scores - mean_score) / std_score
    elif method == "absolute":
        max_abs = np.abs(scores).max()
        if max_abs != 0:
            scores = scores / max_abs
    else:
        raise ValueError(f"Unsupported normalization method: {method}")

    return scores


def compute_explanation_stability(
    explanations: List[Dict[str, Any]],
    component_key: str = "component_id",
    importance_key: str = "importance",
) -> Dict[str, float]:
    """
    Compute stability metrics across multiple explanations of the same instance.

    Parameters
    ----------
    explanations : List[Dict[str, Any]]
        List of explanation dictionaries
    component_key : str, default='component_id'
        Key to identify components across explanations
    importance_key : str, default='importance'
        Key for importance scores

    Returns
    -------
    Dict[str, float]
        Stability metrics including correlation and ranking stability
    """
    if len(explanations) < 2:
        return {"correlation": 1.0, "ranking_stability": 1.0, "n_comparisons": 0}

    # Extract importance vectors for each explanation
    importance_vectors = []
    all_component_ids = set()

    for exp_data in explanations:
        components = exp_data.get("all_explanations", [])
        comp_dict = {comp[component_key]: comp[importance_key] for comp in components}
        all_component_ids.update(comp_dict.keys())
        importance_vectors.append(comp_dict)

    # Create aligned importance matrices
    sorted_ids = sorted(all_component_ids)
    importance_matrix = []

    for comp_dict in importance_vectors:
        vector = [comp_dict.get(comp_id, 0.0) for comp_id in sorted_ids]
        importance_matrix.append(vector)

    importance_matrix = np.array(importance_matrix)

    # Compute pairwise correlations
    correlations = []
    ranking_stabilities = []
    n_comparisons = 0

    for i in range(len(importance_matrix)):
        for j in range(i + 1, len(importance_matrix)):
            # Correlation stability
            corr = np.corrcoef(importance_matrix[i], importance_matrix[j])[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)

            # Ranking stability (Spearman correlation)
            from scipy.stats import spearmanr

            rank_corr, _ = spearmanr(importance_matrix[i], importance_matrix[j])
            if not np.isnan(rank_corr):
                ranking_stabilities.append(rank_corr)

            n_comparisons += 1

    return {
        "correlation": np.mean(correlations) if correlations else 0.0,
        "ranking_stability": (
            np.mean(ranking_stabilities) if ranking_stabilities else 0.0
        ),
        "n_comparisons": n_comparisons,
    }


def create_classifier_wrapper(
    model: Any,
    feature_extractor: Callable,
    audio_processor: Optional[Callable] = None,
    lyrics_processor: Optional[Callable] = None,
) -> Callable:
    """
    Create a classifier wrapper function for use with MusicLIME.

    Parameters
    ----------
    model : Any
        The trained classifier model
    feature_extractor : Callable
        Function that extracts features from (audio, lyrics) -> features
    audio_processor : Optional[Callable], default=None
        Optional audio preprocessing function
    lyrics_processor : Optional[Callable], default=None
        Optional lyrics preprocessing function

    Returns
    -------
    Callable
        Classifier function compatible with MusicLIME
    """

    def classifier_fn(
        audio: np.ndarray, lyrics_lines: List[str]
    ) -> Union[float, np.ndarray]:
        """
        Classifier wrapper function.

        Parameters
        ----------
        audio : np.ndarray
            Audio signal
        lyrics_lines : List[str]
            List of lyric lines

        Returns
        -------
        Union[float, np.ndarray]
            Model prediction (probability or logits)
        """
        try:
            # Apply preprocessing if provided
            if audio_processor is not None:
                audio = audio_processor(audio)

            if lyrics_processor is not None:
                lyrics_lines = lyrics_processor(lyrics_lines)

            # Extract features
            features = feature_extractor(audio, lyrics_lines)

            # Get prediction
            if hasattr(model, "predict_proba"):
                # Classifier with probability output
                probabilities = model.predict_proba(features.reshape(1, -1))
                return probabilities[0]  # Return probabilities for first (only) sample
            elif hasattr(model, "predict"):
                # Regressor or classifier with single output
                prediction = model.predict(features.reshape(1, -1))
                return float(prediction[0])
            else:
                # Direct callable model
                return model(features.reshape(1, -1))

        except Exception as e:
            warnings.warn(f"Classifier wrapper error: {str(e)}")
            # Return neutral prediction
            return 0.5

    return classifier_fn


def merge_explanations(
    explanations: List[Dict[str, Any]], method: str = "mean"
) -> Dict[str, Any]:
    """
    Merge multiple explanations of the same instance.

    Parameters
    ----------
    explanations : List[Dict[str, Any]]
        List of explanation dictionaries to merge
    method : str, default='mean'
        Merging method ('mean', 'median', 'voting')

    Returns
    -------
    Dict[str, Any]
        Merged explanation

    Raises
    ------
    ValueError
        If method is not supported or explanations are incompatible
    """
    if not explanations:
        raise ValueError("Cannot merge empty list of explanations")

    if len(explanations) == 1:
        return explanations[0].copy()

    # Validate compatibility
    base_exp = explanations[0]
    n_audio = base_exp["summary"]["n_audio_components"]
    n_lyrics = base_exp["summary"]["n_lyrics_components"]

    for exp in explanations[1:]:
        if (
            exp["summary"]["n_audio_components"] != n_audio
            or exp["summary"]["n_lyrics_components"] != n_lyrics
        ):
            raise ValueError("Explanations have incompatible component counts")

    # Merge importance scores
    if method == "mean":
        merge_fn = np.mean
    elif method == "median":
        merge_fn = np.median
    else:
        raise ValueError(f"Unsupported merging method: {method}")

    # Collect importance scores by component ID
    component_scores = {}
    component_metadata = {}

    for exp in explanations:
        for comp in exp["all_explanations"]:
            comp_id = comp["component_id"]
            if comp_id not in component_scores:
                component_scores[comp_id] = []
                component_metadata[comp_id] = comp.copy()
            component_scores[comp_id].append(comp["importance"])

    # Merge scores
    merged_components = []
    for comp_id, scores in component_scores.items():
        merged_score = merge_fn(scores)
        comp_data = component_metadata[comp_id].copy()
        comp_data["importance"] = float(merged_score)
        merged_components.append(comp_data)

    # Sort by absolute importance
    merged_components.sort(key=lambda x: abs(x["importance"]), reverse=True)

    # Separate by modality
    audio_components = [c for c in merged_components if c["type"] == "audio"]
    lyrics_components = [c for c in merged_components if c["type"] == "lyrics"]

    # Create merged explanation
    merged_explanation = {
        "audio_explanations": audio_components,
        "lyrics_explanations": lyrics_components,
        "all_explanations": merged_components,
        "summary": base_exp["summary"].copy(),
        "lime_metadata": {"merged_from": len(explanations), "merge_method": method},
        "factorization_info": base_exp["factorization_info"].copy(),
    }

    # Update summary with merged info
    merged_explanation["summary"]["most_important_overall"] = (
        merged_components[0] if merged_components else None
    )
    merged_explanation["summary"]["most_important_audio"] = (
        audio_components[0] if audio_components else None
    )
    merged_explanation["summary"]["most_important_lyrics"] = (
        lyrics_components[0] if lyrics_components else None
    )

    return merged_explanation


def get_component_time_ranges(
    audio_explanations: List[Dict[str, Any]],
) -> List[Tuple[float, float]]:
    """
    Extract time ranges from audio component explanations.

    Parameters
    ----------
    audio_explanations : List[Dict[str, Any]]
        List of audio component explanations

    Returns
    -------
    List[Tuple[float, float]]
        List of (start_time, end_time) tuples for each component
    """
    time_ranges = []

    for exp in audio_explanations:
        metadata = exp.get("metadata", {})

        if "start_time" in metadata and "end_time" in metadata:
            # Temporal factorization
            time_ranges.append((metadata["start_time"], metadata["end_time"]))
        else:
            # Source separation or other - no specific time range
            time_ranges.append((0.0, float("inf")))

    return time_ranges


def validate_explanation_consistency(
    explanation: Dict[str, Any],
) -> Tuple[bool, List[str]]:
    """
    Validate internal consistency of an explanation.

    Parameters
    ----------
    explanation : Dict[str, Any]
        Explanation dictionary to validate

    Returns
    -------
    Tuple[bool, List[str]]
        (is_valid, list_of_errors)
    """
    errors = []

    # Check required keys
    required_keys = [
        "audio_explanations",
        "lyrics_explanations",
        "all_explanations",
        "summary",
    ]
    for key in required_keys:
        if key not in explanation:
            errors.append(f"Missing required key: {key}")

    if errors:  # Don't continue if basic structure is missing
        return False, errors

    # Check component counts
    n_audio = len(explanation["audio_explanations"])
    n_lyrics = len(explanation["lyrics_explanations"])
    n_total = len(explanation["all_explanations"])

    expected_total = n_audio + n_lyrics
    if n_total != expected_total:
        errors.append(
            f"Component count mismatch: {n_total} total != {n_audio} audio + {n_lyrics} lyrics"
        )

    # Check summary consistency
    summary = explanation["summary"]
    if summary.get("n_audio_components") != n_audio:
        errors.append(
            f"Summary audio count mismatch: {summary.get('n_audio_components')} != {n_audio}"
        )

    if summary.get("n_lyrics_components") != n_lyrics:
        errors.append(
            f"Summary lyrics count mismatch: {summary.get('n_lyrics_components')} != {n_lyrics}"
        )

    # Check component IDs are unique
    all_ids = [comp["component_id"] for comp in explanation["all_explanations"]]
    if len(set(all_ids)) != len(all_ids):
        errors.append("Duplicate component IDs found")

    # Check importance scores are numeric
    for comp in explanation["all_explanations"]:
        try:
            float(comp["importance"])
        except (ValueError, TypeError):
            errors.append(
                f"Non-numeric importance score for component {comp['component_id']}"
            )

    return len(errors) == 0, errors
