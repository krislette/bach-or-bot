"""
MusicLIME Explainability Module.

This module provides interpretability and explainability tools for multimodal
music classification models using LIME (Local Interpretable Model-agnostic Explanations).

The module supports both audio and textual (lyrics) modalities, allowing for
comprehensive explanations of how different components contribute to model predictions.

Main Components:
- Factorization strategies for decomposing audio into interpretable components
- LIME-based explanation generation for multimodal inputs
- Explanation storage, analysis, and visualization utilities

Example Usage:
    ```python
    from src.explainability import LimeMusicExplainer, TimeOnlyFactorization, MultimodalExplanation

    # Set up factorization and explainer
    factorizer = TimeOnlyFactorization(segment_duration=2.0)
    explainer = LimeMusicExplainer(factorizer)

    # Generate explanation
    explanation_data = explainer.explain_instance(
        audio_path="song.wav",
        lyrics_lines=["First line", "Second line"],
        classifier_fn=my_classifier,
        n_samples=1000
    )

    # Wrap in explanation container
    explanation = MultimodalExplanation(explanation_data)

    # Analyze results
    top_audio = explanation.get_top_audio_components(n=3)
    modality_contrib = explanation.get_modality_contribution()
    ```
"""

from typing import Optional

# Factorization components
from .factorization import Factorization, TimeOnlyFactorization, OpenunmixFactorization

# LIME components
from .lime import LimeBase, LimeMusicExplainer

# Explanation container
from .musiclime import MultimodalExplanation

# Explanation wrapper
from .musiclime_wrapper import create_musiclime_wrapper

# Utilities
from .utils import (
    validate_audio_file,
    validate_lyrics_lines,
    normalize_importance_scores,
    compute_explanation_stability,
    create_classifier_wrapper,
    merge_explanations,
    get_component_time_ranges,
    validate_explanation_consistency,
)

__version__ = "1.0.0"

__all__ = [
    # Factorization
    "Factorization",
    "TimeOnlyFactorization",
    "OpenunmixFactorization",
    # LIME
    "LimeBase",
    "LimeMusicExplainer",
    # Explanation
    "MultimodalExplanation",
    # Explanation Wrapper
    "create_musiclime_wrapper",
    # Utilities
    "validate_audio_file",
    "validate_lyrics_lines",
    "normalize_importance_scores",
    "compute_explanation_stability",
    "create_classifier_wrapper",
    "merge_explanations",
    "get_component_time_ranges",
    "validate_explanation_consistency",
]


# Convenience function for quick setup
def create_temporal_explainer(
    segment_duration: float = 1.0,
    kernel_width: float = 0.25,
    verbose: bool = False,
    random_state: Optional[int] = None,
) -> LimeMusicExplainer:
    """
    Create a MusicLIME explainer with temporal factorization.

    Parameters
    ----------
    segment_duration : float, default=1.0
        Duration of each temporal segment in seconds
    kernel_width : float, default=0.25
        Kernel width for LIME
    verbose : bool, default=False
        Whether to print progress information
    random_state : Optional[int], default=None
        Random seed for reproducibility

    Returns
    -------
    LimeMusicExplainer
        Configured explainer ready for use
    """
    factorizer = TimeOnlyFactorization(segment_duration=segment_duration)
    return LimeMusicExplainer(
        audio_factorization=factorizer,
        kernel_width=kernel_width,
        verbose=verbose,
        random_state=random_state,
    )


def create_source_separation_explainer(
    model_name: str = "umxhq",
    device: Optional[str] = None,
    kernel_width: float = 0.25,
    verbose: bool = False,
    random_state: Optional[int] = None,
) -> LimeMusicExplainer:
    """
    Create a MusicLIME explainer with source separation factorization.

    Parameters
    ----------
    model_name : str, default="umxhq"
        OpenUnmix model variant to use
    device : Optional[str], default=None
        Device for inference. If None, auto-detects CUDA availability
    kernel_width : float, default=0.25
        Kernel width for LIME
    verbose : bool, default=False
        Whether to print progress information
    random_state : Optional[int], default=None
        Random seed for reproducibility

    Returns
    -------
    LimeMusicExplainer
        Configured explainer ready for use
    """
    factorizer = OpenunmixFactorization(model_name=model_name, device=device)
    return LimeMusicExplainer(
        audio_factorization=factorizer,
        kernel_width=kernel_width,
        verbose=verbose,
        random_state=random_state,
    )
