"""
MusicLIME wrapper and integration module.

This module provides the classifier wrapper function and explainer setup
needed to integrate multimodal model with MusicLIME.
"""

import numpy as np
from typing import List, Callable, Dict
import warnings

from src.preprocessing.preprocessor import single_preprocessing
from src.spectttra.spectttra_trainer import spectttra_train
from src.llm2vectrain.model import load_llm2vec_model
from src.llm2vectrain.llm2vec_trainer import l2vec_train
from src.models.mlp import MLPClassifier
from src.explainability.factorization.source_separation import OpenunmixFactorization
from src.explainability.factorization.temporal import TimeOnlyFactorization
from src.explainability.musiclime import MultimodalExplanation


class MusicLIMEWrapper:
    """
    Wrapper class that integrates model pipeline with MusicLIME.
    """

    def __init__(self, mlp_model_path: str, config: Dict):
        """
        Initialize the MusicLIME wrapper.

        Parameters
        ----------
        mlp_model_path : str
            Path to the trained MLP model
        config : Dict
            Configuration dictionary for the model
        """
        self.mlp_model_path = mlp_model_path
        self.config = config

        # Initialize models
        self.llm2vec_model = load_llm2vec_model()

        # Load MLP model
        # 384 (SpecTTTra) + 4096 (LLM2Vec) = 4480 dim
        input_dim = 384 + 4096
        self.mlp_model = MLPClassifier(input_dim, config)
        self.mlp_model.load_model(mlp_model_path)

        # Initialize factorization methods
        self.source_separator = OpenunmixFactorization()
        self.temporal_segmenter = TimeOnlyFactorization(
            segment_duration=3.0
        )  # 3-second segments

    def create_classifier_function(self) -> Callable:
        """
        Create the classifier function that MusicLIME expects.

        The returned function will receive PERTURBED audio and lyrics
        from MusicLIME's perturbation_fn, so it just needs to process them.

        Returns
        -------
        Callable
            Classifier function compatible with MusicLIME
        """

        def classifier_fn(audio: np.ndarray, lyrics_lines: List[str]) -> float:
            """
            Classifier function for MusicLIME.

            This function receives already-perturbed audio and lyrics from MusicLIME
            and returns a prediction probability.

            Parameters
            ----------
            audio : np.ndarray
                Audio signal (already perturbed by MusicLIME)
            lyrics_lines : List[str]
                List of lyric lines (already perturbed by MusicLIME)

            Returns
            -------
            float
                Prediction probability (0-1 range)
            """
            try:
                # Convert lyrics lines back to single string
                # Empty lines are filtered out by MusicLIME perturbation
                lyrics_string = "\n".join(line for line in lyrics_lines if line.strip())

                # If all lyrics are empty, use empty string
                if not lyrics_string.strip():
                    lyrics_string = ""

                # Preprocess the perturbed audio and lyrics
                processed_audio, processed_lyrics = single_preprocessing(
                    audio, lyrics_string
                )

                # Extract features
                audio_features = spectttra_train(processed_audio)
                lyrics_features = l2vec_train(self.llm2vec_model, [processed_lyrics])

                # Concatenate features
                combined_features = np.concatenate(
                    [audio_features[0], lyrics_features[0]]
                )

                # Get prediction from MLP
                prediction = self.mlp_model.predict_single(combined_features)

                return prediction

            except Exception as e:
                warnings.warn(f"Classifier function error: {str(e)}")
                # Return neutral prediction on error
                return 0.5

        return classifier_fn

    def explain_prediction(
        self,
        audio_path: str,
        lyrics_text: str,
        factorization_type: str = "temporal",
        n_samples: int = 1000,
        **lime_kwargs,
    ) -> MultimodalExplanation:

        # Choose and initialize factorization method
        if factorization_type == "source_separation":
            audio_factorizer = OpenunmixFactorization(
                input=audio_path,
                target_sr=22050,
            )
        else:  # temporal
            audio_factorizer = TimeOnlyFactorization(
                input=audio_path,
                target_sr=22050,
                temporal_segmentation_params={"n_temporal_segments": 10},
            )

        # Split lyrics
        lyrics_lines = [
            line.strip() for line in lyrics_text.split("\n") if line.strip()
        ]

        # Create explainer
        from src.explainability.lime.explainer import LimeMusicExplainer

        explainer = LimeMusicExplainer(
            audio_factorization=audio_factorizer,
            verbose=lime_kwargs.get("verbose", False),
            random_state=lime_kwargs.get("random_state", None),
        )

        # Get classifier function
        classifier_fn = self.create_classifier_function()

        # Generate explanation
        explanation_data = explainer.explain_instance(
            audio_path=audio_path,
            lyrics_lines=lyrics_lines,
            classifier_fn=classifier_fn,
            n_samples=n_samples,
            **lime_kwargs,
        )

        # Create MultimodalExplanation object
        instance_info = {
            "audio_path": audio_path,
            "lyrics_text": lyrics_text,
            "factorization_type": factorization_type,
            "n_samples": n_samples,
        }

        explanation = MultimodalExplanation(
            explanation_data=explanation_data, instance_info=instance_info
        )

        return explanation


def create_musiclime_wrapper(mlp_model_path: str, config: Dict) -> MusicLIMEWrapper:
    """
    Factory function to create a MusicLIME wrapper.

    Parameters
    ----------
    mlp_model_path : str
        Path to the trained MLP model
    config : Dict
        Model configuration

    Returns
    -------
    MusicLIMEWrapper
        Configured wrapper instance
    """
    return MusicLIMEWrapper(mlp_model_path, config)


# Example usage
if __name__ == "__main__":
    # Load model configuration
    from src.models.mlp import load_config

    config = load_config()

    # Create wrapper
    wrapper = create_musiclime_wrapper("models/mlp/mlp_best.pth", config)

    # Generate explanation
    explanation = wrapper.explain_prediction(
        audio_path="path/to/audio.wav",
        lyrics_text="Sample lyrics\nLine by line\nFor explanation",
        factorization_type="temporal",
        n_samples=1000,
    )

    # Use the explanation
    print(explanation.get_summary_text())
