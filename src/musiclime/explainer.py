import json
import numpy as np
import sklearn.metrics
import time

from functools import partial
from sklearn.utils import check_random_state
from lime.lime_base import LimeBase
from pathlib import Path
from datetime import datetime

from src.musiclime.text_utils import LineIndexedString
from src.musiclime.factorization import OpenUnmixFactorization
from src.musiclime.print_utils import green_bold


class MusicLIMEExplainer:
    """
    LIME-based explainer for multimodal music classification models.

    Generates local explanations for AI vs Human music classification by
    perturbing audio (source separation) and lyrics (line removal) components
    and analyzing their impact on model predictions.

    Attributes
    ----------
    random_state : RandomState
        Random number generator for reproducible perturbations
    base : LimeBase
        Core LIME explanation engine with exponential kernel
    """

    def __init__(self, kernel_width=25, random_state=None):
        """
        Initialize MusicLIME explainer with kernel parameters.

        Parameters
        ----------
        kernel_width : int, default=25
            Width parameter for the exponential kernel function
        random_state : int or RandomState, optional
            Random seed for reproducible perturbations
        """
        self.random_state = check_random_state(random_state)

        def kernel(d, kernel_width):
            return np.sqrt(np.exp(-(d**2) / kernel_width**2))

        kernel_fn = partial(kernel, kernel_width=kernel_width)
        self.base = LimeBase(kernel_fn, verbose=False)

    def explain_instance(
        self,
        audio,
        lyrics,
        predict_fn,
        num_samples=1000,
        labels=(1,),
        temporal_segments=10,
    ):
        """
        Generate LIME explanations for a music instance using audio and lyrics.

        Parameters
        ----------
        audio : array-like
            Raw audio waveform data
        lyrics : str
            Song lyrics as text string
        predict_fn : callable
            Prediction function that takes (texts, audios) and returns probabilities (wrapper)
        num_samples : int, default=1000
            Number of perturbed samples to generate for LIME
        labels : tuple, default=(1,)
            Target labels to explain (0=AI-Generated, 1=Human-Composed)
        temporal_segments : int, default=10
            Number of temporal segments for audio factorization

        Returns
        -------
        MusicLIMEExplanation
            Explanation object containing feature importance weights
        """
        # These are for debugging only I have to see THAT progress
        print("[MusicLIME] Starting MusicLIME explanation...")
        print(
            f"[MusicLIME] Audio length: {len(audio)/22050:.1f}s, Temporal segments: {temporal_segments}"
        )
        print(f"[MusicLIME] Lyrics lines: {len(lyrics.split(chr(10)))}")

        # Create factorizations
        print("[MusicLIME] Creating audio factorization (source separation)...")
        audio_factorization = OpenUnmixFactorization(
            audio, temporal_segmentation_params=temporal_segments
        )
        print(
            f"[MusicLIME] Audio components: {audio_factorization.get_number_components()}"
        )

        start_time = time.time()
        print("[MusicLIME] Processing lyrics...")
        text_factorization = LineIndexedString(lyrics)
        print(f"[MusicLIME] Text lines: {text_factorization.num_words()}")
        text_processing_time = time.time() - start_time
        print(
            green_bold(
                f"[MusicLIME] Lyrics processing completed in {text_processing_time:.2f}s"
            )
        )

        # Generate perturbations and get predictions
        print(f"[MusicLIME] Generating {num_samples} perturbations...")
        data, predictions, distances = self._generate_neighborhood(
            audio_factorization, text_factorization, predict_fn, num_samples
        )

        # LIME fitting, create explanation object
        start_time = time.time()
        print("[MusicLIME] Fitting LIME model...")
        explanation = MusicLIMEExplanation(
            audio_factorization, text_factorization, data, predictions
        )

        for label in labels:
            print(f"[MusicLIME] Explaining label {label}...")
            (
                explanation.intercept[label],
                explanation.local_exp[label],
                explanation.score[label],
                explanation.local_pred[label],
            ) = self.base.explain_instance_with_data(
                data, predictions, distances, label, num_features=20
            )

        lime_time = time.time() - start_time
        print(
            green_bold(f"[MusicLIME] LIME model fitting completed in {lime_time:.2f}s")
        )
        print("[MusicLIME] MusicLIME explanation complete!")

        return explanation

    def _generate_neighborhood(self, audio_fact, text_fact, predict_fn, num_samples):
        """
        Generate perturbed samples and predictions for LIME explanation.

        Parameters
        ----------
        audio_fact : OpenUnmixFactorization
            Audio factorization object for source separation
        text_fact : LineIndexedString
            Text factorization object for line-based perturbations
        predict_fn : callable
            Model prediction function
        num_samples : int
            Number of perturbations to generate

        Returns
        -------
        data : ndarray
            Binary perturbation masks (num_samples, total_features)
        predictions : ndarray
            Model predictions for perturbed instances
        distances : ndarray
            Cosine distances from original instance
        """
        n_audio = audio_fact.get_number_components()
        n_text = text_fact.num_words()
        total_features = n_audio + n_text

        print(
            f"[MusicLIME] Total features: {total_features} ({n_audio} audio + {n_text} text)"
        )

        # Generate binary masks
        start_time = time.time()
        print("[MusicLIME] Generating perturbation masks...")
        data = self.random_state.randint(0, 2, num_samples * total_features).reshape(
            (num_samples, total_features)
        )
        data[0, :] = 1  # Original instance
        mask_time = time.time() - start_time
        print(green_bold(f"[MusicLIME] Mask generation completed in {mask_time:.2f}s"))

        # Generate perturbed instances
        start_time = time.time()
        texts = []
        audios = []

        for i, row in enumerate(data):
            # Progress check for every hundred samples
            if i % 100 == 0:
                print(f"[MusicLIME]     Progress: {i}/{num_samples} samples")

            # Audio perturbation & reconstruction
            audio_mask = row[:n_audio]
            active_audio_components = np.where(audio_mask != 0)[0]
            perturbed_audio = audio_fact.compose_model_input(active_audio_components)
            audios.append(perturbed_audio)

            # Text perturbation & reconstruction
            text_mask = row[n_audio:]
            inactive_lines = np.where(text_mask == 0)[0]
            perturbed_text = text_fact.inverse_removing(inactive_lines)
            texts.append(perturbed_text)

        perturbation_time = time.time() - start_time
        print(
            green_bold(
                f"[MusicLIME] Perturbation creation completed in {perturbation_time:.2f}s"
            )
        )

        # Get predictions
        print(f"[MusicLIME] Getting predictions for {len(texts)} samples...")
        predictions = predict_fn(texts, audios)

        # Show the original prediction (first row is always the unperturbed original)
        original_prediction = predictions[0]
        predicted_class = np.argmax(original_prediction)  # 0 = AI, 1 = Human
        confidence = original_prediction[predicted_class]

        # Print original prediction
        print(f"[MusicLIME] Original Prediction:")
        print(
            f"  Raw probabilities: [AI: {original_prediction[0]:.3f}, Human: {original_prediction[1]:.3f}]"
        )
        print(
            f"  Predicted class: {'AI-Generated' if predicted_class == 0 else 'Human-Composed'}"
        )
        print(f"  Confidence: {confidence:.3f}")

        # Debug prints
        print(f"[MusicLIME] Predictions shape: {predictions.shape}")
        print(f"[MusicLIME] Predictions:\n{predictions}")
        print(f"[MusicLIME] Prediction variance: {np.var(predictions, axis=0)}")
        print(
            f"[MusicLIME] Prediction range: min={np.min(predictions, axis=0)}, max={np.max(predictions, axis=0)}"
        )

        # Check if all predictions are identical
        if np.allclose(predictions, predictions[0]):
            print(
                "[MusicLIME] WARNING: All predictions are identical! LIME cannot learn from this."
            )

        # Calculate distances
        print("[MusicLIME] Calculating distances...")
        distances = sklearn.metrics.pairwise_distances(
            data, data[0].reshape(1, -1), metric="cosine"
        ).ravel()

        # Prints for debugging
        print(
            f"[MusicLIME] Distance range: min={np.min(distances)}, max={np.max(distances)}"
        )
        print(
            f"[MusicLIME] Data variance: {np.var(data, axis=0)[:10]}..."
        )  # First 10 features

        return data, predictions, distances


class MusicLIMEExplanation:
    """
    Container for MusicLIME explanation results and analysis methods.

    Stores factorizations, perturbation data, and LIME-fitted explanations
    for a single music instance. Provides methods to extract top features
    and export results to structured formats.

    Attributes
    ----------
    audio_factorization : OpenUnmixFactorization
        Audio source separation components
    text_factorization : LineIndexedString
        Lyrics line segmentation components
    data : ndarray
        Binary perturbation masks used for explanation
    predictions : ndarray
        Model predictions for all perturbations
    intercept : dict
        LIME model intercepts by label
    local_exp : dict
        Feature importance weights by label
    score : dict
        LIME model R² scores by label
    local_pred : dict
        Local model predictions by label
    """

    def __init__(self, audio_factorization, text_factorization, data, predictions):
        """
        Initialize explanation object with factorizations and prediction data.

        Parameters
        ----------
        audio_factorization : OpenUnmixFactorization
            Audio source separation components
        text_factorization : LineIndexedString
            Text line segmentation components
        data : ndarray
            Binary perturbation masks used for explanation
        predictions : ndarray
            Model predictions for all perturbations
        """
        self.audio_factorization = audio_factorization
        self.text_factorization = text_factorization
        self.data = data
        self.predictions = predictions
        self.intercept = {}
        self.local_exp = {}
        self.score = {}
        self.local_pred = {}

    def get_explanation(self, label, num_features=10):
        """
        Extract top feature explanations for a specific label.

        Parameters
        ----------
        label : int
            Target label to explain (0=AI-Generated, 1=Human-Composed)
        num_features : int, default=10
            Number of top features to return

        Returns
        -------
        list of dict
            Feature explanations with type, feature description, and weight
        """
        if label not in self.local_exp:
            return []

        exp = self.local_exp[label][:num_features]
        n_audio = self.audio_factorization.get_number_components()

        explanations = []
        for feature_idx, weight in exp:
            if feature_idx < n_audio:
                # Audio component
                component_name = self.audio_factorization.get_ordered_component_names()[
                    feature_idx
                ]
                explanations.append(
                    {"type": "audio", "feature": component_name, "weight": weight}
                )
            else:
                # Text line
                line_idx = feature_idx - n_audio
                line_text = self.text_factorization.word(line_idx)
                explanations.append(
                    {"type": "lyrics", "feature": line_text, "weight": weight}
                )

        return explanations

    def save_to_json(self, filepath, song_info=None, num_features=10):
        """
        Save explanation results to structured JSON file.

        Parameters
        ----------
        filepath : str
            Output filename for JSON results
        song_info : dict, optional
            Additional metadata about the song
        num_features : int, default=10
            Number of top features to include in output

        Returns
        -------
        Path
            Path to the saved JSON file
        """
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)

        # Get explanation data
        explanation_data = {}
        for label in self.local_exp.keys():
            features = self.get_explanation(label, num_features)

            explanation_data[f"label_{label}"] = {
                "prediction_label": "Human-Composed" if label == 1 else "AI-Generated",
                "intercept": float(self.intercept.get(label, 0)),
                "score": float(self.score.get(label, 0)),
                "local_prediction": (
                    float(self.local_pred.get(label, [0])[0])
                    if self.local_pred.get(label)
                    else 0
                ),
                "top_features": [
                    {
                        "rank": i + 1,
                        "type": item["type"],
                        "feature": item["feature"],
                        "weight": float(item["weight"]),
                    }
                    for i, item in enumerate(features)
                ],
            }

        # Create complete JSON structure
        json_output = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "song_info": song_info or {},
                "model_info": {
                    "total_audio_components": self.audio_factorization.get_number_components(),
                    "total_text_lines": self.text_factorization.num_words(),
                    "total_features": self.audio_factorization.get_number_components()
                    + self.text_factorization.num_words(),
                },
                "explanation_params": {
                    "num_samples": len(self.data),
                    "num_features_shown": num_features,
                },
            },
            "explanations": explanation_data,
        }

        # Save to results folder
        output_path = results_dir / filepath
        with open(output_path, "w") as f:
            json.dump(json_output, f, indent=2)

        print(f"[MusicLIME] Explanation saved to: {output_path}")
        return output_path
