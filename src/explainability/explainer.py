import numpy as np
import sklearn.metrics
from functools import partial
from sklearn.utils import check_random_state
from lime.lime_base import LimeBase

from src.explainability.text_utils import LineIndexedString
from src.explainability.factorization import OpenUnmixFactorization


class MusicLIMEExplainer:
    def __init__(self, kernel_width=25, random_state=None):
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
        # These are for debugging only I have to see THAT progress
        print("[MusicLIME] Starting MusicLIME explanation...")
        print(
            f"[MusicLIME] Audio length: {len(audio)/44100:.1f}s, Temporal segments: {temporal_segments}"
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

        print("[MusicLIME] Processing lyrics...")
        text_factorization = LineIndexedString(lyrics)
        print(f"[MusicLIME] Text lines: {text_factorization.num_words()}")

        # Generate perturbations and get predictions
        print(f"[MusicLIME] Generating {num_samples} perturbations...")
        data, predictions, distances = self._generate_neighborhood(
            audio_factorization, text_factorization, predict_fn, num_samples
        )

        # Create explanation object
        print("[MusicLIME] Fitting LIME model...")
        explanation = MusicLIMEExplanation(
            audio_factorization, text_factorization, data, predictions
        )

        # Fit local model for each label
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

        print("[MusicLIME] MusicLIME explanation complete!")
        return explanation

    def _generate_neighborhood(self, audio_fact, text_fact, predict_fn, num_samples):
        n_audio = audio_fact.get_number_components()
        n_text = text_fact.num_words()
        total_features = n_audio + n_text

        print(
            f"[MusicLIME] Total features: {total_features} ({n_audio} audio + {n_text} text)"
        )

        # Generate binary masks
        print("[MusicLIME] Generating perturbation masks...")
        data = self.random_state.randint(0, 2, num_samples * total_features).reshape(
            (num_samples, total_features)
        )
        data[0, :] = 1  # Original instance

        # Generate perturbed instances
        texts = []
        audios = []

        for i, row in enumerate(data):
            # Progress check for every hundred samples
            if i % 100 == 0:
                print(f"[MusicLIME]     Progress: {i}/{num_samples} samples")

            # Audio perturbation
            audio_mask = row[:n_audio]
            active_audio_components = np.where(audio_mask != 0)[0]
            perturbed_audio = audio_fact.compose_model_input(active_audio_components)
            audios.append(perturbed_audio)

            # Text perturbation
            text_mask = row[n_audio:]
            inactive_lines = np.where(text_mask == 0)[0]
            perturbed_text = text_fact.inverse_removing(inactive_lines)
            texts.append(perturbed_text)

        # Get predictions
        print(f"[MusicLIME] Getting predictions for {len(texts)} samples...")
        predictions = predict_fn(texts, audios)
        print(f"[MusicLIME] Predictions shape: {predictions.shape}")

        # Calculate distances
        print("[MusicLIME] Calculating distances...")
        distances = sklearn.metrics.pairwise_distances(
            data, data[0].reshape(1, -1), metric="cosine"
        ).ravel()

        return data, predictions, distances


class MusicLIMEExplanation:
    def __init__(self, audio_factorization, text_factorization, data, predictions):
        self.audio_factorization = audio_factorization
        self.text_factorization = text_factorization
        self.data = data
        self.predictions = predictions
        self.intercept = {}
        self.local_exp = {}
        self.score = {}
        self.local_pred = {}

    def get_explanation(self, label, num_features=10):
        """Get top features for explanation"""
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
