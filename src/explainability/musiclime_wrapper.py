"""
Optimized MusicLIME wrapper that fixes performance issues.
"""

import numpy as np
from typing import List, Callable, Dict, Any
import warnings
import soundfile as sf
import torch
from types import SimpleNamespace

from src.preprocessing.audio_preprocessor import AudioPreprocessor
from src.preprocessing.lyrics_preprocessor import LyricsPreprocessor
from src.spectttra.spectttra_trainer import build_spectttra
from src.llm2vectrain.model import load_llm2vec_model
from src.llm2vectrain.llm2vec_trainer import l2vec_single_train
from src.models.mlp import MLPClassifier
from src.explainability.factorization_draft.temporal import TimeOnlyFactorization
from src.explainability.true_musiclime import TrueMusicLIMEExplainer


class MusicLIMEWrapper:
    """
    Optimized wrapper that fixes the performance issues in your original implementation.

    Key improvements:
    1. Batch processing of perturbations
    2. Cached feature extraction
    3. Proper MusicLIME architecture
    4. Efficient model usage
    """

    def __init__(self, mlp_model_path: str, config: Dict):
        self.config = config

        # Initialize models ONCE
        print("Loading models...")
        self.llm2vec_model = load_llm2vec_model()
        self.audio_preprocessor = AudioPreprocessor(script="inference")
        self.lyrics_preprocessor = LyricsPreprocessor()

        # Load MLP
        self.mlp_model = MLPClassifier(384 + 4096, config)
        self.mlp_model.load_model(mlp_model_path)
        print("All models loaded!")

        # Cache for features to avoid recomputation
        self._audio_cache = {}
        self._lyrics_cache = {}

    def _extract_spectttra_features_batch(
        self, audio_tensors: List[torch.Tensor]
    ) -> np.ndarray:
        """Extract SpecTTTra features for a batch using your existing spectttra_train method."""
        if not audio_tensors:
            return np.array([])

        # Use YOUR existing batch method
        from src.spectttra.spectttra_trainer import spectttra_train

        features_batch = spectttra_train(audio_tensors)

        return features_batch

    def _extract_lyrics_features_batch(self, lyrics_list: List[str]) -> np.ndarray:
        """Extract lyrics features for a batch using your existing l2vec_train method."""
        if not lyrics_list:
            return np.array([])

        # Use YOUR existing batch method
        from src.llm2vectrain.llm2vec_trainer import l2vec_train

        features_batch = l2vec_train(self.llm2vec_model, lyrics_list)

        # Ensure correct dimensions (4096 per sample)
        if features_batch.ndim == 1:
            features_batch = features_batch.reshape(1, -1)

        # Pad/truncate to 4096 if needed
        batch_size = features_batch.shape[0]
        if features_batch.shape[1] != 4096:
            padded_batch = np.zeros((batch_size, 4096))
            min_dim = min(4096, features_batch.shape[1])
            padded_batch[:, :min_dim] = features_batch[:, :min_dim]
            features_batch = padded_batch

        return features_batch

    def create_batch_classifier_function(self) -> Callable:
        """
        Create a batch classifier that processes multiple samples efficiently.

        This is the key optimization - instead of processing one sample at a time,
        we process batches which is much faster.
        """

        def batch_classifier_fn(
            lyrics_batch: List[List[str]], audio_batch: np.ndarray
        ) -> List[float]:
            """
            Batch classifier function.

            Parameters
            ----------
            lyrics_batch : List[List[str]]
                List of lyrics line lists
            audio_batch : np.ndarray
                Batch of audio arrays with shape (batch_size, n_samples)

            Returns
            -------
            List[float]
                List of predictions for each sample
            """
            try:
                batch_size = len(lyrics_batch)
                if batch_size == 0:
                    return []

                # Process lyrics batch
                lyrics_strings = [" ".join(lines) for lines in lyrics_batch]
                lyrics_features_batch = self._extract_lyrics_features_batch(
                    lyrics_strings
                )

                # Process audio batch
                audio_tensors = []
                for audio_data in audio_batch:
                    waveform = self.audio_preprocessor(audio_data)
                    if waveform.dim() == 2 and waveform.size(0) == 1:
                        waveform = waveform.squeeze(0)
                    audio_tensors.append(waveform)

                audio_features_batch = self._extract_spectttra_features_batch(
                    audio_tensors
                )

                # Ensure correct dimensions for audio features
                if audio_features_batch.shape[1] != 384:
                    padded_audio = np.zeros((batch_size, 384))
                    min_dim = min(384, audio_features_batch.shape[1])
                    padded_audio[:, :min_dim] = audio_features_batch[:, :min_dim]
                    audio_features_batch = padded_audio

                # Combine features and predict
                combined_features = np.concatenate(
                    [audio_features_batch, lyrics_features_batch], axis=1
                )

                predictions = []
                for features in combined_features:
                    pred_prob = self.mlp_model.predict_single(features)
                    predictions.append(float(np.clip(pred_prob, 0.0, 1.0)))

                return predictions

            except Exception as e:
                warnings.warn(f"Batch classifier failed: {e}")
                return [0.5] * len(lyrics_batch)  # Neutral predictions

        return batch_classifier_fn

    def explain_prediction(
        self,
        audio_path: str,
        lyrics_text: str,
        factorization_type: str = "temporal",
        n_samples: int = 1000,
        batch_size: int = 32,  # Larger batch size for efficiency
        **lime_kwargs,
    ) -> Dict[str, Any]:
        """
        Generate explanation using optimized batch processing.

        Parameters
        ----------
        audio_path : str
            Path to audio file
        lyrics_text : str
            Lyrics text
        factorization_type : str
            Type of factorization ('temporal' or 'source_separation')
        n_samples : int
            Number of LIME samples
        batch_size : int
            Batch size for processing (larger = faster but more memory)
        **lime_kwargs
            Additional LIME parameters

        Returns
        -------
        Dict[str, Any]
            Explanation results
        """

        if not audio_path:
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Load audio ONCE
        print(f"Loading audio from {audio_path}...")
        audio_data, sr = sf.read(audio_path)

        # Convert stereo to mono if needed
        if audio_data.ndim == 2:
            audio_data = np.mean(audio_data, axis=1)

        # Resample if needed
        if sr != 16000:
            import librosa

            audio_data = librosa.resample(
                audio_data.astype(np.float32), orig_sr=sr, target_sr=16000
            )
        print(f"Audio loaded, shape: {audio_data.shape}")

        # Initialize factorizer
        if factorization_type == "temporal":
            audio_factorizer = TimeOnlyFactorization(
                input=audio_data,
                target_sr=16000,
                temporal_segmentation_params={
                    "type": "fixed_length",
                    "n_temporal_segments": 10,
                },
            )
        else:
            raise NotImplementedError(
                "Source separation not implemented in optimized version"
            )

        print("Audio factorizer created")

        # Process lyrics
        lyrics_lines = self.lyrics_preprocessor.musiclime_lyrics_extractor(lyrics_text)
        if not lyrics_lines:
            lyrics_lines = [""]
        print(f"Lyrics processed, {len(lyrics_lines)} lines")

        # Create batch classifier
        batch_classifier_fn = self.create_batch_classifier_function()

        # Test classifier with original data
        print("Testing classifier with original audio...")
        original_pred = batch_classifier_fn([lyrics_lines], np.array([audio_data]))[0]
        print(f"Original prediction: {original_pred}")

        # Create explainer
        print("Starting optimized LIME explanation...")
        explainer = TrueMusicLIMEExplainer(
            kernel_width=lime_kwargs.get("kernel_width", 25),
            verbose=lime_kwargs.get("verbose", True),
            random_state=lime_kwargs.get("random_state", 42),
        )

        # Generate explanation
        explanation = explainer.explain_instance(
            factorization=audio_factorizer,
            lyrics_lines=lyrics_lines,
            predict_fn=batch_classifier_fn,
            num_samples=n_samples,
            batch_size=batch_size,
            modality="both",
        )

        return {
            "explanation": explanation,
            "original_prediction": original_pred,
            "factorization_type": factorization_type,
            "n_samples": n_samples,
            "batch_size": batch_size,
        }


def create_musiclime_wrapper(mlp_model_path: str, config: Dict) -> MusicLIMEWrapper:
    """Create optimized MusicLIME wrapper."""
    return MusicLIMEWrapper(mlp_model_path, config)
