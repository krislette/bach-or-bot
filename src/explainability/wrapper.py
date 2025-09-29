import numpy as np
from src.preprocessing.preprocessor import single_preprocessing
from src.spectttra.spectttra_trainer import spectttra_predict
from src.llm2vectrain.model import load_llm2vec_model
from src.llm2vectrain.llm2vec_trainer import l2vec_single_train, load_pca_model
from src.models.mlp import build_mlp, load_config
from src.utils.dataset import instance_scaler


class MusicLIMEPredictor:
    def __init__(self):
        # Load models once
        self.llm2vec_model = load_llm2vec_model()
        config = load_config("config/model_config.yml")

        # We'll set input_dim after first prediction to avoid loading issues
        self.classifier = None
        self.config = config

    def __call__(self, texts, audios):
        """
        Predict function for MusicLIME

        Args:
            texts: List of lyric strings
            audios: Array of audio waveforms

        Returns:
            Array of prediction probabilities
        """
        print(f"[MusicLIME] MusicLIME Predictor: Processing {len(texts)} samples...")
        batch_results = []

        for text, audio in zip(texts, audios):
            # Preprocess
            processed_audio, processed_lyrics = single_preprocessing(audio, text)

            # Extract features
            audio_features = spectttra_predict(processed_audio)
            lyrics_features = l2vec_single_train(self.llm2vec_model, processed_lyrics)

            # Scale and reduce
            audio_features, lyrics_features = instance_scaler(
                audio_features, lyrics_features
            )
            reduced_lyrics = load_pca_model(lyrics_features)

            # Combine features
            combined_features = np.concatenate([audio_features, reduced_lyrics], axis=1)

            # Initialize classifier on first run
            if self.classifier is None:
                self.classifier = build_mlp(
                    input_dim=combined_features.shape[1], config=self.config
                )
                self.classifier.load_model("models/mlp/mlp_multimodal.pth")

            # Get prediction probability
            prob, _, _ = self.classifier.predict_single(combined_features)
            batch_results.append([1 - prob, prob])  # [AI_prob, Human_prob]

        return np.array(batch_results)
