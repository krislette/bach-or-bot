import joblib
import numpy as np
from src.preprocessing.preprocessor import single_preprocessing
from src.spectttra.spectttra_trainer import spectttra_train
from src.llm2vectrain.llm2vec_trainer import l2vec_train
from src.llm2vectrain.model import load_llm2vec_model
from src.models.mlp import build_mlp, load_config


class MusicLIMEPredictor:
    def __init__(self):
        print("[MusicLIME] Loading models for MusicLIME...")
        self.llm2vec_model = load_llm2vec_model()
        config = load_config("config/model_config.yml")
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
        print(f"[MusicLIME] Processing {len(texts)} samples with batch functions...")

        # Step 1: Preprocess all samples (still needs to be individual)
        print("[MusicLIME] Preprocessing samples...")
        processed_audios = []
        processed_lyrics = []

        for i, (text, audio) in enumerate(zip(texts, audios)):
            if i % 100 == 0:
                print(f"   Preprocessing {i+1}/{len(texts)}")
            processed_audio, processed_lyric = single_preprocessing(audio, text)
            processed_audios.append(processed_audio)
            processed_lyrics.append(processed_lyric)

        # Step 2: BATCH feature extraction
        print("[MusicLIME] Extracting audio features (batch)...")
        audio_features_batch = spectttra_train(processed_audios)  # (batch, 384)

        print("[MusicLIME] Extracting lyrics features (batch)...")
        lyrics_features_batch = l2vec_train(
            self.llm2vec_model, processed_lyrics
        )  # (batch, 4096)

        # Step 3: Scale and reduce in batch
        print("[MusicLIME] Scaling and reducing features (batch)...")

        # Load the trained scalers
        audio_scaler = joblib.load("models/fusion/audio_scaler.pkl")
        lyric_scaler = joblib.load("models/fusion/lyrics_scaler.pkl")

        # Then apply scaling to the batch
        scaled_audio_batch = audio_scaler.transform(
            audio_features_batch
        )  # (batch, 384)
        scaled_lyrics_batch = lyric_scaler.transform(
            lyrics_features_batch
        )  # (batch, 2048)

        # Step 4: Apply PCA to lyrics batch
        print("[MusicLIME] Applying PCA to lyrics (batch)")
        pca_model = joblib.load("models/fusion/pca.pkl")
        reduced_lyrics_batch = pca_model.transform(
            scaled_lyrics_batch
        )  # (batch, 512)

        # Step 5: Apply scaler to PCA-scaled lyrics batch
        print("[MusicLIME] Reapplying scaler to PCA-scaled batch")
        pca_scaler = joblib.load("models/fusion/pca_scaler.pkl")
        reduced_lyrics_batch = pca_scaler.transform(
            reduced_lyrics_batch
        ) # (batch, 512)

        # Step 6: Concatenate features
        combined_features_batch = np.concatenate(
            [scaled_audio_batch, reduced_lyrics_batch], axis=1
        )  # (batch, sum of lyrics & audio vector dims)

        # Step 6: BATCH MLP prediction
        print("[MusicLIME] Running MLP predictions (batch)...")
        if self.classifier is None:
            self.classifier = build_mlp(
                input_dim=combined_features_batch.shape[1], config=self.config
            )
            self.classifier.load_model("models/mlp/mlp_best.pth")

        probabilities, predictions = self.classifier.predict(combined_features_batch)

        # Convert to expected format
        batch_results = [[1 - prob, prob] for prob in probabilities]
        print(f"[MusicLIME] Batch processing complete!")
        return np.array(batch_results)
