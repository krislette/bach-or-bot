import numpy as np
from src.preprocessing.preprocessor import single_preprocessing
from src.spectttra.spectttra_trainer import spectttra_train
from src.llm2vectrain.llm2vec_trainer import l2vec_train, load_pca_model
from src.llm2vectrain.model import load_llm2vec_model
from src.models.mlp import build_mlp, load_config
from src.utils.dataset import instance_scaler


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
        audio_features_batch = spectttra_train(processed_audios)

        print("[MusicLIME] Extracting lyrics features (batch)...")
        lyrics_features_batch = l2vec_train(self.llm2vec_model, processed_lyrics)

        # Step 3: Scale and reduce (individual for now, could be batched)
        print("[MusicLIME] Scaling and reducing features...")
        all_features = []
        for audio_feat, lyrics_feat in zip(audio_features_batch, lyrics_features_batch):
            # Reshape for instance_scaler (expects 2D)
            audio_feat = audio_feat.reshape(1, -1)
            lyrics_feat = lyrics_feat.reshape(1, -1)

            scaled_audio, scaled_lyrics = instance_scaler(audio_feat, lyrics_feat)
            reduced_lyrics = load_pca_model(scaled_lyrics)
            combined = np.concatenate([scaled_audio, reduced_lyrics], axis=1)
            all_features.append(combined)

        # Step 4: BATCH MLP prediction
        print("[MusicLIME] Running MLP predictions (batch)...")
        if self.classifier is None:
            self.classifier = build_mlp(
                input_dim=all_features[0].shape[1], config=self.config
            )
            self.classifier.load_model("models/mlp/mlp_multimodal.pth")

        batch_features = np.vstack(all_features)
        probabilities, predictions = self.classifier.predict(batch_features)

        # Convert to expected format
        batch_results = [[1 - prob, prob] for prob in probabilities]
        print(f"[MusicLIME] Batch processing complete!")
        return np.array(batch_results)
