import time
import joblib
import numpy as np

from src.preprocessing.preprocessor import single_preprocessing
from src.spectttra.spectttra_trainer import spectttra_train
from src.llm2vectrain.llm2vec_trainer import l2vec_train
from src.llm2vectrain.model import load_llm2vec_model
from src.models.mlp import build_mlp, load_config
from src.musiclime.print_utils import green_bold


class MusicLIMEPredictor:
    """
    Batch prediction wrapper for MusicLIME explanations.

    Integrates the complete Bach or Bot pipeline (SpecTTTra + LLM2Vec + MLP)
    into a single callable for LIME perturbation processing. Optimized for
    batch processing of multiple perturbed audio-lyrics pairs with detailed
    timing analysis.

    Attributes
    ----------
    llm2vec_model : LLM2Vec
        Pre-loaded LLM2Vec model for lyrics feature extraction
    classifier : MLPClassifier
        Lazy-loaded MLP classifier for final predictions
    config : dict
        Model configuration parameters
    """

    def __init__(self):
        """
        Initialize MusicLIME prediction wrapper with pre-trained models.

        Loads LLM2Vec model and MLP configuration for batch processing
        of perturbed audio-lyrics pairs during LIME explanation.
        """
        print("[MusicLIME] Loading models for MusicLIME...")
        self.llm2vec_model = load_llm2vec_model()
        config = load_config("config/model_config.yml")
        self.classifier = None
        self.config = config

    def __call__(self, texts, audios):
        """
        Batch prediction function for MusicLIME perturbations.

        Processes multiple perturbed audio-lyrics pairs through the complete
        pipeline: preprocessing -> feature extraction -> scaling -> MLP prediction.
        Optimized for batch processing of LIME perturbations.

        Parameters
        ----------
        texts : list of str
            List of perturbed lyrics strings from LIME
        audios : list of array-like
            List of perturbed audio waveforms from LIME

        Returns
        -------
        ndarray
            Prediction probabilities in format [[P(AI), P(Human)], ...]
            for each input pair, shape (n_samples, 2)
        """
        print(f"[MusicLIME] Processing {len(texts)} samples with batch functions...")

        # Step 1: Preprocess all samples (still needs to be individual)
        start_time = time.time()
        print("[MusicLIME] Preprocessing samples...")
        processed_audios = []
        processed_lyrics = []

        for _, (text, audio) in enumerate(zip(texts, audios)):
            processed_audio, processed_lyric = single_preprocessing(audio, text)
            processed_audios.append(processed_audio)
            processed_lyrics.append(processed_lyric)

        preprocessing_time = time.time() - start_time
        print(
            green_bold(
                f"[MusicLIME] Preprocessing completed in {preprocessing_time:.2f}s"
            )
        )

        # Step 2: Batch feature extraction
        start_time = time.time()
        print("[MusicLIME] Extracting audio features (batch)...")
        audio_features_batch = spectttra_train(processed_audios)  # (batch, 384)
        audio_time = time.time() - start_time
        print(
            green_bold(
                f"[MusicLIME] Audio feature extraction completed in {audio_time:.2f}s"
            )
        )

        start_time = time.time()
        print("[MusicLIME] Extracting lyrics features (batch)...")
        lyrics_features_batch = l2vec_train(
            self.llm2vec_model, processed_lyrics
        )  # (batch, 2048)
        lyrics_time = time.time() - start_time
        print(
            green_bold(
                f"[MusicLIME] Lyrics feature extraction completed in {lyrics_time:.2f}s"
            )
        )

        # TODO: Remove debug prints
        print(f"[DEBUG] wrapper.py audio features shape: {audio_features_batch.shape}")
        print(
            f"[DEBUG] wrapper.py lyrics features shape: {lyrics_features_batch.shape}"
        )
        print(
            f"[DEBUG] wrapper.py audio features[0] first 5: {audio_features_batch[0][:5]}"
        )
        print(
            f"[DEBUG] wrapper.py lyrics features[0] first 5: {lyrics_features_batch[0][:5]}"
        )

        # Step 3: Scale and reduce in batch
        start_time = time.time()
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
        reduced_lyrics_batch = pca_model.transform(scaled_lyrics_batch)  # (batch, 512)

        # Step 5: Concatenate features
        combined_features_batch = np.concatenate(
            [scaled_audio_batch, reduced_lyrics_batch], axis=1
        )  # (batch, sum of lyrics & audio vector dims)
        scaling_time = time.time() - start_time
        print(green_bold(f"[MusicLIME] Scaling completed in {scaling_time:.2f}s"))

        # Step 6: Batch MLP prediction
        start_time = time.time()
        print("[MusicLIME] Running MLP predictions (batch)...")
        if self.classifier is None:
            self.classifier = build_mlp(
                input_dim=combined_features_batch.shape[1], config=self.config
            )
            self.classifier.load_model("models/mlp/mlp_best.pth")

        probabilities, predictions = self.classifier.predict(combined_features_batch)

        # Convert to expected format
        batch_results = [[1 - prob, prob] for prob in probabilities]
        mlp_time = time.time() - start_time
        print(green_bold(f"[MusicLIME] MLP prediction completed in {mlp_time:.2f}s"))

        # Total time summary
        total_time = (
            preprocessing_time + audio_time + lyrics_time + scaling_time + mlp_time
        )
        print(f"[MusicLIME] Batch processing complete!")
        print(
            green_bold(
                f"[MusicLIME] Total time: {total_time:.2f}s (Preprocessing: {preprocessing_time:.2f}s, Audio: {audio_time:.2f}s, Lyrics: {lyrics_time:.2f}s, Scaling: {scaling_time:.2f}s, MLP: {mlp_time:.2f}s)"
            )
        )

        return np.array(batch_results)
