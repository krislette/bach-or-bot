from src.preprocessing.preprocessor import single_preprocessing
from src.spectttra.spectttra_trainer import spectttra_predict
from src.llm2vectrain.model import load_llm2vec_model
from src.llm2vectrain.llm2vec_trainer import l2vec_single_train, load_pca_model
from src.models.mlp import build_mlp, load_config
from pathlib import Path
from src.utils.config_loader import DATASET_NPZ
from src.utils.dataset import instance_scaler
from scripts.flask import request_llm2vect_single

from pathlib import Path
import numpy as np
import torch


def predict_pipeline(audio, lyrics: str):
    """
    Predict script which includes preprocessing, feature extraction, and
    training the MLP model for a single data sample.

    Parameters
    ----------
    audio : audio_object
        Audio object file

    lyric : string
        Lyric string

    Returns
    -------
    prediction : str
        A string result of the prediction

    label : int
        A numerical representation of the prediction
    """

    # Instantiate X and Y vectors
    X, Y = None, None

    # Instantiate LLM2Vec Model
    #llm2vec_model = load_llm2vec_model()

    # Preprocess both audio and lyrics
    audio, lyrics = single_preprocessing(audio, lyrics)

    # Call the train method for both models
    audio_features = spectttra_predict(audio)
    lyrics_features = request_llm2vect_single(lyrics)

    # Scale the vectors using Z-Score
    audio_features, lyrics_features = instance_scaler(audio_features, lyrics_features)

    # Reduce the lyrics using saved PCA model
    reduced_lyrics = load_pca_model(lyrics_features)

    # Concatenate the vectors of audio_features + lyrics_features
    results = np.concatenate([audio_features, reduced_lyrics], axis=1)

    # ---- Load MLP Classifier ----
    config = load_config("config/model_config.yml")
    classifier = build_mlp(input_dim=results.shape[1], config=config)

    # Load trained weights (make sure this path matches where you saved your model)
    model_path = "models/mlp/mlp_multimodal.pth"
    classifier.load_model(model_path)
    classifier.model.eval()

    # Run prediction
    probability, prediction, label = classifier.predict_single(results)

    return {
        "probability": probability,
        "label": label,
        "prediction": "Fake" if prediction == 0 else "Real",
    }


if __name__ == "__main__":
    # Example usage (replace with real inputs, place song inside data/raw.)
    audio = "multo"
    lyrics = "Some lyrics text here"
    print(predict_pipeline(audio, lyrics))
