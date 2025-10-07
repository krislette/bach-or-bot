from src.preprocessing.preprocessor import single_preprocessing
from src.spectttra.spectttra_trainer import spectttra_predict
from src.spectttra.spectttra_trainer import spectttra_predict
from src.llm2vectrain.model import load_llm2vec_model
from src.llm2vectrain.llm2vec_trainer import l2vec_single_train, load_pca_model
from src.models.mlp import build_mlp, load_config
from src.utils.dataset import instance_scaler

from pathlib import Path
import numpy as np
import pandas as pd
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

    # Instantiate LLM2Vec Model
    llm2vec_model = load_llm2vec_model()

    # Preprocess both audio and lyrics
    audio, lyrics = single_preprocessing(audio, lyrics)

    # Call the train method for both models
    audio_features = spectttra_predict(audio)
    lyrics_features = l2vec_single_train(llm2vec_model, lyrics)

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
        "prediction": "Fake" if prediction == 0 else "Real"
    }

if __name__ == "_main_":
    # Example usage (replace with real inputs, place song inside data/raw.)
    data = pd.read_csv("data/raw/predict_data.csv")

    result = []
    label = []
    for row in data.itertuples():
        label.append(row.label)
        result.append(predict_pipeline(row.song, row.lyrics))

    for i in range(len(result)):
        print(result[i])
        print(label[i]) 