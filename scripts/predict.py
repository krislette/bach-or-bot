
from src.preprocessing.preprocessor import single_preprocessing
from src.spectttra.spectttra_trainer import spectttra_train
from src.llm2vectrain.model import load_llm2vec_model
from src.llm2vectrain.llm2vec_trainer import l2vec_train
from src.models.mlp import build_mlp, load_config
from pathlib import Path
from src.utils.config_loader import DATASET_NPZ

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
    llm2vec_model = load_llm2vec_model()

    # Preprocess both audio and lyrics
    audio, lyrics = single_preprocessing(audio, lyrics)

    # Call the train method for both models
    audio_features = spectttra_train(audio)
    lyrics_features = l2vec_train(llm2vec_model, [lyrics])

    # Concatenate the vectors of audio_features + lyrics_features
    results = np.concatenate([audio_features[0], lyrics_features[0]])

    # ---- Load MLP Classifier ----
    config = load_config("config/model_config.yml")
    classifier = build_mlp(input_dim=results.shape[0], config=config)

    # Load trained weights (make sure this path matches where you saved your model)
    model_path = "models/mlp.pth"
    classifier.load_model(model_path)
    classifier.eval()

    # Run prediction
    probability, prediction, label = classifier.predict_single(results)

    return {
        "label": int(prediction),
        "prediction": "Fake" if prediction == 0 else "Real"
    }

if __name__ == "__main__":
    # Example usage (replace with real inputs)
    audio = None  # your audio object
    lyrics = "Some lyrics text here"
    print(predict_pipeline(audio, lyrics))
