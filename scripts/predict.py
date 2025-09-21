
from src.preprocessing.preprocessor import single_preprocessing
from src.spectttra.spectttra_trainer import spectttra_train
from src.llm2vectrain.model import load_llm2vec_model
from src.llm2vectrain.llm2vec_trainer import l2vec_train
from pathlib import Path
from src.utils.config_loader import DATASET_NPZ

import numpy as np

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
    lyrics_features = l2vec_train(llm2vec_model, lyrics)

    # Concatenate the vectors of audio_features + lyrics_features
    results = np.concatenate([audio_features, lyrics_features], axis=1)

    # TODO: Call MLP predict script
    # model_predict(results)




if __name__ == "__main__":
    predict_pipeline()