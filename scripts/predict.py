from src.preprocessing.preprocessor import single_preprocessing, single_audio_preprocessing
from src.spectttra.spectttra_trainer import spectttra_predict
from src.llm2vectrain.model import load_llm2vec_model
from src.llm2vectrain.llm2vec_trainer import l2vec_single_train, load_pca_model
from src.models.mlp import build_mlp, load_config
from src.utils.dataset import instance_scaler, audio_instance_scaler

import numpy as np
import pandas as pd


def predict_multimodal(audio_file, lyrics):
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
    # 1.) Instantiate LLM2Vec Model
    llm2vec_model = load_llm2vec_model()

    # 2.) Preprocess both audio and lyrics
    audio, lyrics = single_preprocessing(audio_file, lyrics)

    # 3.) Call the train method for both models
    audio_features = spectttra_predict(audio)
    audio_features = audio_features.reshape(1, -1)
    lyrics_features = l2vec_single_train(llm2vec_model, lyrics)

    # 4.) Scale the vectors using Z-Score
    audio_features, lyrics_features = instance_scaler(audio_features, lyrics_features)

    # 5.) Reduce the lyrics using saved PCA model
    reduced_lyrics = load_pca_model(lyrics_features)

    # 6.) Concatenate the vectors of audio_features + lyrics_features
    results = np.concatenate([audio_features, reduced_lyrics], axis=1)

    # ---- Load MLP Classifier ----
    config = load_config("config/model_config.yml")
    classifier = build_mlp(input_dim=results.shape[1], config=config)

    # 7.) Load trained weights
    model_path = "models/mlp/mlp_best.pth"
    classifier.load_model(model_path)
    classifier.model.eval()

    # 8.) Run prediction
    confidence, prediction, label, probability = classifier.predict_single(
        results.flatten()
    )

    return {
        "confidence": confidence,
        "prediction": prediction,
        "label": label,
        "probability": probability,
    }


def predict_unimodal(audio_file):
    """
    Predict script of AUDIO only which includes preprocessing, feature extraction, and
    training the MLP model for a single data sample.

    Parameters
    ----------
    audio : audio_object
        Audio object file

    Returns
    -------
    prediction : str
        A string result of the prediction

    label : int
        A numerical representation of the prediction
    """

    # 1.) Preprocess the audio
    audio = single_audio_preprocessing(audio_file)

    # 2.) Call the inference method from SpecTTTra
    audio_features = spectttra_predict(audio)
    audio_features = audio_features.reshape(1, -1)

    # 4.) Scale the vector using Z-Score
    audio_features = audio_instance_scaler(audio_features)

    # 5.) Load MLP Classifier
    config = load_config("config/model_config.yml")
    classifier = build_mlp(input_dim=audio_features.shape[1], config=config)

    # 6.) Load trained weights
    model_path = "models/spectttra/mlp_best.pth"
    classifier.load_model(model_path)
    classifier.model.eval()

    # 8.) Run prediction
    confidence, prediction, label, probability = classifier.predict_single(
        audio_features.flatten()
    )

    return {
        "confidence": confidence,
        "prediction": prediction,
        "label": label,
        "probability": probability,
    }


if __name__ == "__main__":
    # Example usage (replace with real inputs, place song inside data/raw.)
    data = pd.read_csv("data/raw/predict_data_final.csv")

    result = []
    label = []
    for row in data.itertuples():
        prediction = predict_multimodal(row.song, row.lyrics)
        result.append(
            {
                "song": row.song,
                "label": row.label,
                "predicted_label": prediction["label"],
                "probability": prediction["probability"],
            }
        )

    for r in result:
        print(f"Song: {r['song']}")
        print(f"Actual Label: {r['label']}")
        print(f"Predicted: {r['predicted_label']}")
        print(f"Confidence: {r['probability']: .8f}%")
        print("-" * 50)
