from src.preprocessing.preprocessor import (
    single_preprocessing,
    single_audio_preprocessing,
)
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
    model_path = "models/mlp/mlp_best_multimodal.pth"
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
    model_path = "models/mlp/mlp_best_unimodal.pth"
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


def predict_combined(audio_file, lyrics):
    """
    Generate both multimodal and audio-only predictions efficiently.

    Follows the exact same logic as separate functions but reuses audio features.

    Parameters
    ----------
    audio_file : audio_object
        Audio object file
    lyrics : str
        Lyric string

    Returns
    -------
    dict
        Combined results containing both multimodal and audio-only predictions
    """
    import time

    start_time = time.time()

    # Load config once
    config = load_config("config/model_config.yml")

    # [1] Multimdoal prediction
    print("[Predict] Running multimodal prediction...")
    multimodal_start = time.time()

    # 1.) Load LLM2Vec Model
    llm2vec_model = load_llm2vec_model()

    # 2.) Preprocess both audio and lyrics
    audio_mm, lyrics_mm = single_preprocessing(audio_file, lyrics)

    # 3.) Extract features
    audio_features_mm = spectttra_predict(audio_mm)
    audio_features_mm = audio_features_mm.reshape(1, -1)
    lyrics_features = l2vec_single_train(llm2vec_model, lyrics_mm)

    # 4.) Scale the vectors using Z-Score
    audio_features_mm_scaled, lyrics_features_scaled = instance_scaler(
        audio_features_mm, lyrics_features
    )

    # 5.) Reduce the lyrics using saved PCA model
    reduced_lyrics = load_pca_model(lyrics_features_scaled)

    # 6.) Concatenate the vectors
    multimodal_features = np.concatenate(
        [audio_features_mm_scaled, reduced_lyrics], axis=1
    )

    # Load MLP Classifier
    multimodal_classifier = build_mlp(
        input_dim=multimodal_features.shape[1], config=config
    )
    multimodal_classifier.load_model("models/mlp/mlp_best_multimodal.pth")
    multimodal_classifier.model.eval()

    # Run prediction
    mm_confidence, mm_prediction, mm_label, mm_probability = (
        multimodal_classifier.predict_single(multimodal_features.flatten())
    )

    multimodal_time = time.time() - multimodal_start
    print(f"[Predict] Multimodal prediction completed in {multimodal_time:.2f}s")

    # [2] Unimodal prediction (audio-only)
    print("[Predict] Running audio-only prediction...")
    audio_only_start = time.time()

    # 1.) Preprocess the audio
    audio_au = single_audio_preprocessing(audio_file)

    # 2.) Extract audio features
    audio_features_au = spectttra_predict(audio_au)
    audio_features_au = audio_features_au.reshape(1, -1)

    # 3.) Scale the vector using Z-Score
    audio_features_au_scaled = audio_instance_scaler(audio_features_au)

    # Load MLP Classifier
    audio_classifier = build_mlp(
        input_dim=audio_features_au_scaled.shape[1], config=config
    )
    audio_classifier.load_model("models/mlp/mlp_best_unimodal.pth")
    audio_classifier.model.eval()

    # Run prediction
    au_confidence, au_prediction, au_label, au_probability = (
        audio_classifier.predict_single(audio_features_au_scaled.flatten())
    )

    audio_only_time = time.time() - audio_only_start
    print(f"[Predict] Audio-only prediction completed in {audio_only_time:.2f}s")

    # Summary
    total_time = time.time() - start_time

    print("\n[Predict] Combined prediction completed!")
    print(f"[Predict] Multimodal: {multimodal_time:.2f}s")
    print(f"[Predict] Audio-only: {audio_only_time:.2f}s")
    print(f"[Predict] Total: {total_time:.2f}s")

    return {
        "multimodal": {
            "confidence": mm_confidence,
            "prediction": mm_prediction,
            "label": mm_label,
            "probability": mm_probability,
        },
        "audio_only": {
            "confidence": au_confidence,
            "prediction": au_prediction,
            "label": au_label,
            "probability": au_probability,
        },
        "performance": {
            "total_time_seconds": total_time,
            "multimodal_time_seconds": multimodal_time,
            "audio_only_time_seconds": audio_only_time,
        },
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
