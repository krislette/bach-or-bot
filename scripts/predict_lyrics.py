from src.llm2vectrain.llm2vec_trainer import l2vec_single_train
from src.llm2vectrain.model import load_llm2vec_model
from src.models.mlp import build_mlp, load_config
from src.utils.config_loader import LYRICS_SCALER, PCA_MODEL
from src.preprocessing.lyrics_preprocessor import LyricsPreprocessor

import pandas as pd
import joblib
import numpy as np


def predict_pipeline(lyric):
    """
    Predict script for a single lyric sample. Returns probabilities,
    numeric prediction, and string labels consistent with CSV mapping.
    
    Args:
        lyric: raw text string of lyrics
    
    Returns:
        dict with:
            - probability: confidence of being human (0.0 to 1.0)
            - numeric_prediction: 0 = AI/fake, 1 = Human/real
            - label: "AI-Generated" or "Human-Composed"
            - prediction: "Fake" or "Real" (matches CSV naming)
    """

    l2v = load_llm2vec_model()

    # --- Preprocess lyrics ---
    lyric_preprocessor = LyricsPreprocessor()
    processed_lyric = lyric_preprocessor(lyrics=lyric)

    # --- Feature extraction ---
    lyric_features = l2vec_single_train(l2v, processed_lyric)

    # --- Load scaler and scale features ---
    lyric_scaler = joblib.load(LYRICS_SCALER)
    lyric_features = lyric_scaler.transform(lyric_features)

    pca_scaler = joblib.load(PCA_MODEL)
    lyric_features = pca_scaler.transform(lyric_features)

    # Ensure 1D (new mlp.py expects 1D input for predict_single)
    lyric_features = np.array(lyric_features).flatten()

    # --- Load trained classifier ---
    config = load_config("config/model_config.yml")
    classifier = build_mlp(input_dim=len(lyric_features), config=config)
    model_path = "models/mlp/lyrics_mlp_multimodal.pth"
    classifier.load_model(model_path)
    classifier.model.eval()

    # --- Run prediction (new mlp.py returns just probability float) ---
    probability = classifier.predict_single(lyric_features)

    # Convert float probability into full outputs
    numeric_prediction = int(probability > 0.5)
    label = "Human-Composed" if numeric_prediction == 1 else "AI-Generated"

    return {
        "probability": float(probability),
        "label": label,
        "prediction": "Real" if numeric_prediction == 1 else "Fake",
        "numeric_prediction": numeric_prediction
    }


if __name__ == "__main__":
    # Example usage: predict on all songs listed in predict_data.csv
    data = pd.read_csv("data/raw/predict_data_final.csv")

    results = []
    for row in data.itertuples():
        results.append(predict_pipeline(row.lyrics))

    for i, res in enumerate(results):
        print(res)
        print(data.iloc[i]["song"])
