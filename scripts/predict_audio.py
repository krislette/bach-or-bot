from src.preprocessing.preprocessor import single_preprocessing
from src.spectttra.spectttra_trainer import spectttra_predict
from src.models.mlp import build_mlp, load_config

import pandas as pd
import joblib

def predict_pipeline(audio):
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
    audio, _ = single_preprocessing(audio, "asd")

    # Call the train method for both models
    audio_features = spectttra_predict(audio)

    audio_scaler = joblib.load("models/fusion/audio_scaler.pkl")

    audio_features = audio_scaler.transform([audio_features])

    # ---- Load MLP Classifier ----
    config = load_config("config/model_config.yml")
    classifier = build_mlp(input_dim=audio_features.shape[1], config=config)

    # Load trained weights (make sure this path matches where you saved your model)
    model_path = "models/mlp/mlp_multimodal.pth"
    classifier.load_model(model_path)
    classifier.model.eval()

    # Run prediction
    probability, prediction, label = classifier.predict_single(audio_features)

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
        result.append(predict_pipeline(row.song))

    for i in range(len(result)):
        print(result[i])
        print(label[i]) 