import numpy as np
from datetime import datetime
from src.musiclime.explainer import MusicLIMEExplainer
from src.musiclime.wrapper import MusicLIMEPredictor


def musiclime(audio_data, lyrics_text):
    """
    MusicLIME wrapper for API usage.

    Args:
        audio_data: Audio array (from librosa.load or similar)
        lyrics_text: String containing lyrics

    Returns:
        dict: Structured explanation results
    """
    start_time = datetime.now()

    # Create musiclime instances
    explainer = MusicLIMEExplainer()
    predictor = MusicLIMEPredictor()

    # Generate explanations
    explanation = explainer.explain_instance(
        audio=audio_data,
        lyrics=lyrics_text,
        predict_fn=predictor,
        num_samples=1000,
        labels=(1,),
    )

    # Get prediction info
    original_prediction = explanation.predictions[0]
    predicted_class = np.argmax(original_prediction)
    confidence = float(np.max(original_prediction))

    # Get top 10 features
    top_features = explanation.get_explanation(label=1, num_features=10)

    # Calculate runtime
    end_time = datetime.now()
    runtime_seconds = (end_time - start_time).total_seconds()

    return {
        "prediction": {
            "class": int(predicted_class),
            "class_name": "Human-Composed" if predicted_class == 1 else "AI-Generated",
            "confidence": confidence,
            "probabilities": original_prediction.tolist(),
        },
        "explanations": [
            {
                "rank": i + 1,
                "modality": item["type"],
                "feature_text": item["feature"],
                "weight": float(item["weight"]),
                "importance": abs(float(item["weight"])),
            }
            for i, item in enumerate(top_features)
        ],
        "summary": {
            "total_features_analyzed": len(top_features),
            "audio_features_count": len(
                [f for f in top_features if f["type"] == "audio"]
            ),
            "lyrics_features_count": len(
                [f for f in top_features if f["type"] == "lyrics"]
            ),
            "runtime_seconds": runtime_seconds,
            "samples_generated": 1000,
            "timestamp": start_time.isoformat(),
        },
    }
