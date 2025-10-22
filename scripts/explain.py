import os
import numpy as np
from datetime import datetime
from src.musiclime.explainer import MusicLIMEExplainer
from src.musiclime.wrapper import MusicLIMEPredictor, AudioOnlyPredictor


def musiclime_multimodal(audio_data, lyrics_text):
    """
    Generate multimodal MusicLIME explanations for audio and lyrics.

    Parameters
    ----------
    audio_data : array-like
        Audio waveform data from librosa.load or similar
    lyrics_text : str
        String containing song lyrics

    Returns
    -------
    dict
        Structured explanation results containing prediction info, feature explanations,
        and processing metadata
    """
    start_time = datetime.now()

    # Get number of samples from environment variable, default to 1000
    num_samples = int(os.getenv("MUSICLIME_NUM_SAMPLES", "1000"))
    num_features = int(os.getenv("MUSICLIME_NUM_FEATURES", "10"))

    print(f"[MusicLIME] Using num_samples={num_samples}, num_features={num_features}")

    # Create musiclime instances
    explainer = MusicLIMEExplainer()
    predictor = MusicLIMEPredictor()

    # Then generate explanations
    explanation = explainer.explain_instance(
        audio=audio_data,
        lyrics=lyrics_text,
        predict_fn=predictor,
        num_samples=num_samples,
        labels=(1,),
    )

    # Get prediction info
    original_prediction = explanation.predictions[0]
    predicted_class = np.argmax(original_prediction)
    confidence = float(np.max(original_prediction))

    # Get top features (I also made this configurable to prevent rebuilding)
    top_features = explanation.get_explanation(label=1, num_features=num_features)

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
            "samples_generated": num_samples,
            "timestamp": start_time.isoformat(),
        },
    }


def musiclime_unimodal(audio_data, modality="audio"):
    """
    Generate unimodal MusicLIME explanations for single modality.

    Parameters
    ----------
    audio_data : array-like
        Audio waveform data from librosa.load or similar
    modality : str, default='audio'
        Explanation modality, currently only supports 'audio'

    Returns
    -------
    dict
        Structured explanation results containing prediction info, audio-only feature
        explanations, and processing metadata

    Raises
    ------
    ValueError
        If modality is not 'audio' (lyrics is not yet implemented)
    """
    if modality != "audio":
        raise ValueError(
            "Currently only 'audio' modality is supported for unimodal explanations"
        )

    start_time = datetime.now()

    # Get number of samples from environment variable, default to 1000
    num_samples = int(os.getenv("MUSICLIME_NUM_SAMPLES", "1000"))
    num_features = int(os.getenv("MUSICLIME_NUM_FEATURES", "10"))

    print(
        f"[MusicLIME] Using num_samples={num_samples}, num_features={num_features} (audio-only mode)"
    )

    # Create musiclime instances
    explainer = MusicLIMEExplainer(random_state=42)
    predictor = AudioOnlyPredictor()

    # Use empty lyrics for audio-only since they're ignored anyways
    dummy_lyrics = ""

    # Generate explanation
    explanation = explainer.explain_instance(
        audio=audio_data,
        lyrics=dummy_lyrics,
        predict_fn=predictor,
        num_samples=num_samples,
        labels=(1,),
        modality=modality,
    )

    # Get prediction info
    original_prediction = explanation.predictions[0]
    predicted_class = np.argmax(original_prediction)
    confidence = float(np.max(original_prediction))

    # Get top features
    top_features = explanation.get_explanation(label=1, num_features=num_features)

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
                "modality": item["type"],  # "audio" for all features
                "feature_text": item["feature"],
                "weight": float(item["weight"]),
                "importance": abs(float(item["weight"])),
            }
            for i, item in enumerate(top_features)
        ],
        "summary": {
            "total_features_analyzed": len(top_features),
            "audio_features_count": len(top_features),  # All features are audio
            "lyrics_features_count": 0,  # No lyrics features
            "runtime_seconds": runtime_seconds,
            "samples_generated": num_samples,
            "timestamp": start_time.isoformat(),
        },
    }
