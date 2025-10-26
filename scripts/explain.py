import os
import numpy as np
from datetime import datetime
from src.musiclime.explainer import MusicLIMEExplainer
from src.musiclime.wrapper import MusicLIMEPredictor, AudioOnlyPredictor
from src.musiclime.print_utils import green_bold


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
    explainer = MusicLIMEExplainer(random_state=42)
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


def musiclime_combined(audio_data, lyrics_text):
    """
    Generate both multimodal and audio-only MusicLIME explanations efficiently.

    Performs source separation once and generates both explanation types
    to reduce total processing time by ~50% compared to separate calls.

    Parameters
    ----------
    audio_data : array-like
        Audio waveform data from librosa.load or similar
    lyrics_text : str
        String containing song lyrics

    Returns
    -------
    dict
        Combined results containing both multimodal and audio-only explanations
    """
    from src.musiclime.factorization import OpenUnmixFactorization
    from src.musiclime.text_utils import LineIndexedString

    start_time = datetime.now()

    # Get configuration
    num_samples = int(os.getenv("MUSICLIME_NUM_SAMPLES", "1000"))
    num_features = int(os.getenv("MUSICLIME_NUM_FEATURES", "10"))

    print(
        "[MusicLIME] Combined mode: generating both multimodal and audio-only explanations"
    )
    print(f"[MusicLIME] Using num_samples={num_samples}, num_features={num_features}")

    # Create factorizations once
    print("[MusicLIME] Creating factorizations once for both explanations...")
    factorization_start = datetime.now()

    audio_factorization = OpenUnmixFactorization(
        audio_data, temporal_segmentation_params=10
    )
    text_factorization = LineIndexedString(lyrics_text)

    factorization_time = (datetime.now() - factorization_start).total_seconds()
    print(
        green_bold(f"[MusicLIME] Factorization completed in {factorization_time:.2f}s")
    )

    # Create explainer and predictors
    explainer = MusicLIMEExplainer(random_state=42)
    multimodal_predictor = MusicLIMEPredictor()
    audio_predictor = AudioOnlyPredictor()

    # Generate multimodal explanation (reusing factorizations)
    print("[MusicLIME] Generating multimodal explanation...")
    multimodal_start = datetime.now()

    multimodal_explanation = explainer.explain_instance_with_factorization(
        audio_factorization,
        text_factorization,
        multimodal_predictor,
        num_samples=num_samples,
        labels=(1,),
        modality="both",
    )

    multimodal_time = (datetime.now() - multimodal_start).total_seconds()
    print(
        green_bold(
            f"[MusicLIME] Multimodal explanation completed in {multimodal_time:.2f}s"
        )
    )

    # Generate audio-only explanation (reusing the same factorization)
    print("[MusicLIME] Generating audio-only explanation (reusing factorizations)...")
    audio_start = datetime.now()

    audio_explanation = explainer.explain_instance_with_factorization(
        audio_factorization,
        text_factorization,
        audio_predictor,
        num_samples=num_samples,
        labels=(1,),
        modality="audio",
    )

    audio_time = (datetime.now() - audio_start).total_seconds()
    print(
        green_bold(f"[MusicLIME] Audio-only explanation completed in {audio_time:.2f}s")
    )

    # Process multimodal results
    multimodal_prediction = multimodal_explanation.predictions[0]
    multimodal_class = np.argmax(multimodal_prediction)
    multimodal_confidence = float(np.max(multimodal_prediction))
    multimodal_features = multimodal_explanation.get_explanation(
        label=1, num_features=num_features
    )

    # Process audio-only results
    audio_prediction = audio_explanation.predictions[0]
    audio_class = np.argmax(audio_prediction)
    audio_confidence = float(np.max(audio_prediction))
    audio_features = audio_explanation.get_explanation(
        label=1, num_features=num_features
    )

    # Calculate total runtime
    end_time = datetime.now()
    total_runtime = (end_time - start_time).total_seconds()

    print(green_bold("[MusicLIME] Combined explanation completed!"))
    print(f"[MusicLIME] Factorization: {factorization_time:.2f}s (done once)")
    print(f"[MusicLIME] Multimodal: {multimodal_time:.2f}s")
    print(f"[MusicLIME] Audio-only: {audio_time:.2f}s")
    print(f"[MusicLIME] Total: {total_runtime:.2f}s")

    return {
        "multimodal": {
            "prediction": {
                "class": int(multimodal_class),
                "class_name": (
                    "Human-Composed" if multimodal_class == 1 else "AI-Generated"
                ),
                "confidence": multimodal_confidence,
                "probabilities": multimodal_prediction.tolist(),
            },
            "explanations": [
                {
                    "rank": i + 1,
                    "modality": item["type"],
                    "feature_text": item["feature"],
                    "weight": float(item["weight"]),
                    "importance": abs(float(item["weight"])),
                }
                for i, item in enumerate(multimodal_features)
            ],
            "summary": {
                "total_features_analyzed": len(multimodal_features),
                "audio_features_count": len(
                    [f for f in multimodal_features if f["type"] == "audio"]
                ),
                "lyrics_features_count": len(
                    [f for f in multimodal_features if f["type"] == "lyrics"]
                ),
                "runtime_seconds": multimodal_time,
                "samples_generated": num_samples,
            },
        },
        "audio_only": {
            "prediction": {
                "class": int(audio_class),
                "class_name": "Human-Composed" if audio_class == 1 else "AI-Generated",
                "confidence": audio_confidence,
                "probabilities": audio_prediction.tolist(),
            },
            "explanations": [
                {
                    "rank": i + 1,
                    "modality": item["type"],
                    "feature_text": item["feature"],
                    "weight": float(item["weight"]),
                    "importance": abs(float(item["weight"])),
                }
                for i, item in enumerate(audio_features)
            ],
            "summary": {
                "total_features_analyzed": len(audio_features),
                "audio_features_count": len(audio_features),
                "lyrics_features_count": 0,
                "runtime_seconds": audio_time,
                "samples_generated": num_samples,
            },
        },
        "combined_summary": {
            "total_runtime_seconds": total_runtime,
            "factorization_time_seconds": factorization_time,
            "source_separation_reused": True,
            "timestamp": start_time.isoformat(),
        },
    }
