from datetime import datetime
import librosa
import numpy as np

from pathlib import Path
from src.musiclime.explainer import MusicLIMEExplainer
from src.musiclime.wrapper import MusicLIMEPredictor


def explain():
    # Create musiclime-related instances
    explainer = MusicLIMEExplainer()
    predictor = MusicLIMEPredictor()

    # Set the path for audio and lyrics [these are samples only - song is Silver Spring]
    audio_path = Path("data/external/sample_2.mp3")
    lyrics_path = Path("data/external/sample_2.txt")

    # Load the audio as an object + load the lyrics as string
    y, sr = librosa.load(audio_path)
    lyrics_text = lyrics_path.read_text(encoding="utf-8")

    # Generate explanations using musiclime
    explanation = explainer.explain_instance(
        audio=y,
        lyrics=lyrics_text,
        predict_fn=predictor,
        num_samples=5,
        labels=(1,),
    )

    # Get original prediction (first sample is always the orig meaning unperturbed)
    original_prediction = explanation.predictions[0]
    predicted_class = np.argmax(original_prediction)
    confidence = original_prediction[predicted_class]

    # Create song info from the prediction
    song_info = {
        "filename": "sample.mp3",
        "duration": f"{len(y)/44100:.1f}s",
        "original_prediction": {
            "class": "Human-Composed" if predicted_class == 1 else "AI-Generated",
            "confidence": float(confidence),
            "raw_probabilities": {
                "AI": float(original_prediction[0]),
                "Human": float(original_prediction[1]),
            },
        },
    }

    # Save with prediction data
    explanation.save_to_json(
        filepath=f"musiclime_explanation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        song_info=song_info,
        num_features=10,
    )

    # Print explanations
    results = explanation.get_explanation(label=1, num_features=10)
    print("\n" + "=" * 80)
    print(
        f"[MusicLIME] Top 10 most important features for {"Human-Composed" if predicted_class == 1 else "AI-Generated"} prediction"
    )
    print("=" * 80)

    for i, item in enumerate(results, 1):
        print(
            f"#{i:2d} | {item['type']:6s} | {item['feature'][:50]:50s} | weight: {item['weight']:+.3f}"
        )

    print("=" * 80)
    print(f"[MusicLIME] Total features analyzed: {len(results)}")
    print("[MusicLIME] Higher absolute weights = more important for the prediction")


if __name__ == "__main__":
    explain()
