from datetime import datetime
import librosa
import numpy as np

from pathlib import Path
from src.musiclime.explainer import MusicLIMEExplainer
from src.musiclime.wrapper import MusicLIMEPredictor
from src.musiclime.print_utils import green_bold


def explain():
    # Start timing and time stamp to record how long the entire explanation thingy is
    start_time = datetime.now()
    print(
        green_bold(
            f"[MusicLIME] Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}"
        )
    )

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
        num_samples=1000,
        labels=(1,),
    )

    # Get original prediction (first sample is always the orig meaning unperturbed)
    original_prediction = explanation.predictions[0]
    predicted_class = np.argmax(original_prediction)

    # Print explanations
    results = explanation.get_explanation(label=1, num_features=10)
    print("\n" + "=" * 80)
    print(
        f"[MusicLIME] Top 10 most important features for {"Human-Composed" if predicted_class == 1 else "AI-Generated"} prediction"
    )
    print("=" * 80)

    for i, item in enumerate(results, 1):
        print(
            f"#{i:2d} | {item['type']:6s} | {item['feature'][:50]:50s} | weight: {item['weight']:+.6f}"
        )

    print("=" * 80)
    print(f"[MusicLIME] Total features analyzed: {len(results)}")
    print("[MusicLIME] Higher absolute weights = more important for the prediction")

    # End timing and timestamp
    end_time = datetime.now()
    total_duration = end_time - start_time
    total_minutes = total_duration.total_seconds() / 60
    print(f"\n[MusicLIME] Finished at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(
        green_bold(
            f"[MusicLIME] Total execution time: {total_minutes:.2f} minutes ({total_duration.total_seconds():.1f} seconds)"
        )
    )


if __name__ == "__main__":
    explain()
