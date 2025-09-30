import librosa
from pathlib import Path

from src.explainability.explainer import MusicLIMEExplainer
from src.explainability.wrapper import MusicLIMEPredictor


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

    # Print explanations
    results = explanation.get_explanation(label=1, num_features=10)
    print("\n" + "=" * 80)
    print("[MusicLIME] Top 10 most important features for the prediction")
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
