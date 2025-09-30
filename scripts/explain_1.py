import librosa
from pathlib import Path

from src.explainability.explainer import MusicLIMEExplainer
from src.explainability.wrapper import MusicLIMEPredictor


def explain():
    # Create musiclime-related instances
    explainer = MusicLIMEExplainer()
    predictor = MusicLIMEPredictor()

    # Set the path for audio and lyrics [these are samples only - song is Silver Spring]
    audio_path = Path("data/external/sample.mp3")
    lyrics_path = Path("data/external/sample.txt")

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
    for item in results:
        print(f"{item['type']}: {item['feature']} (weight: {item['weight']:.3f})")


if __name__ == "__main__":
    explain()
