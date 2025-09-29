import librosa
from pathlib import Path

from src.explainability.explainer import MusicLIMEExplainer
from src.explainability.wrapper import MusicLIMEPredictor

explainer = MusicLIMEExplainer()
predictor = MusicLIMEPredictor()

audio_path = Path("data/external/sample.mp3")
lyrics_path = Path("data/external/sample.txt")

y, sr = librosa.load(audio_path)
lyrics_text = lyrics_path.read_text(encoding="utf-8")

# Fix: pass actual audio array, not shape
explanation = explainer.explain_instance(
    audio=y,
    lyrics=lyrics_text,
    predict_fn=predictor,
    num_samples=100,  # Start small for testing
    labels=(1,),
)

results = explanation.get_explanation(label=1, num_features=10)
for item in results:
    print(f"{item['type']}: {item['feature']} (weight: {item['weight']:.3f})")
