import librosa
from scripts.explain import musiclime

# Load test audio and lyrics
audio_path = "data/external/sample_1.mp3"
lyrics_path = "data/external/sample_1.txt"

# Load audio
audio_data, sr = librosa.load(audio_path)

# Load lyrics
with open(lyrics_path, "r", encoding="utf-8") as f:
    lyrics_text = f.read()

print("Running MusicLIME explanation...")
result = musiclime(audio_data, lyrics_text)

print("\n=== EXPLANATION RESULTS ===")
print(
    f"Prediction: {result['prediction']['class_name']} ({result['prediction']['confidence']:.3f})"
)
print(f"Runtime: {result['summary']['runtime_seconds']:.2f}s")

print("\n=== TOP FEATURES (by absolute importance) ===")
for feature in result["explanations"]:
    print(
        f"Rank {feature['rank']}: {feature['modality']} | Weight: {feature['weight']:.4f} | Importance: {feature['importance']:.4f}"
    )
    print(f"  Feature: {feature['feature_text'][:80]}...")
    print()
