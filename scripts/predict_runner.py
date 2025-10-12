import librosa
from scripts.predict import predict_pipeline

# Load test audio and lyrics
audio_path = "data/external/sample_1.mp3"
lyrics_path = "data/external/sample_1.txt"

# Load audio
audio_data, sr = librosa.load(audio_path)

# Load lyrics
with open(lyrics_path, "r", encoding="utf-8") as f:
    lyrics_text = f.read()

print("Running prediction pipeline...")
prediction = predict_pipeline(audio_data, lyrics_text)

print(f"\n=== PREDICTION RESULT ===")
print(f"Prediction: {prediction}")
