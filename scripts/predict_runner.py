import librosa
from scripts.predict import predict_multimodal, predict_audio


def predict_multimodal_runner(sample: str):
    # Load test audio and lyrics
    audio_path = f"data/external/{sample}.mp3"
    lyrics_path = f"data/external/{sample}.txt"

    # Load audio
    audio_data, sr = librosa.load(audio_path)

    # Load lyrics
    with open(lyrics_path, "r", encoding="utf-8") as f:
        lyrics_text = f.read()

    print("Running prediction pipeline...")
    prediction = predict_multimodal(audio_data, lyrics_text)

    print(f"\n=== PREDICTION RESULT ===")
    print(f"Prediction: {prediction}")


def predict_audio_runner(sample: str):
    # Load test audio and lyrics
    audio_path = f"data/raw/{sample}.mp3"

    # Load audio
    audio_data, sr = librosa.load(audio_path)

    print("Running prediction pipeline...")
    prediction = predict_audio(audio_data)

    print(f"\n=== PREDICTION RESULT ===")
    print(f"Prediction: {prediction}")


if __name__ == "__main__":
    sample = "fake_sunshine"

    predict_audio_runner(sample)