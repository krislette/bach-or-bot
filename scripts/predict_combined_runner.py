import librosa
from scripts.predict import predict_combined


def predict_combined_runner(sample: str):
    # Load test audio and lyrics
    audio_path = f"data/external/{sample}.mp3"
    lyrics_path = f"data/external/{sample}.txt"

    # Load audio
    audio_data, sr = librosa.load(audio_path)

    # Load lyrics
    with open(lyrics_path, "r", encoding="utf-8") as f:
        lyrics_text = f.read()

    print("Running combined prediction (optimized)...")
    result = predict_combined(audio_data, lyrics_text)

    # Display results
    print(f"\n{'='*50}")
    print("=== MULTIMODAL PREDICTION ===")
    print(f"{'='*50}")
    mm = result["multimodal"]
    print(f"Prediction: {mm['prediction']}")
    print(f"Label: {mm['label']}")
    print(f"Confidence: {mm['confidence']:.4f}")
    print(f"Probability: {mm['probability']:.4f}")

    print(f"\n{'='*50}")
    print("=== AUDIO-ONLY PREDICTION ===")
    print(f"{'='*50}")
    au = result["audio_only"]
    print(f"Prediction: {au['prediction']}")
    print(f"Label: {au['label']}")
    print(f"Confidence: {au['confidence']:.4f}")
    print(f"Probability: {au['probability']:.4f}")

    print(f"\n{'='*50}")
    print("=== PERFORMANCE SUMMARY ===")
    print(f"{'='*50}")
    perf = result["performance"]
    print(f"Multimodal prediction: {perf['multimodal_time_seconds']:.2f}s")
    print(f"Audio-only prediction: {perf['audio_only_time_seconds']:.2f}s")
    print(f"Total time: {perf['total_time_seconds']:.2f}s")

    print(f"\n{'='*50}")
    print("=== COMPARISON ===")
    print(f"{'='*50}")
    print(f"Multimodal: {mm['prediction']} ({mm['probability']:.4f})")
    print(f"Audio-only: {au['prediction']} ({au['probability']:.4f})")

    prob_diff = abs(mm["probability"] - au["probability"])
    print(f"Probability difference: {prob_diff:.4f}")

    if mm["prediction"] == au["prediction"]:
        print("Both modalities agree on the prediction")
    else:
        print("Modalities disagree on the prediction")


if __name__ == "__main__":
    sample = "sample"

    predict_combined_runner(sample)
