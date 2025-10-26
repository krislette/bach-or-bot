import librosa
from scripts.explain import musiclime_combined


def explain_combined_runner(sample: str):
    # Load test audio and lyrics
    audio_path = f"data/external/{sample}.mp3"
    lyrics_path = f"data/external/{sample}.txt"

    # Load audio
    audio_data, sr = librosa.load(audio_path)

    # Load lyrics
    with open(lyrics_path, "r", encoding="utf-8") as f:
        lyrics_text = f.read()

    print("Running combined MusicLIME explanation (optimized)...")
    result = musiclime_combined(audio_data, lyrics_text)

    # Display multimodal results
    print(f"\n{'='*60}")
    print("=== MULTIMODAL EXPLANATION RESULTS ===")
    print(f"{'='*60}")
    multimodal = result["multimodal"]
    print(
        f"Prediction: {multimodal['prediction']['class_name']} ({multimodal['prediction']['confidence']:.3f})"
    )
    print(f"Runtime: {multimodal['summary']['runtime_seconds']:.2f}s")

    print("\n=== TOP MULTIMODAL FEATURES ===")
    for feature in multimodal["explanations"]:
        print(
            f"Rank {feature['rank']}: {feature['modality']} | Weight: {feature['weight']:.4f} | Importance: {feature['importance']:.4f}"
        )
        print(f"  Feature: {feature['feature_text'][:80]}...")
        print()

    # Display audio-only results
    print(f"\n{'='*60}")
    print("=== AUDIO-ONLY EXPLANATION RESULTS ===")
    print(f"{'='*60}")
    audio_only = result["audio_only"]
    print(
        f"Prediction: {audio_only['prediction']['class_name']} ({audio_only['prediction']['confidence']:.3f})"
    )
    print(f"Runtime: {audio_only['summary']['runtime_seconds']:.2f}s")

    print("\n=== TOP AUDIO-ONLY FEATURES ===")
    for feature in audio_only["explanations"]:
        print(
            f"Rank {feature['rank']}: {feature['modality']} | Weight: {feature['weight']:.4f} | Importance: {feature['importance']:.4f}"
        )
        print(f"  Feature: {feature['feature_text'][:80]}...")
        print()

    # Display performance summary
    print(f"\n{'='*60}")
    print("=== PERFORMANCE SUMMARY ===")
    print(f"{'='*60}")
    summary = result["combined_summary"]
    print(
        f"Factorization time: {summary['factorization_time_seconds']:.2f}s (done once)"
    )
    print(f"Multimodal explanation: {multimodal['summary']['runtime_seconds']:.2f}s")
    print(f"Audio-only explanation: {audio_only['summary']['runtime_seconds']:.2f}s")
    print(f"Total runtime: {summary['total_runtime_seconds']:.2f}s")
    print(f"Source separation reused: {summary['source_separation_reused']}")

    # Comparison
    print("\n=== PREDICTION COMPARISON ===")
    print(
        f"Multimodal: {multimodal['prediction']['class_name']} ({multimodal['prediction']['confidence']:.3f})"
    )
    print(
        f"Audio-only: {audio_only['prediction']['class_name']} ({audio_only['prediction']['confidence']:.3f})"
    )

    if multimodal["prediction"]["class"] == audio_only["prediction"]["class"]:
        print("Both modalities agree on the prediction")
    else:
        print("Modalities disagree on the prediction")

    confidence_diff = abs(
        multimodal["prediction"]["confidence"] - audio_only["prediction"]["confidence"]
    )
    print(f"Confidence difference: {confidence_diff:.3f}")


if __name__ == "__main__":
    sample = "sample"

    explain_combined_runner(sample)
