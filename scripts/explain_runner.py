import librosa
from scripts.explain import musiclime_multimodal, musiclime_unimodal


def explain_multimodal_runner(sample: str):
    # Load test audio and lyrics
    audio_path = f"data/external/{sample}.mp3"
    lyrics_path = f"data/external/{sample}.txt"

    # Load audio
    audio_data, sr = librosa.load(audio_path)

    # Load lyrics
    with open(lyrics_path, "r", encoding="utf-8") as f:
        lyrics_text = f.read()

    print("Running multimodal MusicLIME explanation...")
    result = musiclime_multimodal(audio_data, lyrics_text)

    print("\n=== MULTIMODAL EXPLANATION RESULTS ===")
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


def explain_unimodal_runner(sample: str):
    # Load test audio
    audio_path = f"data/external/{sample}.mp3"

    # Load audio
    audio_data, sr = librosa.load(audio_path)

    print("Running audio-only MusicLIME explanation...")
    result = musiclime_unimodal(audio_data, modality="audio")

    print("\n=== AUDIO-ONLY EXPLANATION RESULTS ===")
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


if __name__ == "__main__":
    sample = "sample"

    # Run multimodal explanation
    explain_multimodal_runner(sample)

    print("\n" + "=" * 60 + "\n")

    # Run audio-only explanation
    explain_unimodal_runner(sample)
