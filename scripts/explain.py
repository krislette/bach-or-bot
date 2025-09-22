from src.explainability.musiclime_wrapper import create_musiclime_wrapper
from src.models.mlp import load_config
from pathlib import Path


def explain():
    # Paths
    audio_path = Path("data/external/sample.mp3")
    lyrics_path = Path("data/external/sample.txt")

    # Read lyrics from file
    print("Reading lyrics...")
    lyrics_text = lyrics_path.read_text(encoding="utf-8")

    # Load configuration
    print("Loading config...")
    config = load_config()

    # Create wrapper
    print("Creating wrapper...")
    wrapper = create_musiclime_wrapper("models/mlp/mlp_best.pth", config)

    # Generate explanation for a sample
    # Note: Factorization type can be -> source_separation
    explanation = wrapper.explain_prediction(
        audio_path=str(audio_path),
        lyrics_text=lyrics_text,
        factorization_type="temporal",
        n_samples=1000,
    )

    # Print summary
    print(explanation.get_summary_text())

    # Save explanation
    explanation.save_to_json("explanation_results.json")


if __name__ == "__main__":
    explain()
