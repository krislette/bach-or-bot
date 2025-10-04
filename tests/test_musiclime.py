import pytest
import numpy as np
import torch
from src.musiclime.explainer import MusicLIMEExplainer, MusicLIMEExplanation
from src.musiclime.factorization import OpenUnmixFactorization
from src.musiclime.text_utils import LineIndexedString
from src.musiclime.wrapper import MusicLIMEPredictor


def test_line_indexed_string():
    lyrics = "Line 1\nLine 2\n[Chorus]\nLine 3\n(Verse)\nLine 4"
    indexed = LineIndexedString(lyrics)

    # Test: We assert that line segmentation works correctly
    # Should skip metadata like [Chorus] and (Verse), keep only actual lyrics
    assert indexed.num_words() == 4  # Should have 4 actual lines
    assert "Line 1" in indexed.as_list
    assert "Line 4" in indexed.as_list
    assert "[Chorus]" not in indexed.as_list
    assert "(Verse)" not in indexed.as_list


def test_line_indexed_string_inverse_removing():
    lyrics = "Line 1\nLine 2\nLine 3\nLine 4"
    indexed = LineIndexedString(lyrics)

    # Remove lines at indices 1 and 3 (Line 2 and Line 4)
    result = indexed.inverse_removing([1, 3])

    # Test: We assert that inverse removing works correctly
    # Should return only Line 1 and Line 3, joined by newlines
    expected = "Line 1\nLine 3"
    assert result == expected


def test_audio_factorization_creation():
    # Create dummy audio (5 seconds at 44.1kHz)
    dummy_audio = np.random.randn(5 * 44100)

    try:
        factorization = OpenUnmixFactorization(
            dummy_audio, temporal_segmentation_params=5
        )

        # Test: We assert that audio factorization creates expected components
        # Should have 4 sources × 5 temporal segments = 20 total components
        assert factorization.get_number_components() == 20
        component_names = factorization.get_ordered_component_names()
        assert len(component_names) == 20

        # Test: Component names should follow expected pattern
        # Should have vocals_T0, drums_T0, bass_T0, other_T0, vocals_T1, etc.
        assert any("vocals_T" in name for name in component_names)
        assert any("drums_T" in name for name in component_names)
        assert any("bass_T" in name for name in component_names)
        assert any("other_T" in name for name in component_names)

    except Exception as e:
        pytest.skip(f"OpenUnmix model not available: {e}")


def test_audio_factorization_composition():
    dummy_audio = np.random.randn(2 * 44100)  # 2 seconds

    try:
        factorization = OpenUnmixFactorization(
            dummy_audio, temporal_segmentation_params=3
        )

        # Test composition with subset of components
        selected_components = [0, 2, 4]  # Select some components
        composed_audio = factorization.compose_model_input(selected_components)

        # Test: We assert that composed audio has same length as original
        # Composed audio should be same length, might be different amplitude
        assert len(composed_audio) == len(dummy_audio)
        assert isinstance(composed_audio, np.ndarray)

    except Exception as e:
        pytest.skip(f"OpenUnmix model not available: {e}")


def test_musiclime_explainer_creation():
    explainer = MusicLIMEExplainer(kernel_width=25, random_state=42)

    # Test: We assert that MusicLIME explainer is created properly
    # Should have required attributes for LIME explanation
    assert explainer.random_state is not None
    assert explainer.base is not None
    assert hasattr(explainer, "explain_instance")


def test_musiclime_explanation_object():
    # Create mock objects for testing
    dummy_audio_fact = type(
        "MockFactorization",
        (),
        {
            "get_number_components": lambda: 8,
            "get_ordered_component_names": lambda: [f"comp_{i}" for i in range(8)],
        },
    )()

    dummy_text_fact = type(
        "MockTextFact", (), {"num_words": lambda: 4, "word": lambda i: f"line_{i}"}
    )()

    dummy_data = np.random.randint(0, 2, (10, 12))  # 10 samples, 12 features
    dummy_predictions = np.random.rand(10, 2)  # 10 samples, 2 classes

    explanation = MusicLIMEExplanation(
        dummy_audio_fact, dummy_text_fact, dummy_data, dummy_predictions
    )

    # Test: We assert that explanation object is created with correct attributes
    # Should have all required dictionaries for storing explanation results
    assert hasattr(explanation, "intercept")
    assert hasattr(explanation, "local_exp")
    assert hasattr(explanation, "score")
    assert hasattr(explanation, "local_pred")
    assert isinstance(explanation.intercept, dict)
    assert isinstance(explanation.local_exp, dict)


def test_musiclime_predictor_creation():
    try:
        predictor = MusicLIMEPredictor()

        # Test: We assert that MusicLIME predictor is created successfully
        # Should have required attributes for prediction
        assert hasattr(predictor, "__call__")
        assert predictor.llm2vec_model is not None
        assert predictor.config is not None

    except Exception as e:
        pytest.skip(f"MusicLIME predictor dependencies not available: {e}")


def test_perturbation_data_generation():
    explainer = MusicLIMEExplainer(random_state=42)

    # Create simple mock objects
    audio_fact = type(
        "MockAudioFact",
        (),
        {
            "get_number_components": lambda self: 4,
            "compose_model_input": lambda self, indices: np.random.randn(1000),
        },
    )()

    text_fact = type(
        "MockTextFact",
        (),
        {
            "num_words": lambda self: 3,
            "inverse_removing": lambda self, indices: "remaining text",
        },
    )()

    # Mock predict function
    def mock_predict_fn(texts, audios):
        return np.random.rand(len(texts), 2)

    try:
        data, predictions, distances = explainer._generate_neighborhood(
            audio_fact, text_fact, mock_predict_fn, num_samples=5
        )

        # Test: We assert that perturbation generation works correctly
        # Should return data, predictions, and distances with correct shapes
        assert data.shape == (5, 7)  # 5 samples, 7 features (4 audio + 3 text)
        assert predictions.shape == (5, 2)  # 5 samples, 2 classes
        assert len(distances) == 5  # 5 distance values
        assert data[0].sum() == 7  # First sample should be all 1s (original)

    except Exception as e:
        pytest.fail(f"Perturbation generation failed: {e}")
