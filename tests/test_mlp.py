import pytest
import numpy as np
import torch
from src.models.mlp import build_mlp, load_config, MLPClassifier


def test_mlp_model_creation():
    config = load_config("config/model_config.yml")
    classifier = build_mlp(input_dim=640, config=config)

    # Test: We assert that MLP classifier is created properly
    # It should have the correct input dimension and be an instance of MLPClassifier
    assert isinstance(classifier, MLPClassifier)
    assert classifier.model is not None


def test_mlp_prediction_single():
    config = load_config("config/model_config.yml")
    classifier = build_mlp(input_dim=640, config=config)

    # Create dummy features that match expected input dimension
    dummy_features = np.random.randn(640)

    # Test: We assert that single prediction works and returns expected format
    # Probability should be between 0 and 1, prediction should be 0 or 1, label should be string
    try:
        probability, prediction, label = classifier.predict_single(dummy_features)
        assert 0 <= probability <= 1
        assert prediction in [0, 1]
        assert label in ["AI-Generated", "Human-Composed"]
    except ValueError as e:
        # Test: If model isn't trained, it should raise a specific error message
        # This is expected behavior when model hasn't been trained yet
        assert "Model must be trained" in str(e)


def test_mlp_prediction_batch():
    config = load_config("config/model_config.yml")
    classifier = build_mlp(input_dim=640, config=config)

    # Create batch of dummy features
    batch_features = np.random.randn(5, 640)

    try:
        probabilities, predictions = classifier.predict(batch_features)

        # Test: We assert that batch prediction returns correct shapes and value ranges
        # Probabilities should be array of length 5, predictions should be 0s and 1s
        assert probabilities.shape == (5,)
        assert predictions.shape == (5,)
        assert all(0 <= p <= 1 for p in probabilities)
        assert all(pred in [0, 1] for pred in predictions)
    except ValueError as e:
        # Test: Same as single prediction, untrained model should raise error
        assert "Model must be trained" in str(e)


def test_mlp_model_summary():
    config = load_config("config/model_config.yml")
    classifier = build_mlp(input_dim=640, config=config)

    # Test: We assert that model summary doesn't crash and model has parameters
    # This tests that the model architecture is properly constructed
    try:
        classifier.get_model_summary()
        total_params = sum(p.numel() for p in classifier.model.parameters())
        assert total_params > 0
    except Exception as e:
        pytest.fail(f"Model summary failed: {e}")


def test_config_loading():
    config = load_config("config/model_config.yml")

    # Test: We assert that config loads properly and contains expected keys
    assert "hidden_layers" in config
    assert "dropout" in config
    assert isinstance(config["hidden_layers"], list)
    assert len(config["hidden_layers"]) > 0
