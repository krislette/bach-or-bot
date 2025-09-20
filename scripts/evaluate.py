
"""
Evaluation script for the Bach-or-Bot MLP classifier.

This script loads a trained model and evaluates it on test data.
"""

import argparse
import logging
import numpy as np
from pathlib import Path

from src.models.mlp import build_mlp, load_config
from src.utils.config_loader import DATASET_NPZ
from sklearn.model_selection import train_test_split

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def evaluate_model(model_path: str = "models/fusion/mlp_multimodal.pth"):
    """
    Evaluate a trained MLP model.
    
    Args:
        model_path: Path to the trained model
    """
    logger.info(f"Loading model from: {model_path}")
    
    # Load the dataset
    if not Path(DATASET_NPZ).exists():
        raise FileNotFoundError(f"Dataset not found at {DATASET_NPZ}. Run train.py first.")
    
    loaded_data = np.load(DATASET_NPZ)
    X = loaded_data["X"]
    Y = loaded_data["Y"]
    
    logger.info(f"Loaded dataset: {X.shape}, Labels: {len(Y)}")
    
    # Split data (same as training)
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42, stratify=Y
    )
    
    logger.info(f"Test set size: {X_test.shape}")
    
    # Load configuration
    config = load_config("config/model_config.yml")
    
    # Build model architecture (needed for loading weights)
    mlp_classifier = build_mlp(input_dim=X_test.shape[1], config=config)
    
    # Load trained model
    mlp_classifier.load_model(model_path)
    
    # Evaluate on test set
    logger.info("Evaluating model on test set...")
    test_results = mlp_classifier.evaluate(X_test, y_test)
    
    # Get predictions for detailed analysis
    probabilities, predictions = mlp_classifier.predict(X_test)
    
    logger.info("=== Evaluation Results ===")
    logger.info(f"Test Accuracy: {test_results['test_accuracy']:.2f}%")
    logger.info(f"Test Loss: {test_results['test_loss']:.4f}")
    
    # Additional statistics
    true_positives = np.sum((y_test == 1) & (predictions == 1))
    true_negatives = np.sum((y_test == 0) & (predictions == 0))
    false_positives = np.sum((y_test == 0) & (predictions == 1))
    false_negatives = np.sum((y_test == 1) & (predictions == 0))
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1-Score: {f1_score:.4f}")
    
    return test_results


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate Bach-or-Bot MLP classifier')
    parser.add_argument('--model', default='models/fusion/mlp_multimodal.pth',
                       help='Path to trained model')
    args = parser.parse_args()
    
    try:
        results = evaluate_model(args.model)
        logger.info("Evaluation completed successfully!")
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()