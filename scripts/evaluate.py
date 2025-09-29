"""
MLP Model Evaluation Script for AI vs Human Music Detection
==========================================================

This script evaluates the performance of the trained MLP classifier on test data.
It gives a complete performance report showing how well the model can distinguish
between AI-generated and human-composed music.

What this script does:
- Loads our saved/trained MLP model
- Tests it on held-out test data (music the model has never seen)
- Calculates accuracy, precision, recall, and F1-score
- Reports confusion statistics (true positives, true negatives, false positives, false negatives)
- Displays sample predictions with probabilities for transparency

Quick Start:
---------------------------
# Basic evaluation with default model path
python evaluate.py

# Evaluate a specific model
python evaluate.py --model "models/fusion/mlp_multimodal.pth"

# From code
from evaluate import evaluate_model
results = evaluate_model("models/fusion/mlp_multimodal.pth")

Performance Metrics Explained:
------------------------------
- Accuracy: Overall correctness (how many songs classified correctly)
- Precision: Of songs predicted as human, how many actually were human
- Recall: Of all human songs, how many did we correctly identify  
- F1-Score: Balance between precision and recall (harmonic mean)
- Confusion stats: 
    TP = Human songs correctly identified  
    TN = AI songs correctly identified  
    FP = AI songs incorrectly labeled as human  
    FN = Human songs incorrectly labeled as AI  

Expected Output:
----------------
Loading model from: models/fusion/mlp_multimodal.pth
Loaded dataset: (50000, 684), Labels: 50000
Test set size: (10000, 684)
Evaluating model on test set...

Sample predictions:
True: 1, Pred: 1, Prob: 0.8234  # Correctly identified human song
True: 0, Pred: 0, Prob: 0.1456  # Correctly identified AI song
True: 1, Pred: 0, Prob: 0.4123  # Missed a human song (false negative)

=== Evaluation Results ===
Test Accuracy: 87.54%
Test Loss: 0.3412
Precision: 0.8832
Recall: 0.8654
F1-Score: 0.8742
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
    logger.info(f"Loading model from: {model_path}")
    
    # Check if dataset exists
    if not Path(DATASET_NPZ).exists():
        raise FileNotFoundError(f"Dataset not found at {DATASET_NPZ}. Run train.py first.")
    
    # Load the full dataset
    loaded_data = np.load(DATASET_NPZ)
    X = loaded_data["X"]
    Y = loaded_data["Y"]
    
    logger.info(f"Loaded dataset: {X.shape}, Labels: {len(Y)}")
    
    # Split data (same as training)
    from src.utils.dataset import dataset_scaler
    data = dataset_scaler(X, Y)
    X_test, y_test = data["test"]
    
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

    # Show a few sample predictions
    for i in range(10): 
        print(f"True: {y_test[i]}, Pred: {predictions[i]}, Prob: {probabilities[i]:.4f} "
              f"(Probability of predicted class)")
    
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
    
    # Include all metrics in return dict
    return {
        "test_accuracy": test_results["test_accuracy"],
        "test_loss": test_results["test_loss"],
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "true_positives": int(true_positives),
        "true_negatives": int(true_negatives),
        "false_positives": int(false_positives),
        "false_negatives": int(false_negatives)
    }


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
