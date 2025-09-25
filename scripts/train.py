
from src.preprocessing.preprocessor import dataset_read, bulk_preprocessing
from src.spectttra.spectttra_trainer import spectttra_train
from src.llm2vectrain.model import load_llm2vec_model
from src.llm2vectrain.llm2vec_trainer import l2vec_train
from src.models.mlp import build_mlp, load_config
from pathlib import Path
from src.utils.config_loader import DATASET_NPZ
from sklearn.model_selection import train_test_split

import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train_mlp_model(X: np.ndarray, Y: np.ndarray):
    """
    Train the MLP model with extracted features.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        Y: Labels (n_samples,)
    """
    logger.info("Starting MLP training...")
    logger.info(f"Dataset shape: {X.shape}, Labels: {len(Y)}")
    logger.info(f"Class distribution: {np.bincount(Y)}")
    
    # Load MLP configuration
    config = load_config("config/model_config.yml")
    
    # Split the data into train/val/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.1, random_state=42, stratify=Y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2222, random_state=42, stratify=y_train
    )
    
    logger.info(f"Train: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}")
    
    # Build and train MLP
    mlp_classifier = build_mlp(input_dim=X_train.shape[1], config=config)
    
    # Show model summary
    mlp_classifier.get_model_summary()
    
    # Train the model
    history = mlp_classifier.train(X_train, y_train, X_val, y_val)
    
    # Load best model and evaluate on test set
    try:
        mlp_classifier.load_model("models/fusion/mlp_best.pth")
        logger.info("Loaded best model for final evaluation")
    except FileNotFoundError:
        logger.warning("Best model not found, using current model")
    
    # Final evaluation
    test_results = mlp_classifier.evaluate(X_test, y_test)
    
    # Save final model
    mlp_classifier.save_model("models/fusion/mlp_multimodal.pth")
    
    logger.info("MLP training completed successfully!")
    logger.info(f"Final test accuracy: {test_results['test_accuracy']:.2f}%")
    
    return mlp_classifier

def train_pipeline():
    """
    Training script which includes preprocessing, feature extraction, and training the MLP model.

    The train pipeline saves the train dataset in an .npz format.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """

    # Instantiate X and Y vectors
    X, Y = None, None

    dataset_path = Path(DATASET_NPZ)

    if dataset_path.exists():
        print("Training dataset already exists. Loading file...")

        loaded_data = np.load(DATASET_NPZ)
        X = loaded_data["X"]
        Y = loaded_data["Y"]
    else:
        print("Training dataset does not exist. Processing data...")
        # Get batches from dataset and return full Y labels
        batches, Y = dataset_read()
        batch_count = 1

        # Instantiate LLM2Vec Model
        llm2vec_model = load_llm2vec_model()

        # Preallocate space for the whole concatenated sequence (50,000 samples)
        X = np.zeros((len(Y), 684), dtype=np.float32)

        start_idx = 0
        for batch in batches:
            audio, lyrics = None, None  # Gets rid of previous values consuming current memory
            
            print(f"Bulk Preprocessing batch {batch_count}...")
            audio, lyrics = bulk_preprocessing(batch, batch_count)
            batch_count += 1

            # Call the train method for SpecTTTra
            print(f"\nStarting SpecTTTra feature extraction...")
            audio_features = spectttra_train(audio)

            print(f"\nStarting LLM2Vec feature extractor...")
            lyrics_features = l2vec_train(llm2vec_model, lyrics)

            # Concatenate the vectors of audio_features + lyrics_features
            results = np.concatenate([audio_features, lyrics_features], axis=1)
            batch_size = results.shape[0]

            X[start_idx:start_idx + batch_size, :] = results
            start_idx += batch_size

        # Convert label list into np.array
        Y = np.array(Y)

        # Save both X and Y to an .npz file for easier loading
        np.savez(DATASET_NPZ, X=X, Y=Y)
    
    print("Starting MLP training...")
    train_mlp_model(X, Y)

if __name__ == "__main__":
    train_pipeline()

