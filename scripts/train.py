from src.preprocessing.preprocessor import dataset_read, bulk_preprocessing
from src.spectttra.spectttra_trainer import spectttra_train
from src.llm2vectrain.model import load_llm2vec_model
from src.llm2vectrain.llm2vec_trainer import l2vec_train
from src.models.mlp import build_mlp, load_config

from src.utils.config_loader import DATASET_NPZ
from src.utils.dataset import dataset_scaler

from pathlib import Path
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train_mlp_model(data : dict):
    """
    Train the MLP model with extracted features.
    
    Parameters
    ----------
        data : dict{np.array}
            A dictionary of np.arrays, containing the train/test/val split.
    """
    logger.info("Starting MLP training...")
    
    # Load MLP configuration
    config = load_config("config/model_config.yml")

    # Destructure the dictionary to get data split
    X_train, y_train = data["train"]
    X_val, y_val     = data["val"]
    X_test, y_test   = data["test"]
    
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
    mlp_classifier.save_model("models/mlp/mlp_multimodal.pth")
    
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
        logger.info("Training dataset already exists. Loading file...")

        loaded_data = np.load(DATASET_NPZ)
        X = loaded_data["X"]
        Y = loaded_data["Y"]
    else:
        print("Training dataset does not exist. Processing data...")
        # Get batches from dataset and return full Y labels
        batches, Y = dataset_read(batch_size=500)
        batch_count = 1

        # Instantiate LLM2Vec Model
        llm2vec_model = load_llm2vec_model()

        # Preallocate space for the whole concatenated sequence (50,000 samples)
        X = np.zeros((len(Y), 684), dtype=np.float32)

        start_idx = 0
        for batch in batches:

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

            # Delete stored instance for next batch to remove overhead
            del audio, lyrics, audio_features

        # Convert label list into np.array
        Y = np.array(Y)

        # Save both X and Y to an .npz file for easier loading
        np.savez(DATASET_NPZ, X=X, Y=Y)

    # drop_index = [823, 2717, 538, 3230, 5297, 3510, 1025, 2460, 4157, 539]
    # X = np.delete(X, drop_index, axis=0)
    # Y = np.delete(Y, drop_index, axis=0)
    
    # Run standard scaling on audio and lyrics separately
    data = dataset_scaler(X, Y)

    print("Starting MLP training...")
    train_mlp_model(data)

if __name__ == "__main__":
    train_pipeline()