from src.preprocessing.preprocessor import dataset_read, bulk_preprocessing
from src.spectttra.spectttra_trainer import spectttra_train
from src.models.mlp import build_mlp, load_config
from pathlib import Path
from src.utils.config_loader import DATASET_NPZ, RAW_DATASET_NPZ
from src.utils.dataset import scale

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
        mlp_classifier.load_model("models/mlp/mlp_best.pth")
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

    The train pipeline saves the raw and scaled dataset in .npz format.
    """

    AUDIO_LENGTH = 384
    BATCH_SIZE = 200

    dataset_path = Path(RAW_DATASET_NPZ)

    if dataset_path.exists():
        print("Training dataset already exists. Loading file...")

        loaded_data = np.load(RAW_DATASET_NPZ)

        data = {
            "train": (loaded_data["X_train"], loaded_data["y_train"]),
            "test":  (loaded_data["X_test"], loaded_data["y_test"]),
            "val":   (loaded_data["X_val"], loaded_data["y_val"]),
        }

    else:
        print("Training dataset does not exist. Processing data...")

        # Get batches from dataset and lengths
        splits, split_lengths = dataset_read(batch_size=BATCH_SIZE)
        batch_count = 1

        # Preallocate arrays
        X_train = np.zeros((split_lengths[0], AUDIO_LENGTH), dtype=np.float32)
        X_test  = np.zeros((split_lengths[1], AUDIO_LENGTH), dtype=np.float32)
        X_val   = np.zeros((split_lengths[2], AUDIO_LENGTH), dtype=np.float32)

        y_train = np.zeros(split_lengths[0], dtype=np.int32)
        y_test  = np.zeros(split_lengths[1], dtype=np.int32)
        y_val   = np.zeros(split_lengths[2], dtype=np.int32)

        X_splits = [X_train, X_test, X_val]
        y_splits = [y_train, y_test, y_val]

        # Iterate over train=0, test=1, val=2
        for split_idx, split in enumerate(splits):
            start_idx = 0
            for batch in split:
                if len(batch) == 0:
                    continue  # skip empty batch safely

                print(f"Bulk Preprocessing batch {batch_count}...")
                audio, lyrics = bulk_preprocessing(batch, batch_count)
                batch_count += 1

                batch_labels = batch['target'].values

                # Extract audio features
                logger.info("Starting SpecTTTra feature extraction...")
                audio_features = spectttra_train(audio)

                bsz = audio_features.shape[0]
                X_splits[split_idx][start_idx:start_idx + bsz, :] = audio_features
                y_splits[split_idx][start_idx:start_idx + bsz] = batch_labels
                start_idx += bsz

        # Save raw (unscaled) dataset
        np.savez(
            RAW_DATASET_NPZ,
            X_train=X_train, y_train=y_train,
            X_val=X_val,     y_val=y_val,
            X_test=X_test,   y_test=y_test,
        )

        # Run scaling
        logger.info("Running standard scaling...")
        data = {
            "train": (X_train, y_train),
            "val":   (X_val, y_val),
            "test":  (X_test, y_test),
        }
    data = scale(data)

    # Save scaled dataset
    X_train, y_train = data["train"]
    X_val, y_val     = data["val"]
    X_test, y_test   = data["test"]

    logger.info("Saving scaled dataset...")
    np.savez(
        DATASET_NPZ,
        X_train=X_train, y_train=y_train,
        X_val=X_val,     y_val=y_val,
        X_test=X_test,   y_test=y_test,
    )

    print("Starting MLP training...")
    train_mlp_model(data)


if __name__ == "__main__":
    train_pipeline()