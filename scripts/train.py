from src.preprocessing.preprocessor import dataset_read, bulk_preprocessing
from src.spectttra.spectttra_trainer import spectttra_train
from src.llm2vectrain.model import load_llm2vec_model
from src.llm2vectrain.llm2vec_trainer import l2vec_train
from src.models.mlp import build_mlp, load_config

from src.utils.config_loader import DATASET_NPZ, PCA_MODEL
from src.utils.dataset import dataset_scaler, dataset_splitter
from sklearn.decomposition import IncrementalPCA

from pathlib import Path
import numpy as np
import logging
import joblib

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
        logger.info("Training dataset does not exist. Processing data...")
        # Get batches from dataset and return full Y labels
        batches, Y = dataset_read(batch_size=500)
        batch_count = 1

        # Instantiate LLM2Vec and PCA model
        llm2vec_model = load_llm2vec_model()

        # Preallocate spaces for both audio and lyric vectors to reduce memory overhead
        audio_vectors = np.zeros((len(Y), 384), dtype=np.float32)
        lyric_vectors = np.zeros((len(Y), 4096), dtype=np.float32)

        start_idx = 0
        for batch in batches:

            logger.info(f"Bulk Preprocessing - Batch {batch_count}.")
            audio, lyrics = bulk_preprocessing(batch, batch_count)
            batch_count += 1

            # Call the train methods for both SpecTTTra and LLM2Vec
            logger.info("Starting SpecTTTra feature extraction...")
            audio_features = spectttra_train(audio)

            logger.info("Starting LLM2Vec feature extraction...")
            lyrics_features = l2vec_train(llm2vec_model, lyrics)

            batch_size = audio_features.shape[0]

            # Store the results on preallocated spaces
            audio_vectors[start_idx:start_idx + batch_size, :] = audio_features
            lyric_vectors[start_idx:start_idx + batch_size, :] = lyrics_features
        
            # Delete stored instance for next batch to remove overhead
            del audio, lyrics, audio_features, lyrics_features

        # Run standard scaling on audio and lyrics separately
        logger.info("Running standard scaling for audio and lyrics...")
        audio_vectors, lyric_vectors = dataset_scaler(audio_vectors, lyric_vectors)

        # Run PCA per batch to reduce GPU overhead
        ipca = IncrementalPCA(n_components=256)
        batch_size = 1000  # Adjust depending on memory

        # Fit IPCA in batches
        for i in range(0, lyric_vectors.shape[0], batch_size):
            ipca.partial_fit(lyric_vectors[i:i + batch_size])

        # Transform in batches
        lyric_vectors_reduced = np.zeros((lyric_vectors.shape[0], 256), dtype=np.float32)
        for i in range(0, lyric_vectors.shape[0], batch_size):
            lyric_vectors_reduced[i:i + batch_size, :] = ipca.transform(lyric_vectors[i:i + batch_size])

        # Save IncrementalPCA model
        joblib.dump(ipca, "models/fusion/incremental_pca.pkl")
        lyric_vectors = lyric_vectors_reduced

        # Concatenate audio features and reduced lyrics features
        X = np.concatenate([audio_vectors, lyric_vectors], axis=1)
        logger.info(f"Audio and Lyrics Concatenated. Final features shape: {X.shape}")

        # Convert label list into np.array
        Y = np.array(Y)

        # Save both X and Y to an .npz file for easier loading
        logger.info("Saving dataset for future testing...")
        np.savez(DATASET_NPZ, X=X, Y=Y)

    # Do data splitting
    data = dataset_splitter(X, Y)

    logger.info("Starting MLP training...")
    train_mlp_model(data)


if __name__ == "__main__":
    train_pipeline()