
from src.preprocessing.preprocessor import dataset_read, bulk_preprocessing
from src.spectttra.spectttra_trainer import spectttra_train
from src.llm2vectrain.model import load_llm2vec_model
from src.llm2vectrain.llm2vec_trainer import *
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

        all_audio_features = []
        all_lyrics_features = []

        for batch in batches:

            print(f"Bulk Preprocessing batch {batch_count}...")
            audio, lyrics = None, None
            audio, lyrics = bulk_preprocessing(batch, batch_count)

            audio_features = spectttra_train(audio)
            lyrics_features = l2vec_train(llm2vec_model, lyrics) # l2vec_train now only encodes

            all_audio_features.append(audio_features)
            all_lyrics_features.append(lyrics_features)

            print(f"Processed batch {batch_count} for feature collection. Audio shape: {audio_features.shape}, Lyrics shape: {lyrics_features.shape}")
            
            # Delete stored instance for next batch to remove overhead
            del audio, lyrics
            
            batch_count += 1

        # Code below here means that the batches are done.

        pca_trainer = SimplePCATrainer()

        ##Uncomment to check output
        #print(f"All_lyrics_features[0]: {all_lyrics_features[0]}, shape: {all_lyrics_features[0].shape}")
        #print(f"All_audio_features[0]: {all_audio_features[0]}, shape: {all_audio_features[0].shape}")

        for i in range(0, len(all_lyrics_features)):
            all_lyrics_features[i] = pca_trainer.process_batch(all_lyrics_features[i])

        pca_trainer.save_final("/content/drive/MyDrive/data/processed/pca_model.pkl") #Change path as needed

        ##Uncomment to check output (PCA)
        #print(f"Reduced All_lyrics_features[0]: {all_lyrics_features[0]}, shape: {all_lyrics_features[0].shape}")

        # Concatenate audio features and reduced lyrics features
        X = np.concatenate([all_audio_features, all_lyrics_features], axis=-1)
        X = X.reshape(-1, X.shape[-1])
        print(f"Final features shape: {X.shape}")

        

        # Convert label list into np.array
        Y = np.array(Y)

        # Save both X and Y to an .npz file for easier loading
        np.savez(DATASET_NPZ, X=X, Y=Y)
    
    # Run standard scaling on audio and lyrics separately
    data = dataset_scaler(X, Y)

    print("Starting MLP training...")
    train_mlp_model(data)

if __name__ == "__main__":
    train_pipeline()

