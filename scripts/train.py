
from src.preprocessing.preprocessor import dataset_read, bulk_preprocessing
from src.spectttra.spectttra_trainer import spectttra_train
from src.llm2vectrain.model import load_llm2vec_model
from src.llm2vectrain.llm2vec_trainer import l2vec_train
from pathlib import Path
from src.utils.config_loader import DATASET_NPZ

import numpy as np

def train_pipeline():

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

        # Preallocate space for the whole concatenated sequence (20,000 samples)
        X = np.zeros((len(Y), 4480), dtype=np.float32)

        start_idx = 0
        for batch in batches:
            audio, lyrics = None, None  # Gets rid of previous values consuming current memory
            audio, lyrics = bulk_preprocessing(batch, batch_count)
            batch_count += 1

            # Call the train method for both models
            audio_features = spectttra_train(audio)
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

    # TODO: Call MLP training script


if __name__ == "__main__":
    train_pipeline()

