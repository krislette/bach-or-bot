from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path

import numpy as np
import pickle
import torch
import os
import joblib

# Initialize PCA and StandardScaler globally for training
_pca_trainer = None

class SimplePCATrainer:
    """
    A simple PCA trainer that uses IncrementalPCA to fit data in batches.
    It saves checkpoints every 5 batches and can save the final model.
    
    Args:
        None

    Returns:
        None

    Attributes:
        pca: The IncrementalPCA model.
        scaler: StandardScaler for normalizing data.
        fitted: Boolean indicating if the model has been initialized.
        batch_count_pca: Counter for the number of batches processed.

    Methods:
        process_batch(vectors): Processes a batch of vectors, fits the PCA model incrementally.
        save_final(model_path): Saves the final PCA model to the specified path.
    """

    # Initialize the trainer
    def __init__(self):
        self.pca = None
        self.scaler = StandardScaler()
        self.fitted = False
        self.batch_count_pca = 0

    def _determine_optimal_components(self, vectors):
        """
        Determine the optimal number of PCA components to retain 95% variance.
        
        Args:
            vectors: The input data to analyze.
        Returns:
            n_components: The optimal number of components.
        """
        temp_pca = IncrementalPCA()
        temp_pca.fit(vectors)
        cumsum_var = np.cumsum(temp_pca.explained_variance_ratio_)
        n_comp_95 = np.argmax(cumsum_var >= 0.95) + 1
        return min(n_comp_95, vectors.shape[1] // 2)

    def process_batch(self, vectors):
        """
        Process a batch of vectors, fitting the PCA model incrementally.
        
        Args:
            vectors: The input data batch to process.
        Returns:
            reduced_vectors: The PCA-transformed data.

        Note: This method saves a checkpoint every 5 batches.
        """
        if not self.fitted:
            # First batch - initialize everything
            n_components = self._determine_optimal_components(vectors)
            self.pca = IncrementalPCA(n_components=n_components, batch_size=1000)
            self.scaler.fit(vectors)
            self.fitted = True
            print(f"Initialized PCA with {n_components} components")

        # Process batch
        vectors_scaled = self.scaler.transform(vectors)
        self.pca.partial_fit(vectors_scaled)
        reduced_vectors = self.pca.transform(vectors_scaled)

        self.batch_count_pca += 1

        # Save checkpoint every 5 batches
        if self.batch_count_pca % 5 == 0:
            os.makedirs("pca_checkpoints", exist_ok=True)
            with open(f"pca_checkpoints/checkpoint_batch_{self.batch_count_pca}.pkl", 'wb') as f:
                pickle.dump({'pca': self.pca, 'scaler': self.scaler}, f)
            print(f"Saved checkpoint at batch {self.batch_count_pca}")

        print(f"Processed batch {self.batch_count_pca}, shape: {vectors.shape} -> {reduced_vectors.shape}")
        return reduced_vectors

    def save_final(self, model_path):
        """
        Save the final PCA model to the specified path.

        Args:
            model_path: The file path to save the PCA model.

        Returns:
            None
        
        Note: Change the model path as needed in the data_config.yml file.
        """
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump({'pca': self.pca, 'scaler': self.scaler}, f)
        print(f"Final model saved to {model_path}. Total variance explained: {np.sum(self.pca.explained_variance_ratio_):.4f}")

## For Single Input
def load_pca_model(vectors, model_path="models/fusion/pca.pkl"):
    """
    Load a pre-trained PCA model and transform the input vectors.

    Args:
        vectors: The input data to transform.
        model_path: The file path of the pre-trained PCA model.

    Returns:
        output: The PCA-transformed data.

    Note: Change the model path as needed in the data_config.yml file (or set the path file as shown above). Can be used for the main program.
    """
    model_path = Path(model_path)
    pca = joblib.load(model_path)
    return pca.transform(vectors)

def l2vec_single_train(l2v, lyrics):
    """
    Encode a single lyric string using the provided LLM2Vec model.
    
    Args:
        l2v: The LLM2Vec model for encoding lyrics.
        lyrics: A single lyric string to encode.
    
    Returns:
        vectors: The vector representation of the lyrics.

    """
    vectors = l2v.encode([lyrics]).detach().cpu().numpy()
    return vectors

# For Batch Processing
def l2vec_train(l2v, lyrics_list):
    """
    Encode a list of lyric strings using the provided LLM2Vec model.

    Args:
        l2v: The LLM2Vec model for encoding lyrics.
        lyrics_list: A list of lyric strings to encode.
    Returns:
        vectors: The encoded vector representations of the lyrics.

    Note: This function only encodes the lyrics and does not apply PCA reduction. The PCA reduction can be applied separately in the train.py module.
    """
    with torch.no_grad():
        vectors = l2v.encode(lyrics_list)  # lyrics_list: list of strings
    return vectors