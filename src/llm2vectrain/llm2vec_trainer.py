from pathlib import Path
from src.utils.config_loader import PCA_MODEL

import joblib
import torch

## For Single Input
def load_pca_model(vectors, model_path=PCA_MODEL):
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