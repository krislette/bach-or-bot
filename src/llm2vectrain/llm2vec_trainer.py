from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path

import numpy as np
import pickle
import torch

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