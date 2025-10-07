from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from src.utils.config_loader import AUDIO_SCALER, LYRICS_SCALER
from sklearn.decomposition import IncrementalPCA
from src.utils.config_loader import PCA_MODEL

import joblib
import numpy as np
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def dataset_splitter(X: np.ndarray, Y: np.ndarray, ids: np.ndarray = None):
    """
    Splits X, Y (and optional ids) into train/val/test sets.
    Saves metadata CSVs for each split if ids are provided.

    Parameters
    ----------
    X : np.array
        Feature vectors
    Y : np.array
        Labels
    ids : np.array, optional
        Identifiers (filenames or row indices)
    save_metadata : bool
        Whether to save split metadata CSVs
    outdir : str
        Directory to save metadata CSVs

    Returns
    -------
    data : dict
        A dictionary of np.arrays: {train, val, test}
        Each value is a tuple (X_split, y_split, ids_split if provided)
    """

    logger.info(f"Dataset shape: {X.shape}, Labels: {len(Y)}")
    logger.info(f"Class distribution: {np.bincount(Y)}")

    # First split: train vs test
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, ids, test_size=0.1, random_state=42, stratify=Y
    )

    # Second split: train vs val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2222, random_state=42, stratify=y_train
    )

    logger.info(f"Train: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}")

    data = {
        "train": (X_train, y_train),
        "val":   (X_val, y_val),
        "test":  (X_test, y_test),
    }

    return data


def scale_pca(data : dict):
    """
    Script that scales the splits, and applies PCA to the lyrics vector.

    Parameters
    ----------
    data : dictionary
        Dictionary containing the splits

    Returns
    -------
    data : dict{np.array}
        A dictionary of np.arrays, containing the train/test/val split.
    """

    # Destructure the dictionary to get data split
    X_train, y_train = data["train"]
    X_val, y_val     = data["val"]
    X_test, y_test   = data["test"]

    # Segment the concatenated embedding to audio and lyrics
    X_train_audio, X_train = X_train[:, :384], X_train[:, 384:]
    X_test_audio, X_test = X_test[:, :384], X_test[:, 384:]
    X_val_audio, X_val = X_val[:, :384], X_val[:, 384:]

    # Fit the scalers into the train data, return scalers for fitting of test and validation
    audio_scaler, lyric_scaler = dataset_scaler(X_train_audio, X_train)

    # Transform the rest of the splits using the scalers
    X_train_audio = audio_scaler.transform(X_train_audio)
    X_test_audio = audio_scaler.transform(X_test_audio)
    X_val_audio = audio_scaler.transform(X_val_audio)

    X_train = lyric_scaler.transform(X_train)
    X_test = lyric_scaler.transform(X_test)
    X_val = lyric_scaler.transform(X_val)

    # Fit PCA on TRAINING lyrics only
    ipca = IncrementalPCA(n_components=512)
    batch_size = 1000

    for i in range(0, X_train.shape[0], batch_size):
        ipca.partial_fit(X_train[i:i + batch_size])

    # Transform in batches
    X_train = ipca.transform(X_train)
    X_test = ipca.transform(X_test)
    X_val = ipca.transform(X_val)

    # Concatenate them back to their original form, but scaled
    X_train = np.concatenate([X_train_audio, X_train], axis=1)
    X_test = np.concatenate([X_test_audio, X_test], axis=1)
    X_val = np.concatenate([X_val_audio, X_val], axis=1)

    joblib.dump(ipca, PCA_MODEL)

    data = {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test),
    }

    return data


def scale_pca_lyrics(data : dict):
    """
    Script that scales the splits, and applies PCA to the lyrics vector.

    Parameters
    ----------
    data : dictionary
        Dictionary containing the splits

    Returns
    -------
    data : dict{np.array}
        A dictionary of np.arrays, containing the train/test/val split.
    """

    # Destructure the dictionary to get data split
    X_train, y_train = data["train"]
    X_val, y_val     = data["val"]
    X_test, y_test   = data["test"]

    lyric_scaler = StandardScaler().fit(X_train)
    joblib.dump(lyric_scaler, LYRICS_SCALER)

    X_train = lyric_scaler.transform(X_train)
    X_test = lyric_scaler.transform(X_test)
    X_val = lyric_scaler.transform(X_val)

    # Fit PCA on TRAINING lyrics only
    ipca = IncrementalPCA(n_components=512)
    batch_size = 1000

    for i in range(0, X_train.shape[0], batch_size):
        ipca.partial_fit(X_train[i:i + batch_size])

    # Transform in batches
    X_train = ipca.transform(X_train)
    X_test = ipca.transform(X_test)
    X_val = ipca.transform(X_val)

    joblib.dump(ipca, PCA_MODEL)

    data = {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test),
    }

    return data


def scale(data : dict):
    """
    Script that scales the splits, and applies PCA to the lyrics vector.

    Parameters
    ----------
    data : dictionary
        Dictionary containing the splits

    Returns
    -------
    data : dict{np.array}
        A dictionary of np.arrays, containing the train/test/val split.
    """

    # Destructure the dictionary to get data split
    X_train, y_train = data["train"]
    X_val, y_val     = data["val"]
    X_test, y_test   = data["test"]

    audio_scaler = StandardScaler(with_mean=False).fit(X_train)
    joblib.dump(audio_scaler, AUDIO_SCALER)

    # Transform the rest of the splits using the scalers
    X_train = audio_scaler.transform(X_train)
    X_test = audio_scaler.transform(X_test)
    X_val = audio_scaler.transform(X_val)

    data = {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test),
    }

    return data

def dataset_scaler(audio: np.ndarray, lyrics: np.ndarray):
    """
    Method to scale both audio and lyric vectors using Z-Score.
    This allows us to have both vectors with a mean of 0, and ranges up and down based on the
    standard deviation without compromising the information it contains.

    This also saves the scalers through joblib, which will be loaded in the predict script.

    Parameters
    ----------
    audio : np.array
        Array of audio features
    lyrics : np.array
        Array of lyric features

    Returns
    -------
    scaled_audio : np.array
        Array of scaled audio features
    scaleds : np.array
        Array of scaled lyric features
    """

    # Apply scalers to have similar-ranged data for both audio and lyrics training values
    audio_scaler = StandardScaler().fit(audio)
    lyric_scaler = StandardScaler().fit(lyrics)

    # Save the trained scalers for prediction
    joblib.dump(audio_scaler, AUDIO_SCALER)
    joblib.dump(lyric_scaler, LYRICS_SCALER)

    return audio_scaler, lyric_scaler


def instance_scaler(audio: np.ndarray, lyrics: np.ndarray):
    """
    Method to scale the single input audio and lyrics

    Parameters
    ----------
    audio : np.array
        Instance of an audio feature
    lyrics : np.array
        Instance of a lyric feature

    Returns
    -------
    scaled_audio : np.array
        Array of scaled audio feature
    scaleds : np.array
        Array of scaled lyric feature
    """

    # Apply scalers to the single inputs
    audio_scaler = joblib.load("models/fusion/audio_scaler.pkl")
    lyric_scaler = joblib.load("models/fusion/lyric_scaler.pkl")

    scaled_audio = audio_scaler.transform([audio])
    scaleds = lyric_scaler.transform(lyrics)

    return scaled_audio, scaleds