from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import joblib
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def dataset_splitter(X: np.ndarray, Y: np.ndarray):
    """
    Script that splits the X and Y values to train, test, and valid splits.

    Parameters
    ----------
    X : np.array
        Array of feature vectors
    Y : np.array
        Array of labels (real or fake)

    Returns
    -------
    data : dict{np.array}
        A dictionary of np.arrays, containing the train/test/val split.
    """

    logger.info(f"Dataset shape: {X.shape}, Labels: {len(Y)}")
    logger.info(f"Class distribution: {np.bincount(Y)}")

    # Split the data into train/val/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.1, random_state=42, stratify=Y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2222, random_state=42, stratify=y_train
    )
    
    logger.info(f"Train: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}")

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
    scaled_lyrics : np.array
        Array of scaled lyric features
    """

    # Apply scalers to have similar-ranged data for both audio and lyrics training values
    audio_scaler = StandardScaler().fit(audio)
    lyric_scaler = StandardScaler().fit(lyrics)

    scaled_audio = audio_scaler.transform(audio)
    scaled_lyrics = lyric_scaler.transform(lyrics)

    # Save the trained scalers for prediction
    joblib.dump(audio_scaler, "models/fusion/audio_scaler.pkl")
    joblib.dump(lyric_scaler, "models/fusion/lyric_scaler.pkl")

    return scaled_audio, scaled_lyrics


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
    scaled_lyrics : np.array
        Array of scaled lyric feature
    """

    # Apply scalers to the single inputs
    audio_scaler = joblib.load("models/fusion/audio_scaler.pkl")
    lyric_scaler = joblib.load("models/fusion/lyric_scaler.pkl")

    scaled_audio = audio_scaler.transform(audio)
    scaled_lyrics = lyric_scaler.transform(lyrics)

    return scaled_audio, scaled_lyrics