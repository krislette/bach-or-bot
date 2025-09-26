from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import joblib
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def dataset_scaler(X: np.ndarray, Y: np.ndarray):
    """
    Method to scale both audio and lyric vectors using Z-Score.
    This allows us to have both vectors with a mean of 0, and ranges up and down based on the
    standard deviation without compromising the information it contains.

    This also saves the scalers through joblib, which will be loaded in the predict script.

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

    # Split train, test and validate into the audio and lyrical segments
    X_train_audio, X_train_lyrics = X_train[:, :384], X_train[:, 384:]
    X_val_audio,   X_val_lyrics   = X_val[:, :384],   X_val[:, 384:]
    X_test_audio,  X_test_lyrics  = X_test[:, :384],  X_test[:, 384:]

    # Apply scalers to have similar-ranged data for both audio and lyrics training values
    audio_scaler = StandardScaler().fit(X_train_audio)
    lyric_scaler = StandardScaler().fit(X_train_lyrics)

    # Transform all splits based on the trained scaler
    X_train_scaled = np.concatenate([audio_scaler.transform(X_train_audio),
                                 lyric_scaler.transform(X_train_lyrics)], axis=1)
    X_val_scaled   = np.concatenate([audio_scaler.transform(X_val_audio),
                                    lyric_scaler.transform(X_val_lyrics)], axis=1)
    X_test_scaled  = np.concatenate([audio_scaler.transform(X_test_audio),
                                    lyric_scaler.transform(X_test_lyrics)], axis=1)
    
    # Save the trained scalers for prediction
    joblib.dump(audio_scaler, "models/fusion/audio_scaler.pkl")
    joblib.dump(lyric_scaler, "models/fusion/lyric_scaler.pkl")

    data = {
        "train": (X_train_scaled, y_train),
        "val": (X_val_scaled, y_val),
        "test": (X_test_scaled, y_test),
    }

    return data