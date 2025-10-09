import pandas as pd
import numpy as np
import math

from src.preprocessing.audio_preprocessor import AudioPreprocessor
from src.preprocessing.lyrics_preprocessor import LyricsPreprocessor
from src.utils.config_loader import DATASET_CSV


def bulk_preprocessing(batch: pd.DataFrame, batch_count: int):
    """
    Applies audio and lyrics preprocessing to a training batch

    Parameters
    ----------
    batch : pd.dataframe
        Dataframe containing the batch data.

    batch_count : int
        Batch count value.

    Returns
    -------
    audio_list : list
        List of loaded audio in float form.
    
    lyric_list : list
        List of loaded lyrics in string form.
    """

    audio_preprocessor = AudioPreprocessor(script="train")
    lyric_preprocessor = LyricsPreprocessor()

    audio_list, lyric_list = [], []
    count, batch_length = 1, len(batch)

    print(f"Preprocessing training data with length {batch_length}\n")

    for row in batch.itertuples():
        print(f"Batch {batch_count}     -    {count}/{batch_length}")

        # Preprocess song and append to audio list
        processed_song = audio_preprocessor(file=row.directory, skip_time=row.skip_time, train=True)
        audio_list.append(processed_song)

        # Preprocess lyric and append to lyric list
        processed_lyric = lyric_preprocessor(lyrics=row.lyrics)
        lyric_list.append(processed_lyric)

        count += 1

    return audio_list, lyric_list


def bulk_preprocessing_lyrics(batch: pd.DataFrame, batch_count: int):
    """
    Applies lyrics preprocessing to a training batch

    Parameters
    ----------
    batch : pd.dataframe
        Dataframe containing the batch data.

    batch_count : int
        Batch count value.

    Returns
    -------
    lyric_list : list
        List of loaded lyrics in string form.
    """

    lyric_preprocessor = LyricsPreprocessor()

    lyric_list = []
    count, batch_length = 1, len(batch)

    print(f"Preprocessing training data with length {batch_length}\n")

    for row in batch.itertuples():
        print(f"Batch {batch_count}     -    {count}/{batch_length}")

        # Preprocess lyric and append to lyric list
        processed_lyric = lyric_preprocessor(lyrics=row.lyrics)
        lyric_list.append(processed_lyric)

        count += 1

    return lyric_list


def single_preprocessing(audio, lyric: str):
    """
    Preprocesses a single record of audio and lyric data

    Parameters
    ----------
    audio : audio_object
        Audio object file
    
    lyric : string
        Lyric string

    Returns
    -------
    processed_song : tensor
        Tensor version of the audio
    
    processed_lyric : string
        Lyric string
    """
    # Instantiate preprocessor classes
    audio_preprocessor = AudioPreprocessor(script="predict")
    lyric_preprocessor = LyricsPreprocessor()

    # Preprocess both song and lyrics
    processed_song = audio_preprocessor(file=audio)
    processed_lyric = lyric_preprocessor(lyrics=lyric)

    return processed_song, processed_lyric


def dataset_read(batch_size=20):
    """
    Reads the main dataset, splits it into the train/test/valid split, and computes
    optimal number of samples per batch.

    Parameters
    ----------
    batch_size : int
        Number of data per batch

    Returns
    -------
    split: list[splits]
        A collection of the three splits

    split_lengths : list[int]
        List of the split lengths
    """
    dataset = pd.read_csv(DATASET_CSV)

    train = dataset[dataset["split"] == "train"]
    test = dataset[dataset["split"] == "test"]
    val = dataset[dataset["split"] == "valid"]

    # Find the minimum split size (ignoring empty splits)
    min_split_size = min([len(train), len(test), len(val)])
    # Clamp batch_size so it never exceeds the smallest split
    effective_batch_size = min(batch_size, min_split_size if min_split_size > 0 else batch_size)

    def make_splits(df, batch_size):
        if len(df) == 0:
            return []
        n_splits = math.ceil(len(df) / batch_size)
        return np.array_split(df, n_splits)

    train_splits = make_splits(train, effective_batch_size)
    test_splits = make_splits(test, effective_batch_size)
    val_splits = make_splits(val, effective_batch_size)

    splits = [train_splits, test_splits, val_splits]
    split_lengths = [len(train), len(test), len(val)]

    return splits, split_lengths