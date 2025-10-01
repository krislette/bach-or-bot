import pandas as pd
import numpy as np

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


def dataset_read(batch_size = 20):
    """
    Reads the csv file and returns batches of data

    Parameters
    ----------
    None

    Returns
    -------
    data_splits : list
        List of dataframes acting as batches
    
    label : list
        List of real/fake labels (in the formm of 0 and 1)
    """
    dataset = pd.read_csv(DATASET_CSV)
    label = dataset['target'].tolist()

    # Split into x batches (50,000 / x)
    data_splits = np.array_split(dataset, batch_size)

    return data_splits, label
