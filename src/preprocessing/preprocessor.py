import pandas as pd
import numpy as np

from src.preprocessing.audio_preprocessor import AudioPreprocessor
from src.preprocessing.lyrics_preprocessor import LyricsPreprocessor
from src.utils.config_loader import DATASET_CSV

dataset_path = DATASET_CSV

def bulk_preprocessing(batch, batch_count: int):
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

    audio_list = []
    lyric_list = []
    count = 1
    batch_length = len(batch)

    print(f"Preprocessing training data with length {batch_length}\n")

    for row in batch.itertuples():
        print(f"Batch {batch_count}     -    {count}/{batch_length}")
        processed_song = audio_preprocessor(filename=row.id)
        audio_list.append(processed_song)

        processed_lyric = lyric_preprocessor(lyrics=row.lyrics)
        lyric_list.append(processed_lyric)

        count += 1

    return audio_list, lyric_list


def single_preprocessing(audio, lyric):
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
    processed_song : audio_object
        Audio object file
    
    processed_lyric : string
        Lyric string
    """
    # Instantiate preprocessor classes
    audio_preprocessor = AudioPreprocessor(script="predict")
    lyric_preprocessor = LyricsPreprocessor()

    processed_song = audio_preprocessor(filename=audio)

    processed_lyric = lyric_preprocessor(lyrics=lyric)

    return processed_song, processed_lyric


def dataset_read():
    """
    Reads the csv file, filters the records of train split, returns batches of data

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
    dataset = pd.read_csv(dataset_path)
    label = dataset['target'].tolist()

    # split into twenty sections
    data_splits = np.array_split(dataset, 20)

    return data_splits, label
