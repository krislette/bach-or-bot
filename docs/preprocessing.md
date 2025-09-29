# Preprocessing Documentation
This document will serve as a general guide towards method implementations of preprocessing for our thesis!

## Preprocessing Methods
For our preprocessing, we have two main classes that we can use:
- `AudioPreprocessor`
- `LyricsPreprocessor`

Both classes needs to be instantiated before usage. Methods for accessing and instantiating said classes can be accessed via `preprocessor.py`

---
### `preprocessor.py`
Located at *src/preprocessing/preprocessor.py*, this file has several methods that can be accessed for preprocessing:

`dataset_read(batch_size = 20)`
- Reads the main `.csv` file, and returns the batches of data.
- **Parameters:** 
	- `batch_size` → `int`. Determines the number of splits.
- **Returns:** 
	-  `data_splits`, a list of `pd.DataFrame` from the split and.
	- `label`, a list of `boolean` values representing real or fake.


`single_preprocessing(audio, lyric: str)`
- Preprocess a single record of audio and lyric data. The audio parameter provided can be a file and does not need to be saved in the disk.
- **Parameters:**
	- `file` → `[str/audio_object]`. 
		- It can take either a file name and load it from the disk or,
		- Take in an audio object which is most likely coming from an API.
	- `lyric` → `str`, a lyrical string.
- **Returns:**
	- `processed_song` → `tensor` of waveform, for feature extraction.
	- `processed_lyric` → `str`, for feature extraction.


`bulk_preprocessing(batch: pd.DataFrame, batch_count: int)`
- Reads through the `DataFrame`, loads audio from disk and lyrics from `.csv`, applies preprocessing and returns preprocessed lists.
- **Parameters:**
	- `batch` → `pd.DataFrame`, containing sub-split of values.
	- `batch_count` → `int`, used to display what batch is being processed.
- **Returns:**
	- `audio_list` → `list`, containing `tensors`.
	- `lyric_list` → `list`, containing lyrics in `str` type.

---
### `AudioPreprocessor`
This class located inside the preprocessing folder implements the basic audio preprocessing methods.

**Instantiation**
We can instantiate the class via: 
`AudioPreprocessor(script="train", waveform_norm="std")`
- `script` determines if the audio preprocessor will apply training methods (random cropping, gaussian noise) or not. Leave blank if using it for training, use `predict` if inferencing/predicting.
- `waveform_norm` determines normalization that will be applied to the waveform. In default it uses `std`, which is also used by SpecTTTra.

The only relevant script for training and production is the following:

`__call__(file, skip_time=0, train=False)`
- This method can be called by plainly using the instantiated class instance like `audio_preprocessor(file1, 2.3, True)`
- **Parameters:**
	- `file` → `[str/audio_object]`. 
		- It can take either a file name and load it from the disk or,
		- Take in an audio object which is most likely coming from an API.
	- `skip_time` → `int`, determines the amount of seconds to skip through before the start of vocals.
	- `train` → `str`, determines if the preprocessor class will use training methods or opt for prediction.
- **Returns:**
	- `waveform` → `tensor` version of the audio.

---
### `LyricsPreprocessor`
This class located inside the preprocessing folder implements the basic lyric preprocessing methods.

**Instantiation**
We can instantiate the class via:
`LyricsPreprocessor(keep_case=True, keep_punctuation=True)`
- `keep_case`→`bool` determines whether or not the lyrics will be kept to its true form or `lower()` will be applied.
- `keep_punctuation`→`bool` determines whether or not the punctuations will be kept within the lyrics or if they will be removed.

There are two relevant scripts for this preprocessing class:

`__call__(lyrics: str)`
- This method can be called by plainly using the instantiated class instance like `lyrics_preprocessor(lyric1)`
- **Parameters:**
	- `lyrics` → `str`, takes up a raw string of lyrics for preprocessing
- **Returns:**
	- `lyrics_cleaned` → `str`, preprocessed string of lyrics.

`musiclime_lyrics_extractor(lyrics: str)`
- Preprocesses a lyrics and segments it into individual lines, stored in a list.
- **Parameters:**
	- `lyrics` → `str`, takes up a raw string of lyrics for preprocessing
- **Returns:**
	- `line_segmented_lyrics` → `list`, collection of lines from the lyrics.
