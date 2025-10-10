import torchaudio
import librosa
import io
import torch
import random
import numpy as np

from pathlib import Path
from torchaudio import functional as AF
from torch.nn import functional as F
from src.utils.config_loader import RAW_DIR, PROCESSED_DIR

# Gets the absolute path so that we can append our folder paths.
CURRENT_PATH = Path().absolute()


class AudioPreprocessor:
    """
    AudioPreprocessor is a utility class for loading, preprocessing, and converting
    raw audio waveforms into normalized tensor waveforms.

    The preprocessing pipeline includes:
    - Loading audio from disk
    - Resampling to a target sampling rate (default: 16 kHz)
    - Trimming or padding to a fixed length (default: 120 seconds)
    - Waveform normalization (per-sample)
    - Returning or saving waveforms for testing.


    Parameters
    ----------
    script : {"train"}, optional
        Condition to apply certain training methods

    waveform_norm : {"std", "minmax"}, optional
        Normalization method for waveforms:
        - "std": divide by standard deviation
        - "minmax": scale to [0, 1]

    """

    def __init__(self, script="train", waveform_norm="peak"):
        self.SCRIPT = script
        self.INPUT_SAMPLING = 48000
        self.TARGET_SAMPLING = 16000
        self.TARGET_NUM_SAMPLE = 1920000  # This means 120 seconds or 2 minutes
        self.INPUT_PATH = CURRENT_PATH / RAW_DIR
        self.OUTPUT_PATH = CURRENT_PATH / PROCESSED_DIR
        self.WAVEFORM_NORM = waveform_norm

    def load_audio(self, audiofile):
        """
        Load an MP3 audio file (disk or bytes) using librosa,
        then convert to a torch.Tensor.

        Parameters
        ----------
        audiofile : str | bytes | io.BytesIO
            Path (relative to INPUT_PATH) or in-memory audio bytes.

        Returns
        -------
        waveform : torch.Tensor
            Audio waveform as a tensor of shape (channels, num_samples).
        sample_rate : int
            Original sampling rate of the audio.
        """
        try:
            if isinstance(audiofile, str):
                if not audiofile.endswith(".mp3"):
                    audiofile = f"{audiofile}.mp3"
                file = self.INPUT_PATH / audiofile

                # FIXED: Force librosa to load properly
                # Load at native sample rate first, then we will resample later
                y, sr = librosa.load(str(file), sr=None, mono=False, dtype=np.float32)
                
                # If loading fails (all zeros), try with explicit sample rate
                if np.abs(y).max() < 0.0001:
                    print(f"Warning: First load failed, trying with sr=48000")
                    y, sr = librosa.load(str(file), sr=48000, mono=False, dtype=np.float32)
                
                # Last resort: use soundfile instead
                if np.abs(y).max() < 0.0001:
                    print(f"Warning: Librosa failed, trying soundfile")
                    import soundfile as sf
                    y, sr = sf.read(str(file), dtype='float32')
                    if y.ndim == 2:
                        y = y.T  # soundfile returns (samples, channels)
                    else:
                        y = y[None, :]  # make it (1, samples)

            elif isinstance(audiofile, (bytes, io.BytesIO)):
                file = (
                    io.BytesIO(audiofile) if isinstance(audiofile, bytes) else audiofile
                )
                file.seek(0)

                y, sr = librosa.load(file, sr=None, mono=False)

            elif isinstance(audiofile, np.ndarray):
                # Handle numpy array directly (from librosa or OpenUnmix)
                y = audiofile
                # Default sample rate (we can make this configurable moving forward... but I hardcoded for now)
                sr = 44100

            else:
                raise ValueError(f"Unsupported audiofile type: {type(audiofile)}")

            # Verify we actually loaded audio
            if np.abs(y).max() < 0.0001:
                raise RuntimeError(f"Audio file appears to be silent or corrupted: {audiofile}")

            # Ensure consistent shape
            if y.ndim == 1:
                y = y[None, :]
            else:
                y = y.T if y.shape[0] > y.shape[1] else y

            waveform = torch.from_numpy(y).float()
            
            return waveform, sr

        except Exception as e:
            raise RuntimeError(
                f"Error: File cannot be loaded. Check the filename and type. {e}"
            )

    def resample_audio(self, original_sr, waveform):
        """
        Resample waveform to the target sampling rate.

        Parameters
        ----------
        original_sr : int
            Original sampling rate of the waveform.
        waveform : tensor
            Input audio waveform.

        Returns
        -------
        waveform : tensor
            Resampled audio waveform at `TARGET_SAMPLING`.
        """
        if original_sr != self.TARGET_SAMPLING:
            #print(f"Current waveform is {original_sr}, to convert to {self.TARGET_SAMPLING}.")
            waveform = AF.resample(
                waveform, orig_freq=original_sr, new_freq=self.TARGET_SAMPLING
            )
        return waveform

    def pad_trim(self, waveform, random_crop=False):
        """
        Pad or trim waveform to exactly `TARGET_NUM_SAMPLE`.
        If `random_crop=True`, perform random cropping or random padding.

        Parameters
        ----------
        waveform : tensor
            Input audio waveform.
        random_crop : bool
            Whether to randomly crop/pad (augmentation).
        """
        num_samples = waveform.shape[-1]

        if num_samples > self.TARGET_NUM_SAMPLE:
            # Trim with optional random crop
            if random_crop:
                max_start = num_samples - self.TARGET_NUM_SAMPLE
                start = random.randint(0, max_start)
                return waveform[..., start : start + self.TARGET_NUM_SAMPLE]
            else:
                return waveform[..., : self.TARGET_NUM_SAMPLE]

        elif num_samples < self.TARGET_NUM_SAMPLE:
            padding_amount = self.TARGET_NUM_SAMPLE - num_samples
            if random_crop:
                # Randomly distribute padding left vs right
                left = random.randint(0, padding_amount)
                right = padding_amount - left
                return F.pad(waveform, (left, right))
            else:
                # Default: pad at the end
                return F.pad(waveform, (0, padding_amount))

        else:
            return waveform

    def normalize_waveform(self, waveform, method):
        """
        Normalize audio waveform.

        Parameters
        ----------
        waveform : tensor
            Input audio waveform.
        method : {"std", "minmax"}
            Normalization strategy.

        Returns
        -------
        waveform : tensor
            Normalized audio waveform.
        """
        if method == "peak":
            # Normalize to [-1, 1] based on max absolute value to preserves relative dynamics
            peak = waveform.abs().max()
            return waveform / max(peak, 1e-6)
        elif method == "std":
            std = waveform.std()
            return waveform / max(std, 1e-6)
        elif method == "minmax":
            waveform = waveform - waveform.min()
            return waveform / max(waveform.max(), 1e-6)
        return waveform

    def save_waveform(self, waveform, filename) -> None:
        """
        Save waveform to disk as a .wav file.

        Parameters
        ----------
        waveform : tensor
            Song to save.
        filename : str
            Base filename to use.
        """
        self.OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
        #print(f"Saving {filename} to {self.OUTPUT_PATH}.")
        
        output_path = self.OUTPUT_PATH / f"{filename}"

        torchaudio.save(str(output_path), waveform, self.TARGET_SAMPLING)

    def __call__(self, file, skip_time=0, train=False):
        """
        Process an audio file and return its normalized waveform.

        Parameters
        ----------
        file : str/audio_media
            Path of the audio to process or audio media from the API
        skip_time : float
            Number of seconds to skip from the start of the file.
        train : boolean
            False for inference/prediction, True for training.

        Returns
        -------
        tensor
            Normalized tensor of a waveform
        """
        waveform, sample_rate = self.load_audio(file)

        # Resample the audio to 16kHz
        waveform = self.resample_audio(original_sr=sample_rate, waveform=waveform)

        # Convert the audio into mono
        if waveform.shape[0] > 1:
            #print("Current audio is stereo. Converting to mono.")
            waveform = waveform.mean(dim=0, keepdim=True)

        # If there is a skip value provided, trim it
        if skip_time is not None and skip_time > 0:
            # print(f"Skipping first {skip_time:.2f} seconds.")
            start_sample = int(skip_time * self.TARGET_SAMPLING)
            waveform = waveform[:, start_sample:]

        # Trim if more than 120 seconds, pad if less than
        waveform = self.pad_trim(waveform=waveform, random_crop=train)

        # Normalize waveform (used PEAK)
        waveform = self.normalize_waveform(waveform, method=self.WAVEFORM_NORM)

        # Add some gaussian noise to the waveform during training
        if train:
            waveform += torch.randn_like(waveform) * 1e-4

        return waveform
