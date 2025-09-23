import torchaudio
import io

from pathlib import Path
from torchaudio import functional as AF
from torch.nn import functional as F
from src.utils.config_loader import RAW_DIR, PROCESSED_DIR

# Gets the absolute path so that we can append our folder paths.
CURRENT_PATH = Path().absolute()

class AudioPreprocessor:
    """
    AudioPreprocessor is a utility class for loading, preprocessing, and converting
    raw audio waveforms into normalized log-Mel spectrograms.

    The preprocessing pipeline includes:
    - Loading audio from disk
    - Resampling to a target sampling rate (default: 16 kHz)
    - Trimming or padding to a fixed length (default: 120 seconds)
    - Waveform normalization (per-sample)
    - Mel-spectrogram computation
    - Log-scaling and spectrogram normalization
    - Saving spectrograms as `.npy` files

    Parameters
    ----------
    label : str
        The label associated with the audio files ("real", "fake", etc.).
    waveform_norm : {"std", "minmax"}, optional
        Normalization method for waveforms:
        - "std": divide by standard deviation
        - "minmax": scale to [0, 1]
    spec_norm : {"mean_std", "min_max", "simple"}, optional
        Normalization method for spectrograms:
        - "mean_std": standardize to zero mean, unit variance
        - "min_max": scale to [0, 1]
        - "simple": scale using (S - 40)/80 (used in some works)
    """

    def __init__(self, script="train", waveform_norm="std", spec_norm="min_max"):
        self.SCRIPT = script
        self.INPUT_SAMPLING = 48000
        self.TARGET_SAMPLING = 16000
        self.TARGET_NUM_SAMPLE = 1920000    # This means 120 seconds or 2 minutes
        self.INPUT_PATH = CURRENT_PATH / RAW_DIR
        self.OUTPUT_PATH = CURRENT_PATH / PROCESSED_DIR
        self.WAVEFORM_NORM = waveform_norm
        self.SPEC_NORM = spec_norm


    def load_audio(self, audiofile):
        """
        Load an audio file from disk.

        Parameters
        ----------
        filename : str
            Name of the file (without extension) to load from `INPUT_PATH`.

        Returns
        -------
        waveform : tensor
            The mono audio waveform.
        sample_rate : int
            The original sampling rate of the audio.
        """
        try:
            if isinstance(audiofile, str):
                if '.' not in audiofile:
                    audiofile = f"{audiofile}.mp3"
                file = self.INPUT_PATH / audiofile
            elif isinstance(audiofile, (bytes, io.BytesIO)):
                file = io.BytesIO(audiofile) if isinstance(audiofile, bytes) else audiofile
                file.seek(0)
            else:
                raise ValueError(f"Unsupported audiofile type: {type(audiofile)}")
            
            waveform, sample_rate = torchaudio.load(file)

        except Exception as e:
            raise RuntimeError(f"Error: File cannot be loaded. Check the filename and type. {e}")

        return waveform, sample_rate


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
            print(f"Current waveform is {original_sr}, to convert to {self.TARGET_SAMPLING}.")
            waveform = AF.resample(
                waveform, 
                orig_freq=original_sr, 
                new_freq=self.TARGET_SAMPLING
            )
        return waveform
        

    def pad_trim(self, waveform):
        """
        Pad or trim waveform to exactly `TARGET_NUM_SAMPLE`.

        Parameters
        ----------
        waveform : tensor
            Input audio waveform.

        Returns
        -------
        waveform : tensor
            Waveform of fixed length (trimmed or zero-padded).
        """
        num_samples = waveform.shape[-1]

        if self.TARGET_NUM_SAMPLE < num_samples:
            print(f"Audio {num_samples} longer than {self.TARGET_NUM_SAMPLE}. Starting trimming.")
            return waveform[..., :self.TARGET_NUM_SAMPLE]
        elif self.TARGET_NUM_SAMPLE > num_samples:
            print(f"Audio {num_samples} shorter than {self.TARGET_NUM_SAMPLE}. Starting padding.")
            padding_amount = self.TARGET_NUM_SAMPLE - num_samples
            return F.pad(
                waveform, 
                (0, padding_amount), 
            )
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
        if method == "std":
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
        print(f"Saving {filename} to {self.OUTPUT_PATH}.")
        
        output_path = self.OUTPUT_PATH / f"{filename}"

        torchaudio.save(str(output_path), waveform, self.TARGET_SAMPLING)


    def download_preprocessed(self, filename, skip_time=0):
        """
        Full preprocessing pipeline that loads, normalizes, and saves spectrogram.

        Parameters
        ----------
        filename : str
            Name of the audio file to process.
        skip_time : float
            Number of seconds to skip from the start of the file.
        """
        waveform, sample_rate = self.load_saved_audio(filename)
    
        # Resample the audio to 16kHz
        waveform = self.resample_audio(original_sr=sample_rate, waveform=waveform)

        # Convert the audio into mono
        if waveform.shape[0] > 1:
            print("Current audio is stereo. Converting to mono.")
            waveform = waveform.mean(dim=0, keepdim=True)

        # If there is a skip value provided, trim it
        if skip_time is not None and skip_time > 0:
            print(f"Skipping first {skip_time:.2f} seconds.")
            start_sample = int(skip_time * self.TARGET_SAMPLING)
            waveform = waveform[:, start_sample:]

        # Trim if more than 120 seconds, pad if less than
        waveform = self.pad_trim(waveform=waveform)

        # Normalize waveform (aligned with SONICS)
        waveform = self.normalize_waveform(waveform, method=self.WAVEFORM_NORM)

        # Save the spectrogram
        self.save_waveform(waveform, filename)


    def __call__(self, filename, skip_time=0):
        """
        Process an audio file and return its normalized log-Mel spectrogram.

        Parameters
        ----------
        filename : str
            Name of the audio file to process.
        skip_time : float
            Number of seconds to skip from the start of the file.

        Returns
        -------
        tensor
            Normalized log-Mel spectrogram.
        """
        waveform, sample_rate = self.load_audio(filename)
    
        # Resample the audio to 16kHz
        waveform = self.resample_audio(original_sr=sample_rate, waveform=waveform)
        
        # Convert the audio into mono
        if waveform.shape[0] > 1:
            print("Current audio is stereo. Converting to mono.")
            waveform = waveform.mean(dim=0, keepdim=True)

        # If there is a skip value provided, trim it
        if skip_time is not None and skip_time > 0:
            print(f"Skipping first {skip_time:.2f} seconds.")
            start_sample = int(skip_time * self.TARGET_SAMPLING)
            waveform = waveform[:, start_sample:]

        # Trim if more than 120 seconds, pad if less than
        waveform = self.pad_trim(waveform=waveform)

        # Normalize waveform (aligned with SONICS)
        waveform = self.normalize_waveform(waveform, method=self.WAVEFORM_NORM)

        return waveform