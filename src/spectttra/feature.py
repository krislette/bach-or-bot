import torch
import numpy as np
import torch.nn as nn

try:
    from torch.amp import autocast

    torch_amp_new = True
except:
    from torch.cuda.amp import autocast

    torch_amp_new = False

from torchaudio.transforms import AmplitudeToDB, MelSpectrogram


class FeatureExtractor(nn.Module):
    """
    Converts raw audio waveforms into normalized mel-spectrogram features.

    Args:
        cfg (object): Configuration object containing parameters for audio 
            processing and spectrogram generation.
    """
    
    def __init__(
        self,
        cfg,
    ):
        super().__init__()

        self.audio2melspec = MelSpectrogram(
            n_fft=cfg.melspec.n_fft,
            hop_length=cfg.melspec.hop_length,
            win_length=cfg.melspec.win_length,
            n_mels=cfg.melspec.n_mels,
            sample_rate=cfg.audio.sample_rate,
            f_min=cfg.melspec.f_min,
            f_max=cfg.melspec.f_max,
            power=cfg.melspec.power,
        )
        self.amplitude_to_db = AmplitudeToDB(top_db=cfg.melspec.top_db)

        if cfg.melspec.norm == "mean_std":
            self.normalizer = MeanStdNorm()
        elif cfg.melspec.norm == "min_max":
            self.normalizer = MinMaxNorm()
        elif cfg.melspec.norm == "simple":
            self.normalizer = SimpleNorm()
        else:
            self.normalizer = nn.Identity()

    def forward(self, x):
        """
        Forward pass of the feature extractor.

        Args:
            x (torch.Tensor): Raw audio input of shape (batch_size, num_samples).

        Returns:
            torch.Tensor: Normalized mel-spectrogram features of shape 
                (batch_size, n_mels, time).
        """

        with (
            autocast("cuda", enabled=False)
            if torch_amp_new
            else autocast(enabled=False)
        ):
            melspec = self.audio2melspec(x.float())
            melspec = self.amplitude_to_db(melspec)
            melspec = self.normalizer(melspec)

        return melspec


class MinMaxNorm(nn.Module):
    """
    Applies min-max normalization to input tensors.

    Args:
        eps (float, optional): Small constant to prevent division by zero. Defaults to 1e-6.
    """

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, X):
        """
        Forward pass of min-max normalization.

        Args:
            X (torch.Tensor): Input tensor of shape (batch_size, n_mels, time).

        Returns:
            torch.Tensor: Min-max normalized tensor of the same shape.
        """
        min_ = torch.amin(X, dim=(1, 2), keepdim=True)
        max_ = torch.amax(X, dim=(1, 2), keepdim=True)
        return (X - min_) / (max_ - min_ + self.eps)


class SimpleNorm(nn.Module):
    """
    Applies a simple linear normalization to input tensors.
    
    Normalizes values by shifting and scaling using fixed constants:
    (x - 40) / 80.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        Forward pass of simple normalization.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, n_mels, time).

        Returns:
            torch.Tensor: Normalized tensor of the same shape.
        """
        return (x - 40) / 80


class MeanStdNorm(nn.Module):
    """
    Applies mean-std normalization to input tensors.

    Args:
        eps (float, optional): Small constant to prevent division by zero. Defaults to 1e-6.
    """

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, X):
        """
        Forward pass of mean-std normalization.

        Args:
            X (torch.Tensor): Input tensor of shape (batch_size, n_mels, time).

        Returns:
            torch.Tensor: Normalized tensor of the same shape.
        """
        mean = X.mean((1, 2), keepdim=True)
        std = X.reshape(X.size(0), -1).std(1, keepdim=True).unsqueeze(-1)
        return (X - mean) / (std + self.eps)
    