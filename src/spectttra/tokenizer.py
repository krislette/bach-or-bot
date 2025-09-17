import math
import torch
import torch.nn as nn
from .embedding import (
    SinusoidPositionalEncoding,
    LearnedPositionalEncoding,
)


class STTokenizer(nn.Module):
    """
    Spectro-temporal tokenizer that converts mel-spectrograms into a sequence of tokens.

    Both temporal and spectral dimensions are tokenized separately and then
    concatenated to form spectro-temporal tokens.

    Args:
        input_spec_dim (int): Number of frequency bins in the spectrogram.
        input_temp_dim (int): Number of time frames in the spectrogram.
        t_clip (int): Temporal clip size (stride for temporal tokenization).
        f_clip (int): Spectral clip size (stride for spectral tokenization).
        embed_dim (int): Dimensionality of each token embedding.
        pre_norm (bool, optional): Whether to apply pre-normalization with LayerNorm. Defaults to False.
        pe_learnable (bool, optional): Whether to use learnable positional encodings. Defaults to False.
    """

    def __init__(
        self,
        input_spec_dim,
        input_temp_dim,
        t_clip,
        f_clip,
        embed_dim,
        pre_norm=False,
        pe_learnable=False,
    ):
        super(STTokenizer, self).__init__()
        self.input_spec_dim = input_spec_dim
        self.input_temp_dim = input_temp_dim
        self.t_clip = t_clip
        self.f_clip = f_clip
        self.embed_dim = embed_dim
        self.pre_norm = pre_norm
        self.pe_learnable = pe_learnable

        # Compute number of tokens
        self.num_temporal_tokens = math.floor(
            (input_temp_dim - t_clip) / t_clip + 1
        )   # e.g., floor((1280 - 5) / 5 + 1) = 256
        self.num_spectral_tokens = math.floor(
            (input_spec_dim - f_clip) / f_clip + 1
        )   # e.g., floor((128 - 3) / 3 + 1) = 42
        self.num_tokens = (
            self.num_temporal_tokens + self.num_spectral_tokens
        )

         # Temporal and spectral tokenizers
        self.temporal_tokenizer = Tokenizer1D(
            input_spec_dim,
            embed_dim,
            clip_size=t_clip,
            num_clips=self.num_temporal_tokens,
            pre_norm=pre_norm,
            pe_learnable=pe_learnable,
        )
        self.spectral_tokenizer = Tokenizer1D(
            input_temp_dim,
            embed_dim,
            clip_size=f_clip,
            num_clips=self.num_spectral_tokens,
            pre_norm=pre_norm,
            pe_learnable=pe_learnable,
        )

    def forward(self, x):
        """
        Forward pass of spectro-temporal tokenizer.

        Args:
            x (torch.Tensor): Input mel-spectrogram of shape (batch_size, freq_bins, time_frames).

        Returns:
            torch.Tensor: Spectro-temporal tokens of shape 
                (batch_size, num_temporal_tokens + num_spectral_tokens, embed_dim).
        """
        # Temporal tokenization
        temporal_input = x  # shape: (B, F, T)
        temporal_tokens = self.temporal_tokenizer(
            temporal_input
        )   # shape: (B, T/t, dim)

        # Spectral tokenization
        spectral_input = x.permute(0, 2, 1) # shape: (batch_size, T, F)
        spectral_tokens = self.spectral_tokenizer(
            spectral_input
        )   # shape: (B, F/f, dim)

        # Concatenate along token dimension
        spectro_temporal_tokens = torch.cat(
            (temporal_tokens, spectral_tokens), dim=1
        )   # shape: (B, T/t + F/f, dim)
        return spectro_temporal_tokens


class Tokenizer1D(nn.Module):
    """
    One-dimensional tokenizer for either temporal or spectral dimension.

    Applies a 1D convolution with stride equal to the clip size, followed by
    GELU activation, positional encoding, and optional LayerNorm.

    Args:
        input_dim (int): Input dimension size (frequency for temporal, time for spectral).
        token_dim (int): Output token embedding dimension.
        clip_size (int): Window/stride size for tokenization.
        num_clips (int): Number of tokens produced.
        pre_norm (bool, optional): Whether to apply pre-normalization with LayerNorm. Defaults to False.
        pe_learnable (bool, optional): Whether to use learnable positional encodings. Defaults to False.
    """

    def __init__(
        self,
        input_dim,
        token_dim,
        clip_size,
        num_clips,
        pre_norm=False,
        pe_learnable=False,
    ):
        super(Tokenizer1D, self).__init__()
        self.conv1d = nn.Conv1d(
            input_dim,
            token_dim,
            clip_size,
            stride=clip_size,
            bias=not pre_norm,  # Disable bias if pre-norm is used (e.g. CLIP)
        )
        self.act = nn.GELU()
        self.pos_encoder = (
            SinusoidPositionalEncoding(token_dim)
            if not pe_learnable
            else LearnedPositionalEncoding(token_dim, num_clips)
        )
        self.norm_pre = nn.LayerNorm(token_dim, eps=1e-6) if pre_norm else nn.Identity()

    def forward(self, x):
        """
        Forward pass of 1D tokenizer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim, length).

        Returns:
            torch.Tensor: Sequence of tokens with shape (batch_size, num_clips, token_dim).
        """

        x = x                    # (F, T)
        x = self.conv1d(x)       # (F, T) -> (dim, T/t)
        x = self.act(x)
        x = x.transpose(1, 2)    # (dim, T/t) -> (T/t, dim)
        x = self.pos_encoder(x)  # Add position embeddings
        x = self.norm_pre(x)
        return x
