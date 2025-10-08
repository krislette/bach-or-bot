import threading
import torch
import numpy as np
from pathlib import Path
from types import SimpleNamespace

from src.spectttra.spectttra import SpecTTTra, build_spectttra_from_cfg, load_frozen_spectttra
from src.spectttra.feature import FeatureExtractor
from src.spectttra.spectttra import SpecTTTra

# Shared variables for the model and setup, loaded only once and reused (cache)
_PREDICTOR_LOCK = threading.Lock()
_FEAT_EXT = None
_MODEL = None
_CFG = None
_DEVICE = None


def build_spectttra(cfg, device):
    """
    Wrapper that builds SpecTTTra + FeatureExtractor and loads frozen checkpoint.
    """
    feat_ext, model = build_spectttra_from_cfg(cfg, device)
    model = load_frozen_spectttra(model, "models/spectttra/spectttra_frozen.pth", device)
    return feat_ext, model


def _init_predictor_once():
    """
    Initialize and cache FeatureExtractor and SpecTTTra once per process.

    Ensures thread-safe, one-time initialization of the feature extractor and
    transformer model, including moving them to the appropriate device.

    This function also sets default configurations for audio,
    mel-spectrogram extraction, and model architecture.
    """

    global _FEAT_EXT, _MODEL, _CFG, _DEVICE

    if _MODEL is not None and _FEAT_EXT is not None:
        return

    with _PREDICTOR_LOCK:
        if _MODEL is not None and _FEAT_EXT is not None:
            return

        # Configurations of best performing variant for 120s
        cfg = SimpleNamespace(
            audio=SimpleNamespace(sample_rate=16000, max_time=120, max_len=16000 * 120),
            melspec=SimpleNamespace(
                n_fft=2048,
                hop_length=512,
                win_length=2048,
                n_mels=128,
                f_min=20,
                f_max=8000,
                power=2,
                top_db=80,
                norm="mean_std",
            ),
            model=SimpleNamespace(
                embed_dim=384,
                num_heads=6,
                num_layers=12,
                t_clip=3,
                f_clip=1,
                pre_norm=True,
                pe_learnable=True,
                pos_drop_rate=0.1,
                attn_drop_rate=0.1,
                proj_drop_rate=0.0,
                mlp_ratio=2.67,
            ),
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        feat_ext, model = build_spectttra(cfg, device)

        feat_ext.to(device)

        # Move model to device (GPU if available) and allow faster inference with mixed precision
        model.to(device)
        model.eval()

        # Cache
        _FEAT_EXT = feat_ext
        _MODEL = model
        _CFG = cfg
        _DEVICE = device


def spectttra_single(audio_tensor):
    """
    Run single-input inference with SpecTTTra.

    Args:
        audio_tensor (torch.Tensor): Input waveform of shape (1, num_samples).
            Must already be preprocessed including resampled to the target sampling rate (16 kHz).

    Returns:
        np.ndarray:
            1D embedding vector of shape (embed_dim,). The embedding is obtained
            by mean-pooling the transformer token outputs.
    """
    global _FEAT_EXT, _MODEL, _CFG, _DEVICE

    _init_predictor_once()

    device = _DEVICE
    feat_ext = _FEAT_EXT
    model = _MODEL
    cfg = _CFG

    # Move waveform to device but keep float for mel extraction
    waveform = audio_tensor.to(device).float()

    with torch.no_grad():
        # Extract mel-spectrogram
        melspec = feat_ext(waveform)        # (B, n_mels, n_frames)

        if device.type == "cuda":
            with torch.cuda.amp.autocast(enabled=True):
                tokens = model(melspec)     # (B, num_tokens, embed_dim)
                pooled = tokens.mean(dim=1) # (B, embed_dim)
        else:
            tokens = model(melspec)
            pooled = tokens.mean(dim=1)

    # Return numpy vector
    out = pooled.squeeze(0).cpu().numpy()   # (embed_dim,)
    return out


def spectttra_batch(audio_tensors):
    """
    Run batch input training with SpecTTTra.

    Args:
        audio_tensors (list[torch.Tensor]):
            List of input waveforms. Each element should be shaped either
            (num_samples,) or (1, num_samples). Each waveform is processed
            independently and its pooled embedding is collected.

    Returns:
        np.ndarray:
            2D array of shape (batch_size, embed_dim), where each row
            corresponds to the pooled embedding for one input waveform.
    """

    global _FEAT_EXT, _MODEL, _CFG, _DEVICE

    _init_predictor_once()

    if not audio_tensors:
        return np.empty((0, _CFG.model.embed_dim))

    feat_ext = _FEAT_EXT
    model = _MODEL
    device = _DEVICE

    batch = []
    for waveform in audio_tensors:
        with torch.no_grad():
            melspec = feat_ext(waveform.float())    # (B, n_mels, n_frames)

            if device.type == "cuda":
                with torch.cuda.amp.autocast(enabled=True):
                    tokens = model(melspec)         # (B, num_tokens, embed_dim)
                    pooled = tokens.mean(dim=1)     # (B, embed_dim)
            else:
                tokens = model(melspec)
                pooled = tokens.mean(dim=1)
        
        batch.append(pooled.cpu().numpy())

    return np.vstack(batch)

def spectttra_stack(audio_tensors, max_batch_size: int = 100):
    """
    Vectorized inference over a list of waveforms, processed in chunks to cap GPU/CPU memory.

    Args:
        audio_tensors (list[torch.Tensor]): Each tensor shaped (1, num_samples) (batch dim first).
        max_batch_size (int): Maximum number of waveforms to stack and process at once.

    Returns:
        np.ndarray: (N, embed_dim) embeddings in the same order as input.
    """
    global _FEAT_EXT, _MODEL, _CFG, _DEVICE

    _init_predictor_once()

    if not audio_tensors:
        return np.empty((0, _CFG.model.embed_dim))

    feat_ext = _FEAT_EXT
    model = _MODEL
    device = _DEVICE

    embeddings = []

    with torch.no_grad():
        for start in range(0, len(audio_tensors), max_batch_size):
            chunk = audio_tensors[start:start + max_batch_size]

            # Basic shape check
            for i, t in enumerate(chunk):
                if t.dim() != 2 or t.size(0) != 1:
                    raise ValueError(f"[NOTE] Expected each tensor shaped (1, num_samples); got {tuple(t.shape)} at index {start + i}")

            waveform_batch = torch.cat(chunk, dim=0).to(device).float()  # (B, samples)
            melspec = feat_ext(waveform_batch)  # (B, n_mels, n_frames)

            if device.type == "cuda":
                with torch.cuda.amp.autocast(enabled=True):
                    tokens = model(melspec)          # (B, num_tokens, embed_dim)
                    pooled = tokens.mean(dim=1)      # (B, embed_dim)
            else:
                tokens = model(melspec)
                pooled = tokens.mean(dim=1)

            embeddings.append(pooled.cpu())

    return torch.cat(embeddings, dim=0).numpy()