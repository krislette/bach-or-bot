import threading
import torch
import numpy as np
from pathlib import Path
from types import SimpleNamespace

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
    Initialize SpecTTTra and FeatureExtractor modules, and load a frozen checkpoint.

    Args:
        cfg (SimpleNamespace): Configuration containing audio, mel-spectrogram, and model parameters.
        device (torch.device): Target device for model and feature extractor.

    Returns:
        tuple:
            FeatureExtractor: Module for converting raw audio into mel-spectrogram features.
            SpecTTTra: Spectro-temporal transformer model initialized with checkpoint weights.
    """
    feat_ext = FeatureExtractor(cfg).to(device)

    # Build model once using placeholder input to infer mel and frame dimensions
    with torch.no_grad():
        dummy_wave = torch.zeros(1, cfg.audio.max_len, device=device)
        dummy_mel = feat_ext(dummy_wave.float())
    _, n_mels, n_frames = dummy_mel.shape

    model_cfg = cfg.model
    model = SpecTTTra(
        input_spec_dim=n_mels,
        input_temp_dim=n_frames,
        embed_dim=model_cfg.embed_dim,
        t_clip=model_cfg.t_clip,
        f_clip=model_cfg.f_clip,
        num_heads=model_cfg.num_heads,
        num_layers=model_cfg.num_layers,
        pre_norm=model_cfg.pre_norm,
        pe_learnable=model_cfg.pe_learnable,
        pos_drop_rate=model_cfg.pos_drop_rate,
        attn_drop_rate=model_cfg.attn_drop_rate,
        proj_drop_rate=model_cfg.proj_drop_rate,
        mlp_ratio=model_cfg.mlp_ratio,
    ).to(device)

    # Load frozen checkpoint if it exists; otherwise, save initial state
    ckpt_path = Path("models/spectttra/spectttra_frozen.pth")
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state)
        print(f"[INFO] Loaded frozen SpecTTTra checkpoint from {ckpt_path}")
    else:
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt_path)
        print(f"[INFO] Saved frozen SpecTTTra checkpoint to {ckpt_path}")

    model.eval()
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


def spectttra_predict(audio_tensor):
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


def spectttra_train(audio_tensors):
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
        waveform = waveform.to(device)
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