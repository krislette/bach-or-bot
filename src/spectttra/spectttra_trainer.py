import torch
import numpy as np
from pathlib import Path
from types import SimpleNamespace

from src.spectttra.feature import FeatureExtractor
from src.spectttra.spectttra import SpecTTTra


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
        print(f"Loaded frozen SpecTTTra checkpoint from {ckpt_path}")
    else:
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved frozen SpecTTTra checkpoint to {ckpt_path}")

    model.eval()
    return feat_ext, model


def spectttra_train(audio_tensors):
    """
    Extract pooled audio embeddings from preprocessed waveforms.

    Args:
        audio_tensors (list[torch.Tensor]): List of input waveforms, 
            each of shape (1, num_samples) or (num_samples,).

    Returns:
        np.ndarray: Array of shape (batch_size, embed_dim) containing pooled embeddings
            for each input waveform. Returns an empty array if no inputs are given.
    """

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
    pooled_batch = []

    for waveform in audio_tensors:
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        waveform = waveform.to(device)

        with torch.no_grad():
            melspec = feat_ext(waveform.float())      # (B, n_mels, n_frames)
            tokens = model(melspec)                   # (B, num_tokens, embed_dim)
            pooled = tokens.mean(dim=1)               # (B, embed_dim)

        pooled_batch.append(pooled.cpu().numpy())

    if pooled_batch:
        return np.vstack(pooled_batch)
    else:
        return np.empty((0, cfg.model.embed_dim))
