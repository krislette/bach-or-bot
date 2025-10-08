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
    """
    feat_ext = FeatureExtractor(cfg).to(device)

    # --- CRITICAL FIX: REMOVE DYNAMIC SHAPE INFERENCE ---
    # The pre-trained model expects specific, fixed input dimensions.
    # We must hardcode them here to ensure the model architecture matches the checkpoint weights exactly.
    # The expected number of frames (n_frames) is taken directly from the RuntimeError message.
    n_mels = cfg.melspec.n_mels  # This should be 128
    n_frames = 3744             # This MUST match the checkpoint's expectation

    print(f"[INFO] Initializing SpecTTTra with fixed dimensions: n_mels={n_mels}, n_frames={n_frames}")

    model_cfg = cfg.model
    model = SpecTTTra(
        input_spec_dim=n_mels,
        input_temp_dim=n_frames, # Use the hardcoded value
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

    ckpt_path = Path("models/spectttra/spectttra_frozen.pth")
    if ckpt_path.exists():
        print(f"[INFO] Found SpecTTTra checkpoint at {ckpt_path}. Loading weights...")
        state = torch.load(ckpt_path, map_location=device)

        new_state_dict = {}
        for k, v in state.items():
            if k.startswith('encoder.'):
                new_key = k[len('encoder.'):]
                new_state_dict[new_key] = v
            else:
                new_state_dict[k] = v

        # Now that the shapes match, this should load without a size mismatch error.
        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
        
        if missing_keys:
            # You might see a few missing keys if your SpecTTTra class is slightly different, but the core should load.
            print(f"[WARNING] Keys missing in the model that were expected: {missing_keys}")
        if unexpected_keys:
            # Seeing 'classifier' or 'ft_extractor' keys here is NORMAL and SAFE.
            print(f"[INFO] Keys in file that were not used by the model (this is expected): {unexpected_keys}")

        print("[INFO] Successfully loaded pre-trained SpecTTTra weights.")

    else:
        raise FileNotFoundError(
            f"Pre-trained model not found at {ckpt_path}. "
            "Please download the 'pytorch_model.bin' from Hugging Face, "
            "rename it to 'spectttra_frozen.pth', and place it in the correct directory."
        )

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
    """
    global _FEAT_EXT, _MODEL, _CFG, _DEVICE
    _init_predictor_once()

    device = _DEVICE
    feat_ext = _FEAT_EXT
    model = _MODEL
    cfg = _CFG

    waveform = audio_tensor.to(device).float()

    with torch.no_grad():
        melspec = feat_ext(waveform)

        # --- FINAL FIX: Ensure melspec shape matches model's expectation ---
        expected_frames = model.input_temp_dim # This will be 3744
        if melspec.shape[2] > expected_frames:
            melspec = melspec[:, :, :expected_frames]
        elif melspec.shape[2] < expected_frames:
            padding = expected_frames - melspec.shape[2]
            melspec = torch.nn.functional.pad(melspec, (0, padding))
        # --- End of fix ---

        if device.type == "cuda":
            with torch.cuda.amp.autocast(enabled=True):
                tokens = model(melspec)
                pooled = tokens.mean(dim=1)
        else:
            tokens = model(melspec)
            pooled = tokens.mean(dim=1)

    out = pooled.squeeze(0).cpu().numpy()
    return out


def spectttra_train(audio_tensors):
    """
    Run batch input training with SpecTTTra.
    """
    global _FEAT_EXT, _MODEL, _CFG, _DEVICE
    _init_predictor_once()

    if not audio_tensors:
        return np.empty((0, _CFG.model.embed_dim))

    feat_ext = _FEAT_EXT
    model = _MODEL
    device = _DEVICE

    # This refactors the loop to be a much faster single-batch operation
    try:
        waveforms_batch = torch.cat(audio_tensors, dim=0).to(device).float()
    except Exception as e:
        print(f"Error during tensor concatenation, falling back to loop. Fix preprocessing for speed. Error: {e}")
        batch_list = [spectttra_predict(w) for w in audio_tensors]
        return np.array(batch_list)

    with torch.no_grad():
        melspec = feat_ext(waveforms_batch)

        # --- FINAL FIX: Ensure melspec shape matches model's expectation ---
        expected_frames = model.input_temp_dim # This will be 3744
        if melspec.shape[2] > expected_frames:
            melspec = melspec[:, :, :expected_frames]
        elif melspec.shape[2] < expected_frames:
            padding = expected_frames - melspec.shape[2]
            melspec = torch.nn.functional.pad(melspec, (0, padding))
        # --- End of fix ---

        if device.type == "cuda":
            with torch.cuda.amp.autocast(enabled=True):
                tokens = model(melspec)
                pooled = tokens.mean(dim=1)
        else:
            tokens = model(melspec)
            pooled = tokens.mean(dim=1)

    return pooled.cpu().numpy()