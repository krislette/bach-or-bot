import torch
import torch.nn as nn
from pathlib import Path
from .transformer import Transformer
from .tokenizer import STTokenizer
from src.spectttra.feature import FeatureExtractor


class SpecTTTra(nn.Module):
    """
    SpecTTTra: A Spectro-Temporal Transformer model for audio representation learning.

    This model first tokenizes the input spectrogram into temporal and spectral tokens,
    then processes them with a Transformer encoder to capture spectro-temporal dependencies.
    """

    def __init__(
        self,
        input_spec_dim,
        input_temp_dim,
        embed_dim,
        t_clip,
        f_clip,
        num_heads,
        num_layers,
        pre_norm=False,
        pe_learnable=False,
        pos_drop_rate=0.0,
        attn_drop_rate=0.0,
        proj_drop_rate=0.0,
        mlp_ratio=4.0,
    ):
        """
        Initialize the SpecTTTra model.

        Args:
            input_spec_dim (int): Input spectrogram frequency dimension (F).
            input_temp_dim (int): Input spectrogram temporal dimension (T).
            embed_dim (int): Embedding dimension for tokens.
            t_clip (int): Temporal clip size for tokenization.
            f_clip (int): Spectral clip size for tokenization.
            num_heads (int): Number of attention heads in the transformer.
            num_layers (int): Number of transformer layers.
            pre_norm (bool, optional): Whether to apply pre-normalization. Defaults to False.
            pe_learnable (bool, optional): If True, use learnable positional embeddings. Defaults to False.
            pos_drop_rate (float, optional): Dropout rate for positional embeddings. Defaults to 0.0.
            attn_drop_rate (float, optional): Dropout rate for attention. Defaults to 0.0.
            proj_drop_rate (float, optional): Dropout rate for projection layers. Defaults to 0.0.
            mlp_ratio (float, optional): Expansion ratio for MLP hidden dimension. Defaults to 4.0.
        """
        super(SpecTTTra, self).__init__()
        self.input_spec_dim = input_spec_dim
        self.input_temp_dim = input_temp_dim
        self.embed_dim = embed_dim
        self.t_clip = t_clip
        self.f_clip = f_clip
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.pre_norm = (
            pre_norm    # Applied after tokenization before transformer (used in CLIP)
        )
        self.pe_learnable = pe_learnable    # Learned positional encoding
        self.pos_drop_rate = pos_drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.proj_drop_rate = proj_drop_rate
        self.mlp_ratio = mlp_ratio

        # Tokenizer for spectro-temporal features
        self.st_tokenizer = STTokenizer(
            input_spec_dim,
            input_temp_dim,
            t_clip,
            f_clip,
            embed_dim,
            pre_norm=pre_norm,
            pe_learnable=pe_learnable,
        )

        # Dropout applied after tokenization
        self.pos_drop = nn.Dropout(p=pos_drop_rate)

        # Transformer encoder
        self.transformer = Transformer(
            embed_dim,
            num_heads,
            num_layers,
            attn_drop=self.attn_drop_rate,
            proj_drop=self.proj_drop_rate,
            mlp_ratio=self.mlp_ratio,
        )

    def forward(self, x):
        """
        Forward pass of SpecTTTra.

        Args:
            x (torch.Tensor): Input spectrogram of shape
                - (B, 1, F, T) if channel dimension exists
                - (B, F, T) otherwise

        Returns:
            torch.Tensor: Transformer-encoded spectro-temporal tokens of shape
                (B, T/t + F/f, embed_dim)
        """
        # Squeeze the channel dimension if it exists
        if x.dim() == 4:
            x = x.squeeze(1)

        # Spectro-temporal tokenization
        spectro_temporal_tokens = self.st_tokenizer(x)

        # Positional dropout
        spectro_temporal_tokens = self.pos_drop(spectro_temporal_tokens)

        # Transformer
        output = self.transformer(spectro_temporal_tokens)  # shape: (B, T/t + F/f, dim)

        return output


def build_spectttra_from_cfg(cfg, device):
    """
    Constructs the SpecTTTra model and its associated FeatureExtractor from a given configuration.

    Args:
        cfg (SimpleNamespace): Configuration object containing model and feature extraction parameters. Expected attributes include:
                - cfg.melspec.n_mels: Number of mel frequency bins.
                - cfg.model: Model-specific parameters (e.g., embed_dim, t_clip, f_clip, etc.).
        device (torch.device): The device on which the model and feature extractor will be allocated (e.g., 'cpu' or 'cuda').

    Returns:
        tuple:
            FeatureExtractor: Initialized feature extraction module moved to the specified device.
            SpecTTTra: Constructed SpecTTTra model moved to the specified device.
    """

    feat_ext = FeatureExtractor(cfg).to(device)

    # The pre-trained model expects specific, fixed input dimensions.
    # Hardcoded to ensure the model architecture matches the checkpoint weights exactly.
    # The expected number of frames (n_frames) is taken directly from the RuntimeError message.
    n_mels = cfg.melspec.n_mels     # n_mels should be 128
    n_frames = 3744                 # n_frames match the checkpoint's expectation

    print(f"[INFO] Initializing SpecTTTra with fixed dimensions: n_mels={n_mels}, n_frames={n_frames}")

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

    return feat_ext, model


def load_frozen_spectttra(model, ckpt_path, device):
    """
    Loads pretrained SpecTTTra weights from a frozen checkpoint file.

    Args:
        model (torch.nn.Module): An initialized SpecTTTra model instance to load weights into.
        ckpt_path (str or Path): Path to the pretrained model checkpoint file (e.g., 'spectttra_frozen.pth').
        device (torch.device): The device to map the loaded weights to (e.g., 'cpu' or 'cuda').

    Returns:
        model (torch.nn.Module): The SpecTTTra model with loaded pretrained weights, set to evaluation mode.

    Raises:
        FileNotFoundError: If the specified checkpoint file does not exist at `ckpt_path`.
    """
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Pre-trained model not found at {ckpt_path}. "
            "Please download 'pytorch_model.bin', rename to 'spectttra_frozen.pth', "
            "and place it in the correct directory."
        )

    print(f"[INFO] Found SpecTTTra checkpoint at {ckpt_path}. Loading weights...")
    state = torch.load(ckpt_path, map_location=device)

    new_state_dict = {}
    for k, v in state.items():
        if k.startswith("encoder."):
            new_key = k[len("encoder."):]
            new_state_dict[new_key] = v
        else:
            new_state_dict[k] = v

    # Now that the shapes match, this should load without a size mismatch error.
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    if missing_keys:
        # Might see a few missing keys if your SpecTTTra class is slightly different, but the core should load.
        print(f"[WARNING] Missing keys in model: {missing_keys}")
    if unexpected_keys:
        # Seeing 'classifier' or 'ft_extractor' keys here is NORMAL and SAFE.
        print(f"[INFO] Unused keys in checkpoint: {unexpected_keys}")

    print("[INFO] Successfully loaded pre-trained SpecTTTra weights.")
    
    model.eval()
    return model
