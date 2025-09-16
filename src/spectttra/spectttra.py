import torch.nn as nn
from .transformer import Transformer
from .tokenizer import STTokenizer


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