import torch
import torch.nn as nn


class SinusoidPositionalEncoding(nn.Module):
    """
    Applies sinusoidal positional encoding to token embeddings.

    This encoding is deterministic and injects information about 
    token positions into embeddings using sine and cosine functions.

    Args:
        token_dim (int): Dimensionality of each token embedding.
        max_len (int, optional): Maximum sequence length supported. Defaults to 5000.
    """

    def __init__(self, token_dim, max_len=5000):
        super(SinusoidPositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, token_dim)          # shape: (max_len, token_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(
            1
        )                                             # shape: (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, token_dim, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / token_dim)
        )                                             # shape: (token_dim // 2)
        pe[:, 0::2] = torch.sin(position * div_term)  # shape: (max_len, token_dim // 2)
        pe[:, 1::2] = torch.cos(position * div_term)  # shape: (max_len, token_dim // 2)
        pe = pe.unsqueeze(0)                          # shape: (1, max_len, token_dim)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Forward pass of sinusoidal positional encoding.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, token_dim).

        Returns:
            torch.Tensor: Tensor with positional encodings added, 
                          shape (batch_size, seq_len, token_dim).
        """
        x = x + self.pe[:, : x.size(1), :]            # shape: (batch_size, seq_len, token_dim)
        return x


class LearnedPositionalEncoding(nn.Module):
    """
    Applies learned positional encoding to token embeddings.

    Unlike sinusoidal encoding, positional embeddings are trainable 
    parameters learned during model training.

    Args:
        token_dim (int): Dimensionality of each token embedding.
        num_tokens (int): Maximum sequence length supported.
    """

    def __init__(self, token_dim, num_tokens):
        super(LearnedPositionalEncoding, self).__init__()
        self.pe = nn.Parameter(torch.randn(1, num_tokens, token_dim) * 0.02)

    def forward(self, x):
        """
        Forward pass of learned positional encoding.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, token_dim).

        Returns:
            torch.Tensor: Tensor with learned positional encodings added, 
                          shape (batch_size, seq_len, token_dim).
        """
        x = x + self.pe
        return x
