"""
motion2muscle Transformer model.

Matches checkpoint at motion2muscle/checkpoints/transformer_baseline_full/net_best_loss.pth

Architecture:
  model.0:          Conv1d(263, 256, kernel=3) + ReLU
  pos_encoder:      Sinusoidal PositionalEncoding (d_model=256, max_len=5000)
  transformer_encoder: 16 x TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=512)
  out_model.0:      Conv1d(256, 402, kernel=3)
  output:           Sigmoid -> [0,1]

Default parameters match the delivered checkpoint (width=256, nhead=8, num_layers=16).
"""
import math
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding matching checkpoint key 'pos_encoder.pe'."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (seq_len, batch, d_model)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class MotionToMuscleModel(nn.Module):
    """
    Transformer: HumanML3D motion (263-dim) -> muscle activations (402-dim).

    Input:  (batch, T, 263)
    Output: (batch, T, 402) in [0, 1]
    """

    def __init__(
        self,
        input_width: int = 263,
        output_width: int = 402,
        width: int = 256,
        nhead: int = 8,
        num_layers: int = 16,
    ):
        super().__init__()
        # Attribute names match checkpoint keys: 'model.0.weight', 'model.0.bias'
        blocks = [
            nn.Conv1d(input_width, width, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        ]
        self.model = nn.Sequential(*blocks)
        self.pos_encoder = PositionalEncoding(width)
        self.transformer_encoder = TransformerEncoder(
            TransformerEncoderLayer(
                d_model=width,
                nhead=nhead,
                dim_feedforward=width * 2,
                dropout=0.1,
            ),
            num_layers=num_layers,
        )
        # Attribute name matches checkpoint key: 'out_model.0.weight', 'out_model.0.bias'
        self.out_model = nn.Sequential(
            nn.Conv1d(width, output_width, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, 263) -> (B, 263, T)
        x = x.permute(0, 2, 1).float()
        # Conv1d over feature dim
        x = self.model(x)                                 # (B, width, T)
        # (B, width, T) -> (T, B, width) for Transformer
        x = x.permute(2, 0, 1)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)                   # (T, B, width)
        # (T, B, width) -> (B, width, T)
        x = x.permute(1, 2, 0)
        # Output conv
        x = self.out_model(x)                             # (B, 402, T)
        # (B, 402, T) -> (B, T, 402)
        x = x.permute(0, 2, 1)
        # [0, 1] range for muscle activations
        x = torch.sigmoid(x)
        return x


# Alias: loader.py's _default_model_builder() searches for this name
MotionToMuscleTransformer = MotionToMuscleModel
