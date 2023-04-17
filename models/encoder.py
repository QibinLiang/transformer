import torch as tr
import torch.nn as nn
from models.attentions import MultiHeadAttention, PositionwiseFeedForward


class TransformerEncoderBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_heads: int,
            d_ff: int,
            dropout=0.1
    ) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.ff = PositionwiseFeedForward(d_model, d_ff)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask=None) -> tr.Tensor:
        x = x + self.dropout_1(self.attention(x, x, x, mask))
        x = self.norm_1(x)
        x = x + self.dropout_2(self.ff(x))
        x = self.norm_2(x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_heads: int,
            d_ff: int,
            n_layers: int,
            dropout=0.1
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.n_layers = n_layers
        self.dropout = dropout
        self.residual_blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None) -> tr.Tensor:
        for i in range(self.n_layers):
            x = self.residual_blocks[i](x, mask)
        x = self.norm(x)
        return x
