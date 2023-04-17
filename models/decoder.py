import torch as tr
import torch.nn as nn
from models.attentions import MultiHeadAttention, PositionwiseFeedForward


class TransformerDecoderBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_heads: int,
            d_ff: int,
            dropout=0.1
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.masked_attention = MultiHeadAttention(d_model, n_heads)
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.norm_3 = nn.LayerNorm(d_model)
        self.ff = PositionwiseFeedForward(d_model, d_ff)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

    def forward(
            self,
            enc_out: tr.Tensor,
            dec_out: tr.Tensor,
            enc_mask: tr.Tensor,
            dec_mask: tr.Tensor,
    ) -> tr.Tensor:
        """

        Args:
            enc_out (tr.Tensor): [batch_size, src_len, d_model]
            dec_out (tr.Tensor): [batch_size, tgt_len, d_model] 
                the `dec_out` is the token embedding during the training stage 
                (i.e. embedding of `<bos> hello world <eos>`), and the `dec_out` 
                is the output of the decoder during the inference stage.
            enc_mask (tr.Tensor, optional): [batch_size, 1, src_len].
            dec_mask (tr.Tensor, optional): [batch_size, tgt_len, tgt_len].
            cache (tr.Tensor, optional): [batch_size, tgt_len-1, d_model]. Defaults to None.
        
        Returns:
            tr.Tensor: [batch_size, tgt_len, d_model]
        """
        q = dec_out + self.dropout_1(
            self.masked_attention(dec_out, dec_out, dec_out, dec_mask)) 
        q = self.norm_1(q)
        x = q + self.dropout_2(
            self.attention(q, enc_out, enc_out, enc_mask))
        x = self.norm_2(x)
        x = x + self.dropout_3(self.ff(x))
        x = self.norm_3(x)
        return x


class TransformerDecoder(nn.Module):
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
            TransformerDecoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])

    def forward(
            self, 
            enc_out: tr.Tensor, 
            dec_out: tr.Tensor, 
            src_mask: tr.Tensor, 
            tgt_mask: tr.Tensor
        ):
        """
        Args:
            enc_out (tr.Tensor): [batch_size, src_len, d_model]
            dec_out (tr.Tensor): [batch_size, tgt_len, d_model]
                the `dec_out` is the token embedding during the training stage
                (i.e. embedding of `<bos> hello world <eos>`), and the `dec_out`
                is the output of the decoder during the inference stage.
            enc_mask (tr.Tensor, optional): [batch_size, 1, src_len].
            dec_mask (tr.Tensor, optional): [batch_size, tgt_len, tgt_len]
            caches (List, optional): [n_layers, batch_size, tgt_len-1, d_model]

        Returns:
            tr.Tensor: [batch_size, tgt_len, token_size]
        """
        for _, layer in enumerate(self.residual_blocks):
            dec_out = layer(enc_out, dec_out, src_mask, tgt_mask)
        return dec_out
    