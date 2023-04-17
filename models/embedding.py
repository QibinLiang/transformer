import torch as tr
import torch.nn as nn

class RelPositionEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.pos_emb.weight.data = self.get_pos_emb()

    def get_pos_emb(self):
        pos_emb = tr.zeros(self.max_len, self.d_model)
        pos = tr.arange(0, self.max_len, dtype=tr.float).unsqueeze(1)
        div_term = tr.exp(tr.arange(0, self.d_model, 2).float() * (-tr.log(tr.tensor(10000.0)) / self.d_model))
        pos_emb[:, 0::2] = tr.sin(pos * div_term)
        pos_emb[:, 1::2] = tr.cos(pos * div_term)
        return pos_emb

    def forward(self, x):
        seq_len = x.size(1)
        pos = tr.arange(0, seq_len, dtype=tr.long, device=x.device)
        pos = pos.unsqueeze(0).expand(x.size(0), -1)
        pos_emb = self.pos_emb(pos)
        return pos_emb