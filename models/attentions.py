import torch as tr
import torch.nn as nn

# todo : dynamically check the shape of the mask and reshape the mask to adapt the shape of the attention score
# todo : refactor the code to decouple the steps of computing the attention mask and the attention score
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        q = self.w_q(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)
        scores = tr.matmul(q, k.transpose(-2, -1)) / tr.sqrt(tr.tensor(self.d_k, dtype=tr.float))
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1), -1e9)
        scores = self.softmax(scores)
        # ? is this dropout necessary? Does the dropout discrad the attention from some tokens to prevent overfitting?
        scores = self.dropout(scores)
        output = tr.matmul(scores, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.w_o(output)
        return output
    
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.w_1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.w_2(x)
        return x