import torch as tr
import torch.nn as nn
from models.attentions import RelPositionEmbedding
from models.decoder import TransformerDecoder
from models.encoder import TransformerEncoder
from utils.mask import make_seq_mask


class Transformer(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_heads: int,
            d_ff: int,
            n_layers: int,
            src_vocab_size: int,
            tgt_vocab_size: int,
            dropout: float = 0.1,
            blank_id: float = 0,
            bos_eos_id: int = -1,
            max_len: int = 5000,
            length_scale: float = 1.0,
        ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.n_layers = n_layers
        self.dropout = dropout
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.blank_id = blank_id
        self.bos_eos_id = bos_eos_id
        self.max_len = max_len
        self.length_scale = length_scale
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_emb = RelPositionEmbedding(d_model)
        self.encoder = TransformerEncoder(d_model, n_heads, 
                                        d_ff, n_layers, dropout=dropout)
        self.decoder = TransformerDecoder(d_model, n_heads, 
                                        d_ff, n_layers, dropout=dropout)
        self.proj = nn.Linear(d_model, tgt_vocab_size)
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, src_mask, tgt_mask):
        src_emb = self.src_emb(src)
        tgt_emb = self.tgt_emb(tgt)
        src_emb = self.pos_emb(src_emb) + src_emb
        tgt_emb = self.pos_emb(tgt_emb) + tgt_emb

        enc_out = self.encoder(src_emb, src_mask)
        dec_out = self.decoder(enc_out, tgt_emb, src_mask, tgt_mask)
        dec_out = self.proj(dec_out)
        dec_out = tr.log_softmax(dec_out, dim=-1)
        return dec_out

    def encode(self, src, src_mask):
        src_emb = self.src_emb(src)
        src_emb = self.pos_emb(src_emb) + src_emb
        enc_out = self.encoder(src_emb, src_mask)
        return enc_out

    # greedy decoding
    def decode(self, enc_out, enc_mask):
        B, L, N = enc_out.shape
        mem = tr.tensor([[self.bos_eos_id]]*B, dtype=tr.long, device=enc_out.device)
        max_len = min(int(L*self.length_scale), self.max_len)
        for t in range(max_len):
            tgt = mem
            dec_out = self.tgt_emb(tgt)  + self.pos_emb(self.tgt_emb(tgt))
            tgt_mask_t = make_seq_mask(tr.tensor(
                                        [t+1]*B, 
                                        dtype=tr.long, 
                                        device=enc_out.device))
            dec_out= self.decoder(
                            enc_out=enc_out, 
                            dec_out=dec_out, 
                            src_mask=enc_mask, 
                            tgt_mask=tgt_mask_t,)
            dec_out = self.proj(dec_out)
            dec_out = tr.log_softmax(dec_out, dim=-1)

            greedy_idx = tr.argmax(dec_out[:, -1, :], dim=-1, keepdim=True)
            mem = tr.cat([mem, greedy_idx], dim=1)
        return mem

    # todo : provide a reuslt of topk probabilities
    # todo : implement beam search
    def beam_search_decode(self, enc_out, enc_mask, beam_size=10):
        # if this funcion is called then throw an error.
        raise NotImplementedError("Beam search decoding is not implemented yet.")
        B, L, N = enc_out.shape
        enc_out = enc_out.repeat(B * beam_size, 1, 1)
        # ! for best probs
        topk_probs = [[0] * beam_size] * B
        # (B* beam_size, 1)
        mem = tr.tensor([[self.bos_eos_id]] * B * beam_size , 
                        dtype=tr.long, device=enc_out.device)
        max_len = min(int(L*self.length_scale), self.max_len)
        for t in range(max_len):
            #(B * beam_size, t)
            tgt = mem 
            dec_out = self.tgt_emb(tgt) + self.pos_emb(self.tgt_emb(tgt))
            tgt_mask_t = make_seq_mask(tr.tensor(
                                        [t+1]*B, 
                                        dtype=tr.long, 
                                        device=enc_out.device))
            # dec_out is shape of (B * beam_size, t, N)
            dec_out= self.decoder(
                            enc_out=enc_out, 
                            dec_out=dec_out, 
                            src_mask=enc_mask, 
                            tgt_mask=tgt_mask_t,) 
            # (B * beam_size, t, tgt_token_size)
            dec_out = self.proj(dec_out)
            dec_out = tr.log_softmax(dec_out, dim=-1) 

            # expand the memory to (B*beam_size, t+1)
            mem = tr.cat([mem, tr.zeros_like[dec_out[:, -1, :]]], dim=1)  
            for i in range(B):
                # get probabilities of the i_th data of the batch in the t_th step
                # (beam_size, 1, tgt_token_size)
                probs = dec_out[::B, t:, :]
                # (beam_size * tgt_token_size, 1)
                probs = probs.transpose(1,2).contiguous().view(-1,1)
                sorted_probs, sorted_ids = tr.sort(probs, dim=0)
                # (beam_size, 1)
                topk_ids = sorted_ids[:beam_size, :]
                # ! for best probs
                topk_probs = sorted_probs[:beam_size, :]
                predecessors, successor =  topk_ids // B, topk_ids % B
                # (beam_size, t)
                candidates = mem[::B, :]
                for _, (pred, suc) in enumerate(zip(predecessors, successor)):
                    candidates[_, :] = mem[pred*B, :] 
                    candidates[_, -1] = suc
                mem[::B, :] = candidates
        return mem
                    