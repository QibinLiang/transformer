import torch as tr
from utils.utils import load_dict
from utils.mask import make_seq_mask
from utils.preprocessing import tokenize
from models.transformer import Transformer
import jieba
import logging

# turn off the logging
jieba.setLogLevel(logging.INFO)

def inference(
        model: Transformer, 
        src: tr.Tensor,
        src_lens: tr.Tensor,
        idx2token: dict,
    ):
    enc_mask = make_seq_mask(src_lens)
    enc_mask = enc_mask.to(src.device)
    src = src.to(src.device)
    enc_out = model.encode(src, enc_mask)
    dec_out = model.decode(enc_out, enc_mask)
    dec_out = dec_out.tolist()
    dec_out = [[idx2token[idx] for idx in  out] for out in dec_out]
    return dec_out

def init_model(
    d_model = 512,
    n_heads = 8,
    d_ff = 2048,
    n_layers = 6,
    dropout = 0.1,
):
    print("loading model...")
    src_tok2id, src_id2tok = load_dict(r"data/wmt/training-parallel-nc-v13/zhdict_char.txt")
    tgt_tok2id, tgt_id2tok = load_dict(r"data/wmt/training-parallel-nc-v13/endict_char.txt")
    src_vocab_size = len(src_tok2id.items())
    tgt_vocab_size = len(tgt_tok2id.items())

    model = Transformer(
        d_model, 
        n_heads, 
        d_ff, 
        n_layers, 
        src_vocab_size, 
        tgt_vocab_size, 
        dropout, 
        length_scale=3)
    model.load_state_dict(tr.load(r"ckpt/model.pt")['model'])
    model.eval()
    print("model loaded.")
    return model, src_tok2id, src_id2tok, tgt_tok2id, tgt_id2tok

def main(
        model, 
        src_tok2id, 
        src_id2tok, 
        tgt_tok2id, 
        tgt_id2tok):
    
    unk_id = src_tok2id['<unk>']
    sos_eos_id = tgt_tok2id['<sos/eos>']

    with tr.no_grad():
        while True:
            print("input:", end="\t")
            txt = input()
            if txt == "exit":
                break
            elif txt == "":
                continue
            tokens = tokenize(txt, "zh")
            src_lens = tr.tensor([len(tokens)])
            # convert tokens to ids
            src = []
            for token in tokens:
                token_id = src_tok2id.get(token, unk_id)
                if token_id == unk_id:
                    # if the token is not in the dictionary, then split it into characters
                    for char in token:
                        src.append(src_tok2id.get(char, unk_id))
                else:
                    src.append(token_id)
            #src = tr.tensor([[src_tok2id.get(src, unk_id) for src in tokens]])
            src = tr.tensor([src])
            src_lens = tr.tensor([len(src) for src in src.tolist()])
            
            model.bos_eos_id = sos_eos_id
            res = inference(model, src, src_lens, tgt_id2tok)
            for items in res:
                # obtain the tokens between the first <sos/eos> and the second <sos/eos>
                sent = []
                num_sos_eos = 0
                for i, item in enumerate(items):
                    if item == "<sos/eos>":
                        num_sos_eos += 1
                        if num_sos_eos == 2:
                            break
                    else:
                        sent.append(item)
                print("output:\t"+" ".join(sent))

if __name__ == "__main__":
    model, src_tok2id, src_id2tok, tgt_tok2id, tgt_id2tok = init_model()
    main(model, src_tok2id, src_id2tok, tgt_tok2id, tgt_id2tok)