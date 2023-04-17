# An implementation of Transformer🤖 from the scratch

This project implements the Transformer architecture from the scratch as well as applies the Transformer to the machine translation. The project aims to construct the Transformer in a straightforward way to help me better understand how the Transformer works. I found that learning AI architecture from the paper doesn't really help me to fully understand the paper's idea. There is always something trivial but important behind the idea but never detailed in the paper, which makes me feel there is a huge gap between the paper and the practice🤨. So I create this repo to help me understand the Transformer and its variants, also hope this repo can help someone who is suffering from the Transformer.🤗

## Quickstart 🚀

train on a single GPU machine
```bash
./run_single.sh
```

parallel train on multi-GPUs 
```bash
./prepare.sh
./run.sh
```

parallel train on multi-GPUs by Slurm
```bash
./prepare.sh
./run_slurm.sh
```

inference
```bash
$ python inference.py 
loading model...
model loaded.
input:    人工智能并不能帮我们解决所有的问题
output:    ai does not help us solve all problems in our bodies
input:    人工智能解决一些问题    
output:    ai will solve some problems
input:    人工只能解决一些问题
output:    workers only do something about it
input:    exit
```

## Todo list 📝
- [x] Transformer
- [x] checkpointer
- [x] distributed training
- [ ] beam search
- [ ] prefix beam search
- [ ] evaluate step
- [ ] test step
- [ ] yaml config
- [ ] perplexity

----
## Reference 📚
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)