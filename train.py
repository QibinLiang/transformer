import os
import argparse

import torch as tr
import torch.distributed as dist
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from models.transformer import Transformer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.mask import get_joint_mask, make_seq_mask
from utils.utils import load_dict, get_dataloader, Logger

import tensorboardX as tbx

# todo : add description for args
# argparser
def get_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--src_dict", type=str, 
                           default=r"data/wmt/")
    argparser.add_argument("--tgt_dict", type=str, 
                           default=r"data/wmt/")
    argparser.add_argument("--train_data", type=str, 
                           default=r"data/wmt/")
    argparser.add_argument("--dev_data", type=str, 
                           default=r"data/wmt/")
    argparser.add_argument("--token_level", type=str, default="char")
    argparser.add_argument("--batch_size", type=int, default=12)
    argparser.add_argument("--epochs", type=int, default=10)
    # optimizer setting
    argparser.add_argument("--lr", type=float, default=1e-4)
    argparser.add_argument("--min_lr", type=float, default=1e-5)
    argparser.add_argument("--weight_decay", type=float, default=1e-5)
    # model setting
    argparser.add_argument("--d_model", type=int, default=512)
    argparser.add_argument("--n_layers", type=int, default=6)
    argparser.add_argument("--n_heads", type=int, default=8)
    argparser.add_argument("--dropout", type=float, default=0.1)
    argparser.add_argument("--d_ff", type=int, default=2048)
    argparser.add_argument("--max_len", type=int, default=5000)
    argparser.add_argument("--length_scale", type=float, default=2)
    # training setting
    argparser.add_argument("--accumulation_steps", type=int, default=16)
    argparser.add_argument("--warmup_steps", type=int, default=4000)
    argparser.add_argument("--max_grad_norm", type=float, default=1.0)
    argparser.add_argument("--log_dir", type=str, default=r"logs")
    # verbose interval
    argparser.add_argument("--verbose_interval", type=int, default=100)
    # save model
    argparser.add_argument("--save_path", type=str, default=None)
    argparser.add_argument("--save_interval", type=int, default=4)
    # if use cuda
    argparser.add_argument("--use_cuda", type=bool, default=True)
    # load model
    argparser.add_argument("--load_model", type=str, default=None)
    # if use ddp
    argparser.add_argument("--use_ddp", type=bool, default=False)
    argparser.add_argument("--rank", type=int, default=0)
    argparser.add_argument("--world_size", type=int, default=1)
    # ddp data communication backend
    argparser.add_argument("--backend", type=str, default="nccl")
    # init method
    argparser.add_argument("--init_method", type=str, default="file://ckpt/init_method")
    return argparser.parse_args()


def trainer(
        model: Transformer,
        optimizer: tr.optim.Optimizer,
        scheduler: tr.optim.lr_scheduler.ReduceLROnPlateau, # todo : change to `_LRScheduler` based scheduler
        criterion: tr.nn.Module,
        train_loader: tr.utils.data.DataLoader,
        dev_loader: tr.utils.data.DataLoader,
        epochs: int,
        summary_writer: tbx.SummaryWriter = None,
        device: tr.device = tr.device("cuda" if tr.cuda.is_available() else "cpu"), 
        ckpt: str = None,
        save_interval: int = 4,
        if_ddp:bool = False,
        rank: int = 0,
        accumulation_steps: int = 16,
        verbose_interval: int = 100,
        token_level: str = "char",
    ):

    log = Logger(__name__)

    if ckpt is not None:
        #ckpt = tr.load(ckpt_path, map_location=device)
        #model.load_state_dict(ckpt['model'])
        total_steps = ckpt['total_steps']
        start_epochs = ckpt['epoch']
    else:
        total_steps = 0
        start_epochs = 0

    if not if_ddp:
        model = model.to(device)
    model.train()
    
    total_loss = 0
    dev_step = 0
    dev_loss = 0

    for epoch in range(start_epochs, epochs):
        # train step
        for batch in train_loader:
            
            tgt = batch['tgt']
            src = batch['src']
            tgt = batch['tgt_tokens']
            src = batch['src_tokens']
            tgt_lens = batch['tgt_lens']
            src_lens = batch['src_lens']
            src_mask = make_seq_mask(src_lens)
            tgt_mask = get_joint_mask(tgt_lens-1)

            src = src.to(device)
            tgt = tgt.to(device)
            src_mask = src_mask.to(device)
            tgt_mask = tgt_mask.to(device)

            out = model(src, tgt[:, :-1], src_mask, tgt_mask)
            loss = criterion(out.transpose(1,2), tgt[:, 1:])
            loss.backward()
            if total_steps % accumulation_steps  == (accumulation_steps -1):
                optimizer.step()
                optimizer.zero_grad()
            total_steps += 1
            total_loss += loss.item()
            if total_steps % verbose_interval == 0 and rank == 0:
                log.info(f"train - epoch: {epoch} - step: {total_steps} - loss: {total_loss/verbose_interval}")
                if summary_writer is not None:
                    summary_writer.add_scalar("train_loss", total_loss/verbose_interval, total_steps)
                total_loss = 0
        
        # dev step
        with tr.no_grad():
            model.eval()
            avg_dev_loss = 0
            current_dev_step = 0
            for batch in dev_loader:
                tgt = batch['tgt']
                src = batch['src']
                tgt = batch['tgt_tokens']
                src = batch['src_tokens']
                tgt_lens = batch['tgt_lens']
                src_lens = batch['src_lens']
                src_mask = make_seq_mask(src_lens)
                tgt_mask = get_joint_mask(tgt_lens-1)

                src = src.to(device)
                tgt = tgt.to(device)
                src_mask = src_mask.to(device)
                tgt_mask = tgt_mask.to(device)

                out = model(src, tgt[:, :-1], src_mask, tgt_mask)
                loss = criterion(out.transpose(1,2), tgt[:, 1:])
                dev_step += 1
                current_dev_step += 1
                dev_loss += loss.item()
                avg_dev_loss += loss.item()
                if dev_step % verbose_interval == 0 and rank == 0:
                    log.info(f"dev - epoch: {epoch} - step: {dev_step} - loss: {dev_loss/verbose_interval}")
                    if summary_writer is not None:
                        summary_writer.add_scalar("dev_loss", dev_loss/verbose_interval, dev_step)
                    dev_loss = 0
            
            avg_dev_loss /= current_dev_step
            # if use ddp, all_reduce the loss
            if if_ddp:
                avg_dev_loss = tr.tensor(avg_dev_loss, device=device)
                dist.all_reduce(avg_dev_loss, op=dist.ReduceOp.SUM)
                avg_dev_loss /= dist.get_world_size()
            log.info(f"dev - epoch: {epoch} - loss: {avg_dev_loss}")
                # update the learning rate
            if scheduler is not None:
                scheduler.step(avg_dev_loss)
            model.train()
        # save the model for every 4 epochs
        if rank==0:
            if epoch % save_interval == 0:
                chpt = {
                    "model": model.state_dict() if \
                        not if_ddp else model.module.state_dict(),
                    "epoch": epoch,
                    "total_steps": total_steps,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if \
                        scheduler is not None else None,
                }
                tr.save(chpt, f"ckpt/model_{token_level}_{epoch}.pt")

def main(args):
    src_dict = os.path.join(args.src_dict, "zh_dict_"+args.token_level+".txt")
    tgt_dict = os.path.join(args.tgt_dict, "en_dict_"+args.token_level+".txt")
    train_data = os.path.join(args.train_data, "train_"+args.token_level+".json")
    dev_data = os.path.join(args.dev_data, "dev_"+args.token_level+".json")
    
    # if `./ckpt` not exists, create it
    if not os.path.exists('ckpt'):
        os.makedirs('ckpt')

    src_tok2id, src_id2tok = load_dict(src_dict)
    tgt_tok2id, tgt_id2tok = load_dict(tgt_dict)
    src_vocab_size = len(src_tok2id.items())
    tgt_vocab_size = len(tgt_tok2id.items())
    d_model = args.d_model
    n_heads = args.n_heads
    d_ff = args.d_ff
    n_layers = args.n_layers
    dropout = args.dropout
    bs = args.batch_size
    lr = args.lr
    epochs = args.epochs
    max_len = args.max_len
    length_scale = args.length_scale

    # init the model
    model = Transformer(
        d_model, 
        n_heads, 
        d_ff, 
        n_layers, 
        src_vocab_size, 
        tgt_vocab_size, 
        dropout, 
        bos_eos_id=tgt_tok2id['<sos/eos>'],
        max_len=max_len,
        length_scale=length_scale)
    criterion = tr.nn.CrossEntropyLoss()
    writer = tbx.SummaryWriter(f"tensorboard/{args.token_level}")
    # load the model from the checkpoint
    if args.load_model is not None:
        ckpt = tr.load(args.load_model, map_location=tr.device("cpu"))
        model.load_state_dict(ckpt['model'])
    else:
        ckpt = None

    # init the ddp
    if args.use_ddp:
        # init the distributed backend
        init_process_group(
            backend=args.backend,
            init_method=args.init_method,
            world_size=args.world_size,
            rank=args.rank,
        )
        # init the model
        device = tr.device("cuda:"+ str(args.rank))
        model.to(device)
        model = DDP(model, find_unused_parameters=True)
    else:
        device = tr.device("cuda" if tr.cuda.is_available() else "cpu")

    train_dataloader = get_dataloader(train_data, 
                                src_tok2id, 
                                tgt_tok2id, 
                                batch_size=bs, 
                                if_ddp=args.use_ddp)
    
    dev_dataloader = get_dataloader(dev_data, 
                                src_tok2id, 
                                tgt_tok2id, 
                                batch_size=bs, 
                                if_ddp=args.use_ddp)

    optimizer = tr.optim.Adam(
        model.parameters(), 
        lr=lr, 
        weight_decay=args.weight_decay)
    
    if args.lr > args.min_lr:
        scheduler = ReduceLROnPlateau(
            optimizer=optimizer,
            min_lr=args.min_lr, 
            patience=5, 
            factor=0.5)
    else:
        scheduler = None
    trainer(model, 
            optimizer,
            scheduler, 
            criterion, 
            train_dataloader, 
            dev_dataloader,
            epochs,
            summary_writer=writer, 
            ckpt=ckpt, 
            save_interval=args.save_interval,
            device=device,
            if_ddp=args.use_ddp,
            rank=args.rank,
            accumulation_steps=args.accumulation_steps,
            verbose_interval=args.verbose_interval,
            token_level=args.token_level)

if __name__ == "__main__":
    args = get_args()
    main(args)
