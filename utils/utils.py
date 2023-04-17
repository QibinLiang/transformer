"""
Authors:
    * Qibin Liang 2023
"""

import logging
import json
import torch as tr
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

def load_mnt_data(path):
    data = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            data.append(json.loads(line))
    return data

# todo : add a function for tokenizing data at character-level when dict is not provided
class MNTDataset(Dataset):
    def __init__(self, path, src_dict=None, tgt_dict=None, pad_bos_eos=False):
        self.data = load_mnt_data(path)
        if src_dict is not None:
            self.src_dict = src_dict
        if tgt_dict is not None:
            self.tgt_dict = tgt_dict
        self.pad_bos_eos = pad_bos_eos
        if self.pad_bos_eos:
            assert self.tgt_dict.get('<sos/eos>', -1) != -1, \
                '<sos/eos> token not found in tgt_dict'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src = self.data[idx]['src']
        tgt = self.data[idx]['tgt']
        tgt_tokens = [self.tgt_dict.get(c, self.tgt_dict['<unk>']) for c in tgt]
        src_tokens = [self.src_dict.get(s, self.src_dict['<unk>']) for s in src]
        if self.pad_bos_eos:
            tgt_tokens = [self.tgt_dict['<sos/eos>']] + \
                            tgt_tokens + [self.tgt_dict['<sos/eos>']]
            src_tokens = [self.src_dict['<sos/eos>']] + \
                            src_tokens + [self.src_dict['<sos/eos>']]
        return {
            'tgt': tgt,
            'src': src,
            'tgt_lens': len(tgt_tokens),
            'src_lens': len(src_tokens),
            'tgt_tokens': tgt_tokens,
            'src_tokens': src_tokens
        }

# collate function for padding the data in a batch
def pad_collate_fn(batch):
    tgt = [item['tgt'] for item in batch]
    src = [item['src'] for item in batch]
    tgt_tokens = [tr.tensor(item['tgt_tokens'], dtype=tr.long) for item in batch]
    src_tokens = [tr.tensor(item['src_tokens'], dtype=tr.long) for item in batch]
    tgt_tokens = tr.nn.utils.rnn.pad_sequence(tgt_tokens, 
                                              batch_first=True, padding_value=0)
    src_tokens = tr.nn.utils.rnn.pad_sequence(src_tokens, 
                                              batch_first=True, padding_value=0)
    tgt_lens = tr.tensor([item['tgt_lens'] for item in batch], dtype=tr.long)
    src_lens = tr.tensor([item['src_lens'] for item in batch], dtype=tr.long)
    return {
        'tgt': tgt,
        'src': src,
        'tgt_tokens': tgt_tokens,
        'src_tokens': src_tokens,
        'tgt_lens': tgt_lens,
        'src_lens': src_lens
    }


def get_dataloader(
        path, 
        src_dict, 
        tgt_dict, 
        pad_bos_eos=True, 
        batch_size=32, 
        shuffle=True, 
        num_workers=0, 
        pin_memory=False,
        if_ddp=False
    ):
    dataset = MNTDataset(path, src_dict, tgt_dict, pad_bos_eos=pad_bos_eos)
    if if_ddp: 
        sampler = DistributedSampler(dataset, shuffle=shuffle)
    else:
        sampler = None
    dataloader = DataLoader(dataset, 
                            batch_size=batch_size, 
                            shuffle=shuffle and (not if_ddp), 
                            num_workers=num_workers,
                            collate_fn=pad_collate_fn, 
                            pin_memory=pin_memory, 
                            sampler=sampler)
    return dataloader


def load_dict(path):
    # read dict form aishell preprocessed data which is formed as:
    # token idx
    # token idx
    # ...
    def process_line(line):
        return list(line.rstrip().split(' '))
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        token2idx = {process_line(item)[0]: int(process_line(item)[1]) for item in lines}
        idx2token = {v: k for k, v in token2idx.items()}
    return token2idx, idx2token


class Logger():
    def __init__(self, logname=__name__, log_file=None, log_level=logging.INFO):
        self.log_file = log_file
        self.log_level = log_level
        self.logger = logging.getLogger(logname)
        self.logger.setLevel(self.log_level)
        self.formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        self._setup_stream_handler()
        if self.log_file is not None:
            self._setup_file_handler()
    
    def set_output_file(self, log_file):
        self.log_file = log_file
        self._setup_file_handler()

    def _setup_stream_handler(self):
        self.stream_handler = logging.StreamHandler()
        self.stream_handler.setLevel(self.log_level)
        self.stream_handler.setFormatter(self.formatter)
        self.logger.addHandler(self.stream_handler)

    def _setup_file_handler(self):
        self.file_handler = logging.FileHandler(self.log_file)
        self.file_handler.setLevel(self.log_level)
        self.file_handler.setFormatter(self.formatter)
        self.logger.addHandler(self.file_handler)

    def info(self, msg):
        self.logger.info(msg)
    
    def debug(self, msg):
        self.logger.debug(msg)

    def warning(self, msg):
        self.logger.warning(msg)
    
    def error(self, msg):
        self.logger.error(msg)
    
    def critical(self, msg):
        self.logger.critical(msg)
    
    def exception(self, msg):
        self.logger.exception(msg)
    
    def set_level(self, level):
        self.logger.setLevel(level)
        self.stream_handler.setLevel(level)
        if self.log_file is not None:
            self.file_handler.setLevel(level)

