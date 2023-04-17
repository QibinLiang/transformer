"""
Authors:
    * Qibin Liang 2022
"""
import torch 

# todo: all the mask's dimension should not be expanded

def make_subsequent_mask(
    seq_size, 
    return_bool=True, 
    device=torch.device('cpu'), 
    dtype=torch.int64
    ):
    """ make subsequent mask for the input lengths
    Args:
        seq_size (int): batch size
    Returns:
        torch.Tensor: mask tensor

    Example:
        >>> subsequent_mask(seq_size=2)
        tensor([[1, 0],
                [1, 1]], dtype=torch.int32)
    """
    mask = torch.tril(torch.ones(
            (seq_size, seq_size), 
            device=device, 
            dtype=dtype
        ), diagonal=0)
    if return_bool:
        mask = ~mask.bool()
    return mask.unsqueeze(0)

def make_subsequent_chunk_mask(
    seq_size, 
    chunk_size, 
    num_left_chunks=0, 
    return_bool=True, 
    device=torch.device("cpu")
    ):
    """ make subsequent chunk mask for the input lengths
    Args:
        seq_size (int): batch size
        chunk_size (int): chunk size
        num_left_chunks (int): number of left chunks
        device (torch.device): device

    Returns:
        torch.Tensor: mask tensor

    """
    mask = torch.zeros(seq_size, seq_size, dtype=torch.int32, device=device)
    for i in range(seq_size):
        tail = (i // chunk_size + 1) * chunk_size
        if num_left_chunks > 0:
            mask[i, num_left_chunks : tail] = 1
        else:
            mask[i, 0: tail] = 1
    if return_bool:
        mask = ~mask.bool()
    return mask.unsqueeze(0)

def make_seq_mask(lengths, return_bool=True):
    """ make mask for the input lengths
    Args:
        lengths (torch.Tensor): tensor of lengths
    Returns:
        torch.Tensor: mask tensor

    Example:
        >>> lengths = torch.tensor([2,1,3,8], dtype=torch.int64)
        >>> make_mask(lengths=lengths)
        tensor([[1, 1, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1]], dtype=torch.int32)
    """
    assert len(lengths.shape) == 1 

    max_len = torch.max(lengths).long().item()
    mask = torch.arange(
        0, max_len,).expand(
            lengths.shape[0], 
            max_len).to(lengths.device) < lengths.unsqueeze(1)
    mask = torch.as_tensor(mask, device=lengths.device, dtype=lengths.dtype)
    if return_bool:
        mask = ~mask.bool().to(lengths.device)
    return mask.unsqueeze(1)

def joint_subseq_seq_mask(length_mask, seq_mask):
    return length_mask | seq_mask

def get_joint_mask(lens):
    len_mask = make_subsequent_mask(lens.max(), lens.device)
    seq_mask = make_seq_mask(lens)
    return joint_subseq_seq_mask(len_mask, seq_mask)