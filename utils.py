import os
import random

import numpy as np
import torch


# recommended to seed from 42 to 45
def global_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_available_device():
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'

    return device


def prepare_padded_seq(seq, max_seq_len):
    padded_seq = np.zeros([max_seq_len], dtype=np.int32)
    if len(seq) < max_seq_len:
        padded_seq[-len(seq):] = seq
    else:
        padded_seq[:] = seq[-max_seq_len:]

    return padded_seq


def sample_negative(seq, num_item):
    while True:
        item = random.randint(1, num_item)
        if item not in seq:
            return item


def batch_sample_negative(seq, num_item, neg_num):
    closed_set = set()
    while len(closed_set) < neg_num:
        closed_set.add(sample_negative(seq, num_item))

    return list(closed_set)
