import numpy as np
import torch

from utils import prepare_padded_seq


def fourier_encoding(ts_seq, d, min_ts, angular_frequency):
    start_idx = len(ts_seq) - np.count_nonzero(ts_seq)
    ts_encoding = np.zeros((len(ts_seq), d), dtype='f')

    # temporal encodings
    for i in range(start_idx, len(ts_seq)):
        ts_gap = ts_seq[i] - min_ts
        basic_angular = ts_gap * angular_frequency
        angular_vector = np.array([j * basic_angular for j in range(1, 1 + d // 2)])
        # odd positions
        ts_encoding[i, 0::2] = np.sin(angular_vector)
        # even positions
        ts_encoding[i, 1::2] = np.cos(angular_vector)

    return ts_encoding


def temporal_encoding(ts_seq, d, min_ts, angular_frequency, pow_base=10000):
    start_idx = len(ts_seq) - np.count_nonzero(ts_seq)
    ts_encoding = np.zeros((len(ts_seq), d), dtype='f')

    # temporal encodings
    for i in range(start_idx, len(ts_seq)):
        ts_gap = ts_seq[i] - min_ts
        basic_angular = ts_gap * angular_frequency
        angular_vector = np.array([pow(pow_base, j / (d - 2)) * basic_angular for j in range(d // 2)])
        # odd positions
        ts_encoding[i, 0::2] = np.sin(angular_vector)
        # even positions
        ts_encoding[i, 1::2] = np.cos(angular_vector)

    return ts_encoding


class TemporalEncodingParser:
    def __init__(self, config):
        self.is_bert = config.is_bert
        self.max_seq_len = config.max_seq_len
        self.d_temporal = config.d_temporal
        self.min_ts = config.min_ts
        self.angular_frequency = config.angular_frequency

    def prepare_train_inputs(self, inputs):
        if self.is_bert:
            return self._prepare_train_bert(inputs)
        else:
            return self._prepare_train_left_wise(inputs)

    def prepare_predict_inputs(self, inputs):
        user, seq, ts, item_indices = inputs

        # paddings
        seq = prepare_padded_seq(seq, self.max_seq_len)
        ts = prepare_padded_seq(ts, self.max_seq_len)
        item_indices = np.array(item_indices)

        # sequence mask
        seq_masks = torch.BoolTensor(seq == 0)

        # temporal encoding
        ts_encoding = temporal_encoding(ts, self.d_temporal, self.min_ts, self.angular_frequency)

        return torch.LongTensor(seq), torch.Tensor(ts_encoding), seq_masks, torch.LongTensor(item_indices)

    def _prepare_train_bert(self, inputs):
        user, seq, ts, pos, neg = inputs

        # paddings
        seq = prepare_padded_seq(seq, self.max_seq_len)
        ts = prepare_padded_seq(ts, self.max_seq_len)
        labels = prepare_padded_seq(pos, self.max_seq_len)

        # sequence mask
        seq_masks = torch.BoolTensor(seq == 0)

        # temporal encoding
        ts_encoding = temporal_encoding(ts, self.d_temporal, self.min_ts, self.angular_frequency)

        return torch.LongTensor(seq), torch.Tensor(ts_encoding), seq_masks, torch.LongTensor(labels)

    def _prepare_train_left_wise(self, inputs):
        user, seq, ts, pos, neg = inputs

        # paddings
        seq = prepare_padded_seq(seq, self.max_seq_len)
        ts = prepare_padded_seq(ts, self.max_seq_len)
        pos = prepare_padded_seq(pos, self.max_seq_len)
        neg = prepare_padded_seq(neg, self.max_seq_len)

        # sequence mask
        seq_masks = torch.BoolTensor(seq == 0)

        # temporal encoding
        ts_encoding = temporal_encoding(ts, self.d_temporal, self.min_ts, self.angular_frequency)

        return torch.LongTensor(seq), torch.Tensor(ts_encoding), seq_masks, torch.LongTensor(pos), torch.LongTensor(neg)
