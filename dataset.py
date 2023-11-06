import collections
import os
import random

from torch.utils.data import Dataset

from utils import sample_negative, batch_sample_negative


def load_user_seqs(dp):
    user2seq_map = collections.defaultdict(list)
    user2ts_map = collections.defaultdict(list)

    for file in os.listdir(dp):
        fp = dp + file if dp.endswith('/') else dp + '/' + file
        with open(fp) as f:
            for line in f:
                user, item, ts = line.rstrip().split(',')
                user, item, ts = int(user), int(item), int(ts)
                user2seq_map[user].append(item)
                user2ts_map[user].append(ts)

    return user2seq_map, user2ts_map


def partition_dataset(dp, tr=3, test_ratio=0.5):
    user2seq_map, user2ts_map = load_user_seqs(dp)

    num_user = len(user2seq_map)
    num_item = max([max(seq) for seq in user2seq_map.values()])

    # users = list(user2seq_map.keys())
    pred_users = {user for user, seq in user2seq_map.items() if len(seq) >= tr}
    test_users = set(random.sample(pred_users, k=int(len(pred_users) * test_ratio)))
    valid_users = pred_users.difference(test_users)

    history_user2seq_map = dict()
    history_user2ts_map = dict()
    valid_user2seq_map = dict()
    valid_user2ts_map = dict()
    test_user2seq_map = dict()
    test_user2ts_map = dict()

    for user, seq in user2seq_map.items():
        ts_seq = user2ts_map[user]
        if user in valid_users:
            history_user2seq_map[user] = seq[:-1]
            history_user2ts_map[user] = ts_seq[:-1]
            valid_user2seq_map[user] = seq[-1:]
            valid_user2ts_map[user] = ts_seq[-1:]
        elif user in test_users:
            history_user2seq_map[user] = seq[:-1]
            history_user2ts_map[user] = ts_seq[:-1]
            test_user2seq_map[user] = seq[-1:]
            test_user2ts_map[user] = ts_seq[-1:]
        else:
            history_user2seq_map[user] = seq
            history_user2ts_map[user] = ts_seq

    seq_maps = (history_user2seq_map,
                history_user2ts_map,
                valid_user2seq_map,
                valid_user2ts_map,
                test_user2seq_map,
                test_user2ts_map)

    return num_user, num_item, seq_maps


class TrainDataset(Dataset):
    def __init__(self, num_item,
                 history_user2seq_map,
                 history_user2ts_map,
                 max_seq_len,
                 limit=2,
                 is_bert=True,
                 mask_ratio=0.2,
                 transform_fn=None):
        self.num_item = num_item
        self.is_bert = is_bert
        # masking ratio only used for BERT-based models
        self.mask_ratio = mask_ratio
        self.user2seq_map = {user: seq[-max_seq_len:] for user, seq in history_user2seq_map.items()
                             if len(seq) >= limit}
        self.user2ts_map = {user: history_user2ts_map[user][-max_seq_len:] for user in self.user2seq_map}
        self.idx2user_map = {i: user for i, user in enumerate(self.user2seq_map)}
        self.transform_fn = transform_fn

    def __len__(self):
        return len(self.idx2user_map)

    def __getitem__(self, idx):
        user = self.idx2user_map.get(idx)
        instance = self._prepare_bert(user) if self.is_bert else self._prepare_left_wise(user)

        return instance if self.transform_fn is None else self.transform_fn(instance)

    def _prepare_bert(self, user):
        seq = list(self.user2seq_map.get(user))
        ts = self.user2ts_map.get(user)
        pos = list()
        # negative sampling might not be used
        neg = list()

        for i, item in enumerate(seq):
            rand_ratio = random.random()
            if rand_ratio < self.mask_ratio:
                pos.append(item)
                neg.append(sample_negative(seq, self.num_item))
                # [mask] item index
                seq[i] = self.num_item + 1
            else:
                # padding not for loss calculation
                pos.append(0)
                neg.append(0)

        return user, seq, ts, pos, neg

    def _prepare_left_wise(self, user):
        user_history = self.user2seq_map.get(user)
        seq = user_history[:-1]
        ts = self.user2ts_map.get(user)[1:]
        pos = user_history[1:]
        # allow repeating for sampling in training stage
        neg = [sample_negative(user_history, self.num_item) for _ in range(len(pos))]

        return user, seq, ts, pos, neg


class PredictionDataset(Dataset):
    def __init__(self, num_item,
                 history_user2seq_map,
                 history_user2ts_map,
                 pred_user2seq_map,
                 pred_user2ts_map,
                 max_seq_len,
                 is_bert=True,
                 is_sampling=False,
                 neg_num=100,
                 transform_fn=None):
        self.num_item = num_item
        self.is_bert = is_bert
        # 100 negative sampling for evaluation when is_sampling=True
        self.is_sampling = is_sampling
        self.neg_num = neg_num
        self.history_user2seq_map = {user: seq[-max_seq_len:] for user, seq in history_user2seq_map.items()}
        self.history_user2ts_map = {user: ts[-max_seq_len:] for user, ts in history_user2ts_map.items()}
        self.pred_user2seq_map = pred_user2seq_map
        self.pred_user2ts_map = pred_user2ts_map
        self.idx2user_map = {i: user for i, user in enumerate(pred_user2seq_map)}
        self.transform_fn = transform_fn

    def __len__(self):
        return len(self.idx2user_map)

    def __getitem__(self, idx):
        user = self.idx2user_map.get(idx)
        instance = self._prepare_bert(user) if self.is_bert else self._prepare_left_wise(user)

        return instance if self.transform_fn is None else self.transform_fn(instance)

    def _prepare_bert(self, user):
        seq = self.history_user2seq_map.get(user) + [self.num_item + 1]
        ts = self.history_user2ts_map.get(user) + self.pred_user2ts_map.get(user)

        indices = self._prepare_indices(user)

        return user, seq, ts, indices

    def _prepare_left_wise(self, user):
        seq = self.history_user2seq_map.get(user)
        ts = self.history_user2ts_map.get(user)[1:] + self.pred_user2ts_map.get(user)

        indices = self._prepare_indices(user)

        return user, seq, ts, indices

    def _prepare_indices(self, user):
        if self.is_sampling:
            pos = self.pred_user2seq_map.get(user)
            neg = batch_sample_negative(pos, self.num_item, self.neg_num)
            indices = pos + neg
        else:
            pos_label = self.pred_user2seq_map.get(user)[0]
            indices = [i for i in range(1, self.num_item + 1)]
            indices[0] = pos_label
            indices[pos_label - 1] = 1

        return indices
