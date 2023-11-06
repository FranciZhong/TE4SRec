import copy
import math
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader

import utils
from dataset import *
from model.temporal_encoding import temporal_encoding
from result_view import *


class EarlyStopping:
    def __init__(self, delta=0, patience=10, checkpoint_path=None):
        self.delta = delta
        self.patience = patience
        self.checkpoint_path = checkpoint_path
        self.counter = 0
        self.best_model = None
        self.best_scores = None
        self.is_stopped = False

    def __call__(self, model, scores):
        if self.best_scores is None:
            self.best_scores = scores
            self.save_model(model)
            return self.is_stopped

        # update best scores only when all scores are better than the previous
        is_better_score = not any([score <= self.best_scores[i] + self.delta for i, score in enumerate(scores)])
        if is_better_score:
            self.best_scores = scores
            self.save_model(model)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter == self.patience:
                self.load_model(model)
                self.is_stopped = True

        return self.is_stopped

    def save_model(self, model):
        if self.checkpoint_path is None:
            self.best_model = copy.deepcopy(model)
        else:
            torch.save(model.state_dict(), self.checkpoint_path)

    def load_model(self, model):
        if self.checkpoint_path is None:
            model.load_state_dict(self.best_model.state_dict())
        else:
            model.load_state_dict(torch.load(self.checkpoint_path))


class Trainer:
    def __init__(self, config, model, data_parser):
        utils.global_seed(config.seed)

        num_user, num_item, seq_maps = partition_dataset(config.data_path)
        config.num_user = num_user
        config.num_item = num_item

        # params from dataset distribution
        self.config = config
        self._init_config(seq_maps)

        self.device = config.device
        self.checkpoint_path = self.config.output_path + '/model_checkpoint.pt'

        self.model = model(config).to(config.device)
        if config.load_model:
            self.model.load_state_dict(torch.load(self.checkpoint_path))
        else:
            self._init_model()

        # print(self.model)

        self.data_parser = data_parser(config)
        train_dataset, valid_dataset, test_dataset = self._prepare_datasets(seq_maps)
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset

        self.log_init()

    def _init_config(self, seq_maps):
        (history_user2seq_map,
         history_user2ts_map,
         valid_user2seq_map,
         valid_user2ts_map,
         test_user2seq_map,
         test_user2ts_map) = seq_maps
        max_ts = max(max(ts_seq) for ts_seq in history_user2ts_map.values())
        min_ts = min(min(ts_seq) for ts_seq in history_user2ts_map.values())
        angular_frequency = 2 * math.pi * self.config.alpha / (max_ts - min_ts)
        self.config.max_ts = max_ts
        self.config.min_ts = min_ts
        self.config.angular_frequency = angular_frequency

    def _init_model(self):
        for name, param in self.model.named_parameters():
            try:
                torch.nn.init.xavier_normal_(param.data)
            except:
                pass

    def _prepare_datasets(self, seq_maps):
        (history_user2seq_map,
         history_user2ts_map,
         valid_user2seq_map,
         valid_user2ts_map,
         test_user2seq_map,
         test_user2ts_map) = seq_maps

        train_dataset = TrainDataset(self.config.num_item,
                                     history_user2seq_map,
                                     history_user2ts_map,
                                     self.config.max_seq_len,
                                     is_bert=self.config.is_bert,
                                     mask_ratio=self.config.mask_ratio,
                                     transform_fn=self.data_parser.prepare_train_inputs)

        valid_dataset = PredictionDataset(self.config.num_item,
                                          history_user2seq_map,
                                          history_user2ts_map,
                                          valid_user2seq_map,
                                          valid_user2ts_map,
                                          self.config.max_seq_len,
                                          is_bert=self.config.is_bert,
                                          is_sampling=self.config.is_sampling,
                                          neg_num=self.config.neg_num,
                                          transform_fn=self.data_parser.prepare_predict_inputs)

        test_dataset = PredictionDataset(self.config.num_item,
                                         history_user2seq_map,
                                         history_user2ts_map,
                                         test_user2seq_map,
                                         test_user2ts_map,
                                         self.config.max_seq_len,
                                         is_bert=self.config.is_bert,
                                         is_sampling=self.config.is_sampling,
                                         neg_num=self.config.neg_num,
                                         transform_fn=self.data_parser.prepare_predict_inputs)

        return train_dataset, valid_dataset, test_dataset

    def log_init(self):
        print(f'device: {self.config.device}')
        print(f'users: {self.config.num_user}')
        print(f'items: {self.config.num_item}')
        start_date = datetime.fromtimestamp(self.config.min_ts)
        end_date = datetime.fromtimestamp(self.config.max_ts)
        print(f'train-time range from {start_date} to {end_date}')
        print(f'angular frequency: {self.config.angular_frequency}')
        print(f'Train set sequences: {len(self.train_dataset)}')
        print(f'Valid set sequences: {len(self.valid_dataset)}')
        print(f'Test set sequences: {len(self.test_dataset)}')
        print()

    def train(self):
        start_time = datetime.now()
        eval_valid_results = [list() for _ in range(len(self.config.top_k))]
        mrr_results = list()
        es = EarlyStopping(patience=self.config.stop_patience, checkpoint_path=self.checkpoint_path)
        epoch = 0

        # pretraining epochs without early stops
        while epoch < self.config.pretrain_epoch:
            epoch += 1
            if self.config.is_bert:
                loss = self._train_iter_bert()
            else:
                loss = self._train_iter_left_wise()

            print(f'Epoch: {epoch}, time: {datetime.now() - start_time}, loss: {loss}')

        print('------ Start early stopping ------')

        while not es.is_stopped:
            epoch += 1
            if self.config.is_bert:
                loss = self._train_iter_bert()
            else:
                loss = self._train_iter_left_wise()

            eval_top_k_result, mrr = self._evaluate_top_k(is_test=False)

            # scores for early stop mechanism
            scores = [mrr]
            # scores = list(eval_top_k_result[-1])

            mrr_results.append(mrr)
            for i, (ndcg, hit_rate) in enumerate(eval_top_k_result):
                eval_valid_results[i].append((ndcg, hit_rate))

            es(self.model, scores)
            print(f'Epoch: {epoch}, time: {datetime.now() - start_time}, loss: {loss}')
            print(f'Patience: [{es.counter}/{es.patience}], scores: {scores}')

        print('------ Finish training ------')

        final_epoch = epoch - self.config.stop_patience - self.config.pretrain_epoch
        log_eval_history(self.config.top_k, eval_valid_results, mrr_results, final_epoch, self.config.output_path)

    def evaluate(self, is_test=True):
        eval_top_k_results, mrr = self._evaluate_top_k(is_test=is_test)

        print(('Test' if is_test else 'Validation') + ' results:')
        for i, (ndcg, hit_rate) in enumerate(eval_top_k_results):
            print(f'NDCG@{self.config.top_k[i]}: {ndcg}, HR@{self.config.top_k[i]}: {hit_rate}')

        print(f'MRR: {mrr}')

        output_file = self.config.output_path + ('/test_metrics.csv' if is_test else '/valid_metrics.csv')
        log_eval_top_k(self.config.top_k, eval_top_k_results, mrr, output_file)

    def _train_iter_bert(self):
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
        betas = (self.config.adam_beta1, self.config.adam_beta2)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr, betas=betas)

        self.model.train()
        loss_count = 0
        loss_sum = 0

        train_dataloader = DataLoader(self.train_dataset,
                                      batch_size=self.config.train_batch_size,
                                      shuffle=True,
                                      num_workers=self.config.num_worker)

        for batch_data in train_dataloader:
            user, seq, ts, labels = batch_data
            batch_data = [x.to(self.device) for x in [user, seq, ts]]

            indices = np.where(labels > 0)

            outputs = self.model.forward(batch_data)

            optimizer.zero_grad()

            outputs = outputs[indices]
            labels = labels[indices].to(self.device)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            count = len(indices[0])
            loss_count += count
            loss_sum += loss.item() * count

        return loss_sum / loss_count

    def _train_iter_left_wise(self):
        criterion = torch.nn.BCEWithLogitsLoss()
        betas = (self.config.adam_beta1, self.config.adam_beta2)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr, betas=betas)

        self.model.train()
        loss_count = 0
        loss_sum = 0

        train_dataloader = DataLoader(self.train_dataset,
                                      batch_size=self.config.train_batch_size,
                                      shuffle=True,
                                      num_workers=self.config.num_worker)

        for batch_data in train_dataloader:
            # batch data request format like (seq, ..., pos, neg)
            user, seq, ts, pos, neg = batch_data
            batch_data = [x.to(self.device) for x in [user, seq, ts]]

            # Only collect the loss from non-padding positions
            indices = np.where(pos > 0)
            outputs = self.model.forward(batch_data)
            outputs = outputs[indices]

            # pos/neg labels
            pos_labels = torch.ones(outputs.shape[0], device=self.device)
            neg_labels = torch.zeros(outputs.shape[0], device=self.device)

            # pos/neg scores
            pos_indices = pos[indices].numpy()
            pos_scores = outputs[(np.arange(0, pos_indices.shape[0]), pos_indices)]

            neg_indices = neg[indices].numpy()
            neg_scores = outputs[(np.arange(0, neg_indices.shape[0]), neg_indices)]

            optimizer.zero_grad()

            loss = criterion(pos_scores, pos_labels)
            loss += criterion(neg_scores, neg_labels)

            loss.backward()
            optimizer.step()

            count = len(indices[0])
            loss_count += count
            loss_sum += loss.item() * count

        return loss_sum / loss_count

    @torch.no_grad()
    def _evaluate_top_k(self, is_test=True):
        self.model.eval()

        # NDCG@K & HitRate@K
        eval_metrics = [(0.0, 0.0) for _ in self.config.top_k]
        mrr = 0.0

        dataset = self.test_dataset if is_test else self.valid_dataset
        num_user = len(dataset)
        dataloader = DataLoader(dataset,
                                batch_size=self.config.pred_batch_size,
                                shuffle=True,
                                num_workers=self.config.num_worker)

        for batch_data in dataloader:
            user, seq, ts, indices = batch_data
            batch_data = [x.to(self.device) for x in [user, seq, ts]]

            # negative operation for ranking
            predictions = -self.model.forward(batch_data)
            predictions = predictions[:, -1, :].gather(1, indices.to(self.device))

            # mps doesn't support arg sort
            if predictions.device.type == 'mps':
                predictions = predictions.to('cpu')

            # get first-item rank
            ranks = predictions.argsort().argsort()[:, 0]
            mrr += sum([1 / (rank.item() + 1) for rank in ranks])

            for i, k in enumerate(self.config.top_k):
                ndcg, hit_rate = eval_metrics[i]
                ndcg_scores = [1 / np.log2(rank.item() + 2) for rank in ranks if rank < k]
                ndcg += sum(ndcg_scores)
                hit_rate += len(ndcg_scores)
                eval_metrics[i] = (ndcg, hit_rate)

        mrr /= num_user

        for i in range(len(self.config.top_k)):
            ndcg, hit_rate = eval_metrics[i]
            ndcg /= num_user
            hit_rate /= num_user
            eval_metrics[i] = (ndcg, hit_rate)

        return eval_metrics, mrr

    def scan_temporal_signals(self, num_point=1000):
        time_interval = (self.config.max_ts - self.config.min_ts) / num_point
        time_points = [self.config.min_ts + int(i * time_interval) for i in range(num_point + 1)]

        ts_encoding = temporal_encoding(time_points,
                                        self.config.d_temporal,
                                        self.config.min_ts,
                                        self.config.angular_frequency)

        ts_tensor = torch.Tensor(ts_encoding).to(self.config.device)
        with torch.no_grad():
            signal_tensor = self.model.temporal_ffn.forward(ts_tensor)

        signal_arr = np.transpose(signal_tensor.to('cpu').numpy())

        print_signals(signal_arr, self.config.output_path)
