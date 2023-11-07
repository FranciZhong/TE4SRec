import argparse
import ast
import os

from model.TE4SRec import TE4SRec
from model.temporal_encoding import TemporalEncodingParser
from trainer import Trainer
from utils import get_available_device

parser = argparse.ArgumentParser()

# global options
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--output_path', default='./output/results', type=str)

# data params
parser.add_argument('--data_dir', default='./data', type=str)
parser.add_argument('--dataset_name', default='steam', type=str)
parser.add_argument('--pred_threshold', default=3, type=int)

# load model
parser.add_argument('--is_bert', default=False, type=bool)
parser.add_argument('--model_name', default='TE4SRec', type=str)
parser.add_argument('--load_model', default=False, type=bool)

# model params
parser.add_argument('--max_seq_len', default=50, type=int)
parser.add_argument('--alpha', default=0.5, type=float)
parser.add_argument('--d_temporal', default=50, type=int)
parser.add_argument('--d_model', default=50, type=int)
parser.add_argument('--num_block', default=2, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--is_gelu', default=True, type=bool)
parser.add_argument('--dropout_rate', default=0.2, type=float)

# training bert
parser.add_argument('--mask_ratio', default=0.2, type=float)

# training params
parser.add_argument('--pretrain_epoch', default=0, type=int)
parser.add_argument('--stop_patience', default=10, type=int)
parser.add_argument('--train_batch_size', default=512, type=int)
parser.add_argument('--num_worker', default=1, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--adam_beta1', default=0.9, type=float)
parser.add_argument('--adam_beta2', default=0.98, type=float)

# evaluation params
parser.add_argument('--is_sampling', default=False, type=bool)
parser.add_argument('--neg_num', default=100, type=int)
parser.add_argument('--pred_batch_size', default=128, type=int)

config = parser.parse_args()

if __name__ == '__main__':
    config.model_name += '-b' if config.is_bert else '-l'

    output_path = config.output_path
    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    output_path = output_path + '/' + config.dataset_name
    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    output_path = output_path + '/' + config.model_name
    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    config.output_path = output_path
    config.data_path = config.data_dir + '/' + config.dataset_name
    config.top_k = [1, 5, 10]

    if config.load_model:
        with open(config.output_path + '/' + 'config', mode='r') as rf:
            config = type('', (object,), ast.literal_eval(rf.readline()))
            config.load_model = True
    else:
        with open(config.output_path + '/' + 'config', mode='w') as wf:
            wf.write(str(config.__dict__))

    device = get_available_device()
    config.device = device

    print(config.__dict__)

    trainer = Trainer(config, model=TE4SRec, data_parser=TemporalEncodingParser)
    if not config.load_model:
        trainer.train()

    trainer.evaluate(is_test=False)
    trainer.evaluate(is_test=True)

    trainer.scan_temporal_signals()

    print('------ Finish ------')
