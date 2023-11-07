# TE4SRec

This repository is the public version of the research project about Temporal Encoding for Sequential Recommendations (
TE4Rec).

All the model networks and training pipelines are based on PyTorch framework.

## Quick Start

Start training TE4SRec-b with the following command on Steam dataset.

```
python main.py --dataset_name steam --is_bert True --max_seq_len 50 --num_worker 1 --train_batch_size 512 --pred_batch_size 128
```

## Dataset Collection

- [Steam](https://cseweb.ucsd.edu/~jmcauley/datasets.html)
- [Amazon Toys](https://cseweb.ucsd.edu/~jmcauley/datasets.html)
- [Yelp](https://www.yelp.com/dataset/download)
- [Goodreads](https://mengtingwan.github.io/data/goodreads#overview)

## Project Structure

```
│   .gitignore
│   dataset.py
│   data_view.py
│   LICENSE
│   main.py
│   preprocess.py
│   README.md
│   result_view.py
│   trainer.py
│   utils.py
│
├───data
│   ├───steam
│   │       data.csv
│   └───yelp
│
├───data-raw
│   ├───steam
│   │       steam_new.json
│   └───yelp
│
├───image
│       steam_temporal_signals.png
│
├───model
│   │   attention.py
│   │   feedforward.py
│   │   TE4SRec.py
│   │   temporal_encoding.py
│
├───output
│   ├───data
│   │   ├───steam
│   │   │       basic.txt
│   │   │       item_day_gap.png
│   │   │       item_freq_count.png
│   │   │       month_action.png
│   │   │       seq_day_gap.png
│   │   │       seq_len_count.png
│   │
│   └───results
│       ├───steam
│       │   ├───TE4SRec-b
│       │   │       config
│       │   │       eval_valid_line.png
│       │   │       model_checkpoint.pt
│       │   │       temporal_encoding.png
│       │   │       test_metrics.csv
│       │   │       top_k_tables.csv
│       │   │       valid_metrics.csv
```
