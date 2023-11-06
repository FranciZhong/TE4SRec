import math
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme()


def print_data_dist(sd, td):
    if os.path.isdir(td):
        return

    print(f'Start analyse dateset from {sd}')
    start = datetime.now()
    df = load_df(sd)

    print_basic_info(df, td)
    get_seq_len_dist(df, td)
    get_item_freq(df, td)
    get_date_distribution(df, td)
    get_user_day_gap_dist(df, td)
    get_item_day_gap_dist(df, td)

    print(f'Output results in {td} within {datetime.now() - start}')


def load_df(sd):
    files = os.listdir(sd)
    files = [sd + file if sd.endswith('/') else sd + '/' + file for file in files]
    return pd.concat([pd.read_csv(file,
                                  names=['user', 'item', 'datetime'],
                                  dtype={
                                      'user': np.int32,
                                      'item': np.int32,
                                  },
                                  converters={
                                      'datetime': lambda ts: datetime.fromtimestamp(int(ts))
                                  },
                                  parse_dates=['datetime']) for file in files])


def print_basic_info(df, td):
    num_user = len(np.unique(df['user']))
    num_item = len(np.unique(df['item']))
    num_action = len(df)
    sparsity = 1 - num_action / (num_user * num_item)

    # average length of user sequences
    user_seq_lens = df.groupby(by='user')['item'].count()
    avg_seq_len = sum(user_seq_lens.values) / len(user_seq_lens.values)

    # average frequency of items
    item_counts = df.groupby(by='item')['user'].count()
    avg_item_frequency = sum(item_counts) / len(item_counts)

    start_date = min(df['datetime'])
    end_date = max(df['datetime'])
    time_gap = end_date - start_date

    user_day_gaps = df.groupby(by='user')['datetime'].apply(lambda x: (max(x) - min(x)).days)
    avg_user_day_gap = sum(user_day_gaps.values) / len(user_day_gaps.values)

    item_day_gaps = df.groupby(by='item')['datetime'].apply(lambda x: (max(x) - min(x)).days)
    avg_item_day_gap = sum(item_day_gaps.values) / len(item_day_gaps.values)

    if not os.path.isdir(td):
        os.mkdir(td)

    file = 'basic.txt'
    file = td + file if td.endswith('/') else td + '/' + file

    with open(file, mode='w') as fp:
        fp.write(f'users: {num_user}\n')
        fp.write(f'items: {num_item}\n')
        fp.write(f'actions: {num_action}\n')
        fp.write('sparsity: {:.4f}%\n'.format(sparsity * 100))
        fp.write('average sequence length: {:.4f}\n'.format(avg_seq_len))
        fp.write('average item frequency: {:.4f}\n'.format(avg_item_frequency))
        fp.write(f'from {start_date} to {end_date}\n')
        fp.write(f'days gap: {time_gap.days}\n')
        fp.write('average user day gap: {:.2f}\n'.format(avg_user_day_gap))
        fp.write('average item day gap: {:.2f}\n'.format(avg_item_day_gap))


def get_seq_len_dist(df, td):
    user_seq_lens = df.groupby(by='user')['item'].count()

    # sns.histplot(user_seq_lens, fill=True, log_scale=(max(user_seq_lens) > 50, True))

    bin_width = pow(10, max(0, int(math.log10(max(user_seq_lens))) - 2))
    sns.histplot(user_seq_lens, fill=True, binwidth=bin_width, log_scale=(False, True))

    plt.xlabel('Sequence length')
    plt.ylabel('Count')
    plt.title('Sequence length count')

    file = 'seq_len_count.png'
    file = td + file if td.endswith('/') else td + '/' + file
    plt.savefig(file)
    plt.clf()


def get_item_freq(df, td):
    item_counts = df.groupby(by='item')['user'].count()

    # sns.kdeplot(item_counts, fill=True, log_scale=(True, False))

    bin_width = pow(10, max(0, int(math.log10(max(item_counts))) - 2))
    sns.histplot(item_counts, fill=True, binwidth=bin_width, log_scale=(False, True))

    plt.xlabel('Item frequency')
    plt.ylabel('Count')
    plt.title('Item frequency distribution')

    file = 'item_freq_count.png'
    file = td + file if td.endswith('/') else td + '/' + file
    plt.savefig(file)
    plt.clf()


def get_user_day_gap_dist(df, td):
    user_day_gaps = df.groupby(by='user')['datetime'].apply(lambda x: (max(x) - min(x)).days)

    sns.histplot(user_day_gaps, fill=True, log_scale=(False, True))

    plt.xlabel('Sequence day gap')
    plt.ylabel('Count')
    plt.title('Sequence day-gap distribution')

    file = 'seq_day_gap.png'
    file = td + file if td.endswith('/') else td + '/' + file
    plt.savefig(file)
    plt.clf()


def get_item_day_gap_dist(df, td):
    item_day_gaps = df.groupby(by='item')['datetime'].apply(lambda x: (max(x) - min(x)).days)

    sns.histplot(item_day_gaps, fill=True, log_scale=(False, True))

    plt.xlabel('Item day gap')
    plt.ylabel('Count')
    plt.title('Item day-gap distribution')

    file = 'item_day_gap.png'
    file = td + file if td.endswith('/') else td + '/' + file
    plt.savefig(file)
    plt.clf()


def get_date_distribution(df, td):
    # df.resample('m', on='datetime')['user'].count().plot()

    sns.lineplot(df.resample('m', on='datetime')['user'].count())

    plt.title('Actions per month')
    plt.xlabel('Month of Date')
    plt.ylabel('Number of actions')

    file = 'month_action.png'
    file = td + file if td.endswith('/') else td + '/' + file
    plt.savefig(file)
    plt.clf()


if __name__ == '__main__':
    data_dir = './data/'
    output_dir = './output/'
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    output_dir += 'data/'
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    datasets = [
        'steam',
        'yelp',
        'goodreads',
        'beer',
        'toys',
        'sports',
    ]
    for dataset in datasets:
        print_data_dist(data_dir + dataset, output_dir + dataset)
