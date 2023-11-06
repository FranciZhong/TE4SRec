import ast
import json
import os
import random
from collections import defaultdict
from datetime import datetime

import pandas as pd
from tqdm import tqdm

from utils import global_seed

SAMPLE_ITEM = 'sample_item'
SAMPLE_USER = 'sample_user'


def preprocess_amazon(sp, tp, sample_types=None, sample_ratios=None, ts_filter_ratio=0.0):
    def amazon_adapter(line):
        obj = json.loads(line.strip())
        return obj['reviewerID'], obj['asin'], int(obj['unixReviewTime'])

    preprocess_file(sp, tp, amazon_adapter, sample_types, sample_ratios, ts_filter_ratio)


def preprocess_steam(sp, tp, sample_types=None, sample_ratios=None, ts_filter_ratio=0.0):
    def steam_adapter(line):
        obj = ast.literal_eval(line)
        return obj['username'], obj['product_id'], int(datetime.strptime(obj['date'], '%Y-%m-%d').timestamp())

    preprocess_file(sp, tp, steam_adapter, sample_types, sample_ratios, ts_filter_ratio)


def preprocess_yelp(sp, tp, sample_types=None, sample_ratios=None, ts_filter_ratio=0.0):
    def yelp_adapter(line):
        obj = ast.literal_eval(line)
        return obj['user_id'], obj['business_id'], int(datetime.strptime(obj['date'], '%Y-%m-%d %H:%M:%S').timestamp())

    preprocess_file(sp, tp, yelp_adapter, sample_types, sample_ratios, ts_filter_ratio)


def preprocess_goodreads(sp, tp, sample_types=None, sample_ratios=None, ts_filter_ratio=0.0):
    def goodreads_adapter(line):
        obj = json.loads(line)
        return obj['user_id'], obj['book_id'], int(datetime.strptime(obj['timestamp'], '%Y-%m-%d').timestamp())

    preprocess_file(sp, tp, goodreads_adapter, sample_types, sample_ratios, ts_filter_ratio)


def preprocess_beer(sp, tp, sample_types=None, sample_ratios=None, ts_filter_ratio=0.0):
    def beer_adapter(line):
        obj = ast.literal_eval(line)
        return obj['review/profileName'], obj['beer/beerId'], int(obj['review/time'])

    preprocess_file(sp, tp, beer_adapter, sample_types, sample_ratios, ts_filter_ratio)


def preprocess_ml(sp, tp, sample_types=None, sample_ratios=None, ts_filter_ratio=0.0):
    def ml_adapter(row):
        return int(row.userId), int(row.movieId), int(row.timestamp)

    preprocess_file(sp, tp, ml_adapter, sample_types, sample_ratios, ts_filter_ratio)


def preprocess_file(sp, tp, adapter, sample_types, sample_ratios, ts_filter_ratio):
    # skip if dataset already exists, determined by the directory
    dir_exists = check_dataset_dir(tp)
    if dir_exists:
        print(f'Skip the existing dataset {tp}.\n')
        return

    print(f'Start preprocessing from {sp} to {tp}')
    user2items_map, item2users_map = defaultdict(list), defaultdict(list)
    pairs = list()

    if sp.endswith('.json') or sp.endswith('.jsonl'):
        # read jsonl file with adapter to load (user, item, timestamp)
        with open(sp, encoding='utf-8') as rf:
            for line in tqdm(rf):
                user, item, timestamp = adapter(line)
                user2items_map[user].append(item)
                item2users_map[item].append(user)
                pairs.append((user, item, timestamp))

    elif sp.endswith('.csv'):
        # read csv file with adapter to load (user, item, timestamp)
        df = pd.read_csv(sp)
        for i, row in tqdm(df.iterrows()):
            user, item, timestamp = adapter(row)
            user2items_map[user].append(item)
            item2users_map[item].append(user)
            pairs.append((user, item, timestamp))

    else:
        assert 'Invalid file path'

    # item and user sampling
    if sample_types is not None:
        if len(sample_types) != len(sample_ratios):
            assert 'The length of sample_types does not match sample_ratios.'

        for i, sample_type in enumerate(sample_types):
            sample_ratio = sample_ratios[i]
            if sample_type == SAMPLE_ITEM:
                user2items_map, item2users_map, pairs = sample_by_items(user2items_map, item2users_map, pairs,
                                                                        sample_ratio)
            elif sample_type == SAMPLE_USER:
                user2items_map, item2users_map, pairs = sample_by_users(user2items_map, item2users_map, pairs,
                                                                        sample_ratio)

    # filter item collection requesting enough long time span in history
    if ts_filter_ratio != 0.0:
        user2items_map, item2users_map, pairs = filter_time_span(user2items_map, item2users_map, pairs, ts_filter_ratio)

    # hot encode
    actions = encode_actions(pairs, item2users_map, user2items_map)
    save_sorted_actions(actions, tp)


def sample_by_items(user2items_map, item2users_map, pairs, ratio):
    print('Sampling item from: {} entries within {} items and {} users.'
          .format(len(pairs), len(item2users_map), len(user2items_map)))

    sample_items = set(random.sample([item for item in item2users_map], k=int(len(item2users_map) * ratio)))

    new_item2users_map, new_pairs, new_user2item_map = filter_by_item(sample_items,
                                                                      user2items_map,
                                                                      item2users_map,
                                                                      pairs)

    print('                to: {} entries within {} items and {} users.'
          .format(len(new_pairs), len(new_item2users_map), len(new_user2item_map)))

    return new_user2item_map, new_item2users_map, new_pairs


def sample_by_users(user2items_map, item2users_map, pairs, ratio):
    print('Sampling user from: {} entries within {} items and {} users.'
          .format(len(pairs), len(item2users_map), len(user2items_map)))

    sample_users = set(random.sample([user for user in user2items_map], k=int(len(user2items_map) * ratio)))

    new_item2users_map, new_pairs, new_user2item_map = filter_by_user(sample_users, user2items_map, item2users_map,
                                                                      pairs)

    print('                to: {} entries within {} items and {} users.'
          .format(len(new_pairs), len(new_item2users_map), len(new_user2item_map)))

    return new_user2item_map, new_item2users_map, new_pairs


def filter_time_span(user2items_map, item2users_map, pairs, filter_ratio):
    print('Filter time span from: {} entries within {} items and {} users.'
          .format(len(pairs), len(item2users_map), len(user2items_map)))

    max_ts = max([timestamp for user, item, timestamp in pairs])
    min_ts = min([timestamp for user, item, timestamp in pairs])
    time_span_threshold = (max_ts - min_ts) * filter_ratio

    item2min_ts_map = dict()
    item2max_ts_map = dict()

    for user, item, timestamp in pairs:
        if item not in item2max_ts_map:
            item2max_ts_map[item] = timestamp
            item2min_ts_map[item] = timestamp

        item2max_ts_map[item] = max(item2max_ts_map.get(item), timestamp)
        item2min_ts_map[item] = min(item2min_ts_map.get(item), timestamp)

    filtered_item_set = {item for item in item2min_ts_map
                         if item2max_ts_map[item] - item2min_ts_map[item] > time_span_threshold}

    new_item2users_map, new_pairs, new_user2item_map = filter_by_item(filtered_item_set,
                                                                      user2items_map,
                                                                      item2users_map,
                                                                      pairs)

    print('                   to: {} entries within {} items and {} users.'
          .format(len(new_pairs), len(new_item2users_map), len(new_user2item_map)))

    return new_user2item_map, new_item2users_map, new_pairs


def filter_by_item(filtered_item_set, user2items_map, item2users_map, pairs):
    new_item2users_map = {item: users for item, users in item2users_map.items() if item in filtered_item_set}

    new_user2item_map = dict()
    for user, items in user2items_map.items():
        filtered_items = [item for item in items if item in filtered_item_set]
        if len(filtered_items) > 0:
            new_user2item_map[user] = filtered_items

    new_pairs = [(user, item, timestamp)
                 for user, item, timestamp in pairs if user in new_user2item_map and item in new_item2users_map]

    return new_item2users_map, new_pairs, new_user2item_map


def filter_by_user(filtered_user_set, user2items_map, item2users_map, pairs):
    new_user2item_map = {user: items for user, items in user2items_map.items() if user in filtered_user_set}

    new_item2users_map = dict()
    for item, users in item2users_map.items():
        filtered_users = [user for user in users if user in filtered_user_set]
        if len(filtered_users) > 0:
            new_item2users_map[item] = filtered_users

    new_pairs = [(user, item, timestamp)
                 for user, item, timestamp in pairs if user in new_user2item_map and item in new_item2users_map]

    return new_item2users_map, new_pairs, new_user2item_map


def encode_actions(pairs, item2users_map, user2items_map):
    # users and items need to be filtered before hot encoding
    item_set, user_set = get_basic_encode_set(item2users_map, user2items_map)

    print('Encoding from: {} entries within {} items and {} users.'
          .format(len(pairs), len(item2users_map), len(user2items_map)))

    # hot encoding
    user2id_map = build_vocab(user_set)
    item2id_map = build_vocab(item_set)
    actions = list()
    for user, item, timestamp in pairs:
        if user in user2id_map and item in item2id_map:
            actions.append([user2id_map.get(user), item2id_map.get(item), timestamp])

    print('           to: {} entries within {} items and {} users.'
          .format(len(actions), len(item2id_map), len(user2id_map)))

    return actions


def get_basic_encode_set(item2users_map, user2items_map):
    user_set = set()

    # filter out sequences with a length shorter than 2 and get the final user set
    for user, items in user2items_map.items():
        # sequence request length larger than 1
        if len(set(items)) > 1:
            user_set.add(user)

    # filter the item set again to ensure each item at least existing in one sequence
    item_set = set()
    for item, users in item2users_map.items():
        if any(u in user_set for u in users):
            item_set.add(item)

    return item_set, user_set


def get_filtered_encode_set(item2users_map, user2items_map, filter_item, filter_seq):
    filtered_item2user_map = {item: users for item, users in item2users_map.items() if len(users) > filter_item}

    user_set = set()
    for user, items in user2items_map.items():
        if len({item for item in items if item in filtered_item2user_map}) > filter_seq:
            user_set.add(user)

    item_set = set()
    for item, users in filtered_item2user_map.items():
        if any(u in user_set for u in users):
            item_set.add(item)

    return item_set, user_set


def build_vocab(item_set):
    return {item: i + 1 for i, item in enumerate(item_set)}


def check_dataset_dir(tp):
    if os.path.isdir(tp):
        return True
    else:
        return False


def save_sorted_actions(actions, tp):
    os.mkdir(tp)

    # encodings are not saved
    df = pd.DataFrame(actions, columns=['user', 'item', 'timestamp']).sort_values(by=['user', 'timestamp'])
    suffix = '' if tp.endswith('/') else '/'
    suffix += 'data.csv'
    df.to_csv(tp + suffix, header=False, index=False)

    print(f'Finish dataset {tp}\n')


if __name__ == '__main__':
    global_seed(42)
    start_time = datetime.now()

    if not os.path.isdir('./data'):
        os.mkdir('./data')

    preprocess_steam('./data-raw/steam/steam_new.json',
                     './data/steam',
                     sample_types=[SAMPLE_USER],
                     sample_ratios=[0.25])

    preprocess_yelp('./data-raw/yelp/yelp_academic_dataset_review.json',
                    './data/yelp',
                    sample_types=[SAMPLE_ITEM],
                    sample_ratios=[0.1])

    preprocess_goodreads('./data-raw/goodreads/goodreads_reviews_spoiler.json',
                         './data/goodreads',
                         sample_types=[SAMPLE_ITEM],
                         sample_ratios=[0.5])

    preprocess_beer('./data-raw/beer/ratebeer.json',
                    './data/beer',
                    sample_types=[SAMPLE_ITEM],
                    sample_ratios=[0.5])

    preprocess_amazon('./data-raw/toys/Toys_and_Games.json',
                      './data/toys',
                      sample_types=[SAMPLE_USER, SAMPLE_ITEM],
                      sample_ratios=[0.2, 0.2])

    preprocess_amazon('./data-raw/sports/Sports_and_Outdoors.json',
                      './data/sports',
                      sample_types=[SAMPLE_USER, SAMPLE_ITEM],
                      sample_ratios=[0.2, 0.2])

    # preprocess_amazon('./data-raw/beauty/All_Beauty.json',
    #                   './data/beauty')

    # preprocess_ml('./data-raw/ml-25m/ratings.csv',
    #               './data/ml-25m')

    print('Finish data preprocessing in:', datetime.now() - start_time)
