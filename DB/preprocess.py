import os
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from DB.DB_handler import read_ratings_df
from DB.db_consts import user_c_name, item_c_name, get_preprocessed_data_path, preprocess_test_size
from general_utils import log_f


def convert_df_ids(df, column_name, mapping):
    item_columns = df[column_name].to_list()
    item_columns_converted = [mapping[item] for item in item_columns]
    df[column_name] = item_columns_converted


def filter_test_df(df, items_mapping):
    df = df[df[item_c_name].isin(items_mapping.keys())]
    return df


def filter_low_rate_values(ratings_df):
    ratings_df_o2stars = ratings_df[ratings_df.rate.gt(2)]
    return ratings_df_o2stars


def create_train_test_data(ratings_df):
    ratings_df = filter_low_rate_values(ratings_df)
    ratings_df = ratings_df.sort_values(by=['time'])
    users_df = ratings_df.groupby([user_c_name])

    print('splitting ratings_df to train and test')
    log_f('splitting ratings_df to train and test')
    ratings_train_df = []
    ratings_val_df = []
    ratings_test_df = []
    for idx, user_df in tqdm(users_df):
        if len(user_df) < 5:  # should not enter here
            continue
        if len(user_df) <= 10:
            user_train, user_validation, user_test = user_df[:-2], user_df[-2:-1], user_df[-1:]
        elif len(user_df) <= 12:
            user_train, user_validation, user_test = user_df[:-4], user_df[-4:-2], user_df[-2:]
        else:
            user_train, user_validation_test_df = train_test_split(user_df, test_size=preprocess_test_size,
                                                                   random_state=3, shuffle=False)
            user_validation, user_test = train_test_split(user_validation_test_df, test_size=1 / 2,
                                                          random_state=3, shuffle=False)

        ratings_train_df.append(user_train)
        ratings_val_df.append(user_validation)
        ratings_test_df.append(user_test)

    ratings_train_df = pd.concat(ratings_train_df)
    ratings_val_df = pd.concat(ratings_val_df)
    ratings_test_df = pd.concat(ratings_test_df)

    print('Creating Ids Mapping')
    log_f('Creating Ids Mapping')
    origin_items_ids = [int(id) for id in ratings_train_df[item_c_name].unique()]
    origin_users_ids = [int(id) for id in ratings_train_df[user_c_name].unique()]
    origin_items_ids = sorted(origin_items_ids)
    origin_users_ids = sorted(origin_users_ids)
    items_mapping = {origin_id: idx for idx, origin_id in enumerate(origin_items_ids)}
    users_mapping = {origin_id: idx for idx, origin_id in enumerate(origin_users_ids)}

    print('Converting train_df')
    log_f('Converting train_df')
    convert_df_ids(ratings_train_df, item_c_name, items_mapping)
    convert_df_ids(ratings_train_df, user_c_name, users_mapping)

    print('Filtering test_df')
    log_f('Filtering test_df')
    ratings_val_df = filter_test_df(ratings_val_df, items_mapping)
    ratings_test_df = filter_test_df(ratings_test_df, items_mapping)

    print('Converting test_df')
    log_f('Converting test_df')
    convert_df_ids(ratings_val_df, item_c_name, items_mapping)
    convert_df_ids(ratings_val_df, user_c_name, users_mapping)
    convert_df_ids(ratings_test_df, item_c_name, items_mapping)
    convert_df_ids(ratings_test_df, user_c_name, users_mapping)

    return ratings_train_df, ratings_val_df, ratings_test_df, origin_users_ids, \
           origin_items_ids, users_mapping, items_mapping


def create_tuples(ratings_df, n_users):
    users_set = [set([])] * n_users
    users_tuples = []
    users_df = ratings_df.groupby([user_c_name])
    for idx, user_df in users_df:
        user_l = list(user_df[user_c_name])
        user = user_l[0]
        items_l = list(user_df[item_c_name])

        users_set[user] = set(items_l)
        assert len(items_l) == len(users_set[user])

        users_tuples.extend(list(zip(user_l, items_l)))

    return users_set, users_tuples


def calc_cold_items_list(user_item_train_pairs):
    cold_items_t5 = []
    cold_items_t10 = []
    cold_items_t20 = []

    i_pop_map_train = {}
    for u, i in user_item_train_pairs:
        if i not in i_pop_map_train:
            i_pop_map_train[i] = 0
        i_pop_map_train[i] += 1

    for i, c in i_pop_map_train.items():
        if c <= 5:
            cold_items_t5.append(i)
        if c <= 10:
            cold_items_t10.append(i)
        if c <= 20:
            cold_items_t20.append(i)

    return cold_items_t5, cold_items_t10, cold_items_t20


def create_percentile_dict(items_popularity_map):
    train_items_percentile = np.array_split(
        np.array(sorted(list(items_popularity_map.items()), key=lambda x: x[1]))[:, 0], 10)
    train_items_percentile = [set(p.tolist()) for p in train_items_percentile]
    return train_items_percentile


def create_items_popularity_map(ratings_df):
    items_popularity_map = {}
    items_df = ratings_df.groupby([item_c_name])
    for idx, item_df in items_df:
        item_id = list(item_df[item_c_name])[0]
        n_users = len(item_df)
        items_popularity_map[str(item_id)] = n_users
    return items_popularity_map


def run_preprocess():
    print('Running Pre-process on ml-20m dataset')

    print('Reading ratings_df')
    log_f('Reading ratings_df')
    ratings_df = read_ratings_df()
    ratings_train_df, ratings_val_df, ratings_test_df, origin_users_ids, origin_items_ids, \
    users_mapping, items_mapping = create_train_test_data(ratings_df)

    n_users = len(origin_users_ids)
    n_items = len(origin_items_ids)

    print('Creating Train tuples')
    log_f('Creating Train tuples')
    train_user_set, train_users_tuples = create_tuples(ratings_train_df, n_users)

    print('Creating Validation tuples')
    log_f('Creating Validation tuples')
    val_user_set, val_users_tuples = create_tuples(ratings_val_df, n_users)

    print('Creating Test tuples')
    log_f('Creating Test tuples')
    test_user_set, test_users_tuples = create_tuples(ratings_test_df, n_users)

    print('Calculating cold_items_lists')
    log_f('Calculating cold_items_lists')
    cold_items_t5, cold_items_t10, cold_items_t20 = calc_cold_items_list(train_users_tuples)

    print('Creating items_popularity_map')
    log_f('Creating items_popularity_map')
    items_popularity_map = create_items_popularity_map(ratings_train_df)
    train_items_percentile = create_percentile_dict(items_popularity_map)

    print('Saving preprocessed data to file')
    dataset = {'n_users': n_users, 'n_items': n_items,
               'train_items': origin_items_ids,
               'users_id_mapping': users_mapping,
               'items_id_mapping': items_mapping,

               'user_item_train_pairs': train_users_tuples,
               'user_item_val_pairs': val_users_tuples,
               'user_item_test_pairs': test_users_tuples,

               'users_train_set_items': train_user_set,
               'users_val_set_items': val_user_set,
               'users_test_set_items': test_user_set,

               'cold_items_t5': cold_items_t5,
               'cold_items_t10': cold_items_t10,
               'cold_items_t20': cold_items_t20,
               'train_items_percentile': train_items_percentile,

               }

    output_data = get_preprocessed_data_path()

    dirname = os.path.dirname(os.path.abspath(output_data))
    os.makedirs(dirname, exist_ok=True)
    with open(output_data, 'wb') as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
