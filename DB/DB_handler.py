import os
import pickle

import numpy as np
import pandas as pd

import Consts.consts as consts
from DB.db_consts import movielens_20m_dir_path, get_preprocessed_data_path, \
    models_dir_path


def load_movielens_20m_ratings(subs_filtered=True):
    ratings_file = 'ratings.csv'
    if subs_filtered:
        ratings_file = 'ratings_subs_filtered.csv'
    rating_df = pd.read_csv(os.path.join(movielens_20m_dir_path, ratings_file),
                            sep=',',
                            names=['user', 'item', 'rate', 'time'],
                            skiprows=1)
    return rating_df


def load_movielens_20m_movies(subs_filtered=True):
    movies_file = 'movies.csv'
    if subs_filtered:
        movies_file = 'movies_subs_filtered.csv'
    movies_df = pd.read_csv(os.path.join(movielens_20m_dir_path, movies_file),
                            sep=',',
                            names=['movielens_id', 'movie_name', 'genre'],
                            skiprows=1)
    return movies_df


def load_movielens_imdb_index_table(subs_filtered=True):
    ratings_file = 'links.csv'
    if subs_filtered:
        ratings_file = 'links_subs_filtered.csv'
    index_df = pd.read_csv(os.path.join(movielens_20m_dir_path, ratings_file),
                           sep=',',
                           names=['movielens_id', 'imdb_id', 'tmb_id'],
                           skiprows=1)
    return index_df


def load_movielens_20m_bert_scenes_vectors():
    bert_scenes_vectors_file_name = 'movies_bert_base_scenes_vectors_20m.npz'
    movies_bert_scenes_vec = np.load(os.path.join(movielens_20m_dir_path, bert_scenes_vectors_file_name))
    return movies_bert_scenes_vec


def read_ratings_df():
    ratings_df = load_movielens_20m_ratings(True)
    return ratings_df


def load_preprocessed_data():
    preprocessed_data_file_path = get_preprocessed_data_path()

    # Load run_preprocess data
    with open(preprocessed_data_file_path, 'rb') as f:
        dataset = pickle.load(f)

    n_users, n_items_in_train = dataset['n_users'], dataset['n_items']
    train_items = dataset['train_items']
    users_id_mapping = dataset['users_id_mapping']
    items_id_mapping = dataset['items_id_mapping']

    user_item_train_pairs = dataset['user_item_train_pairs']
    users_train_set_items = dataset['users_train_set_items']

    if consts.load_validation_data:
        user_item_test_pairs = dataset['user_item_val_pairs']
        users_test_set_items = dataset['users_val_set_items']
    else:
        user_item_test_pairs = dataset['user_item_test_pairs']
        users_test_set_items = dataset['users_test_set_items']

    cold_items_t5 = dataset['cold_items_t5']
    cold_items_t10 = dataset['cold_items_t10']
    cold_items_t20 = dataset['cold_items_t20']

    train_items_percentile = dataset['train_items_percentile']

    return n_users, n_items_in_train, user_item_train_pairs, user_item_test_pairs, \
           users_train_set_items, users_test_set_items, train_items, \
           users_id_mapping, items_id_mapping, cold_items_t5, cold_items_t10, cold_items_t20, train_items_percentile


def get_model_path(model_name):
    model_path = os.path.join(models_dir_path, f'{model_name}.pt')
    return model_path
