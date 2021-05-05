import torch
from sklearn.model_selection import train_test_split
from torch import nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

import Consts.consts as consts
import SI.CNN.cnn_config as cnn_conf
import SI.CNN.cnn_config as conf
from DB.DB_handler import load_preprocessed_data, get_model_path
from DB.db_consts import models_dir_path
from SI.CNN.BERT_utils import load_movie_bert_embeddings
from general_utils import init_seeds, get_dev

dev = get_dev()


class CNNDatasetIO(Dataset):

    def __init__(self, movielens_ids, items_id_mapping):
        self.movielens_ids = movielens_ids
        self.items_id_mapping = items_id_mapping
        self.n_items = len(movielens_ids)

    def __getitem__(self, idx):
        movielens_id = self.movielens_ids[idx]
        item_id = self.items_id_mapping[movielens_id]
        movie_embeddings = load_movie_bert_embeddings(movielens_id)
        return item_id, movie_embeddings

    def __len__(self):
        return self.n_items


def calc_nlp_loss(i_emb_factors, i_si_factors):
    fixed_i_factors = i_emb_factors.detach()

    sim_function = nn.CosineSimilarity(eps=1e-6)  # cosine similarity
    nlp_score_by_item_emb = - sim_function(i_si_factors, fixed_i_factors)

    nlp_loss = nlp_score_by_item_emb.mean()
    return nlp_loss, nlp_score_by_item_emb


def get_reliable_items(train_items_percentiles, items_id_mapping):
    items_precentiles = train_items_percentiles
    reliable_items = [int(item) for s in items_precentiles[10 - cnn_conf.reliable_groups_threshold:] for item in s]
    reverse_mapping = dict(zip(items_id_mapping.values(), items_id_mapping.keys()))
    reliable_items = [reverse_mapping[i] for i in reliable_items]

    return reliable_items


def cnn_expr_setup():
    dev = get_dev()
    init_seeds()

    n_users, n_items, user_item_train_pairs, user_item_test_pairs, users_train_set_items, users_test_set_items, \
    train_items, users_id_mapping, items_id_mapping, \
    cold_items_t5, cold_items_t10, cold_items_t20, train_items_percentiles = load_preprocessed_data()

    reliable_items = get_reliable_items(train_items_percentiles, items_id_mapping)

    train_movies_cnn_data = train_items
    train_reliable_movies_cnn_data = reliable_items

    if conf.split_cnn_train_test:
        movies_data_train, movies_data_test = train_test_split(train_reliable_movies_cnn_data, test_size=0.3,
                                                               random_state=3, shuffle=True)
        train_reliable_movies_cnn_data = {'train': movies_data_train, 'test': movies_data_test}

    setup_params = {}
    setup_params[consts.n_items_setup] = n_items
    setup_params[consts.train_items_percentiles_setup] = train_items_percentiles
    setup_params[consts.train_movies_cnn_data_setup] = train_movies_cnn_data
    setup_params[consts.train_reliable_movies_cnn_data_setup] = train_reliable_movies_cnn_data
    setup_params[consts.dev_setup] = dev
    setup_params[consts.items_id_mapping_setup] = items_id_mapping
    setup_params[consts.train_items] = train_items

    return setup_params


def load_base_model_embeddings():
    model_path = get_model_path(conf.base_model_name)
    state = torch.load(model_path, map_location=get_dev())
    i_factors = state['rs_model']['item_factors']
    return i_factors


def get_cnn_dataloader(movies_data, items_id_mapping):
    # BERT Embeddings Dataloader
    collate_fn = cnn_bert_collate_fn
    movielens_ids = movies_data

    cnn_loader = DataLoader(
        CNNDatasetIO(movielens_ids, items_id_mapping),
        batch_size=cnn_conf.cnn_batch_size,
        shuffle=True,
        collate_fn=collate_fn)

    return cnn_loader


def pad_bert_sentences_embeddings(i_sentences_embeddings):
    sentences_embeddings = [torch.as_tensor(movie_sentences_embeddings, dtype=torch.float, device=dev) for
                            movie_sentences_embeddings in i_sentences_embeddings]
    sentences_embeddings = pad_sequence(sentences_embeddings, batch_first=True, padding_value=0)
    return sentences_embeddings


def cnn_bert_collate_fn(batch):
    (i, i_sentences_embeddings) = zip(*batch)
    i_bert_embeddings_pad = pad_bert_sentences_embeddings(i_sentences_embeddings)
    i = torch.as_tensor(i, dtype=torch.long, device=dev)
    return [i, i_bert_embeddings_pad]


def load_cnn_model(epoch, model_name):
    model_folder = f'{models_dir_path}/CNN/{model_name}'
    cnn_model_path = model_folder + f'/{model_name}_ep{epoch}'
    cnn_model = torch.load(cnn_model_path)
    return cnn_model


def save_cnn_model(model_name, movies_vectors_after_fc, model):
    cnn_model = {'movies_vectors_after_fc': movies_vectors_after_fc, 'model': model}
    cnn_model_path = models_dir_path + f'/{model_name}'
    torch.save(cnn_model, cnn_model_path)
