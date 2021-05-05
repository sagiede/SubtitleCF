import os

import numpy as np
import torch
from torch import nn as nn
from torch.utils.data import Dataset

import Consts.consts as consts
import RS.Core.rs_config as conf
from DB.DB_handler import load_preprocessed_data, get_model_path
from DB.db_consts import side_info_vectors_dir
from general_utils import get_dev, init_seeds


class TripletUniformPair(Dataset):
    def __init__(self, n_items, users_set_items, users_items_pairs):
        self.n_items = n_items
        self.users_set_items = users_set_items
        self.users_items_pairs = users_items_pairs

    def __getitem__(self, idx):
        u = self.users_items_pairs[idx][0]
        i = self.users_items_pairs[idx][1]

        j_idx = np.random.randint(len(self.users_items_pairs))
        j = self.users_items_pairs[j_idx][1]
        while j in self.users_set_items[u]:
            j_idx = np.random.randint(len(self.users_items_pairs))
            j = self.users_items_pairs[j_idx][1]

        return u, i, j

    def __len__(self):
        return len(self.users_items_pairs)


def bpr_collate_fn(batch):
    (u, i, j) = zip(*batch)
    u = torch.LongTensor(u)
    i = torch.LongTensor(i)
    j = torch.LongTensor(j)
    return [u, i, j]


def rs_expr_setup():
    dev = get_dev()
    init_seeds()

    n_users, n_items, user_item_train_pairs, user_item_test_pairs, users_train_set_items, users_test_set_items, \
    train_items, users_id_mapping, items_id_mapping, \
    cold_items_t5, cold_items_t10, cold_items_t20, _ = load_preprocessed_data()

    setup_params = {}
    setup_params[consts.n_users_setup] = n_users
    setup_params[consts.n_items_setup] = n_items
    setup_params[consts.user_item_train_pairs_setup] = user_item_train_pairs
    setup_params[consts.user_item_test_pairs_setup] = user_item_test_pairs
    setup_params[consts.users_train_set_items_setup] = users_train_set_items
    setup_params[consts.users_test_set_items_setup] = users_test_set_items
    setup_params[consts.dev_setup] = dev
    setup_params[consts.items_id_mapping_setup] = items_id_mapping
    setup_params[consts.cold_items_t5_setup] = cold_items_t5
    setup_params[consts.cold_items_t10_setup] = cold_items_t10
    setup_params[consts.cold_items_t20_setup] = cold_items_t20
    setup_params['train_items'] = train_items

    return setup_params


def save_models_to_file(rs_model, epoch):
    state = {'epoch': epoch,
             'optimizer': rs_model.optimizer.state_dict(),
             'rs_model': rs_model.state_dict(),
             }
    path = get_model_path(model_name=consts.model_name)
    torch.save(state, path)


def update_model_from_file(model_name, rs_model):
    path = get_model_path(model_name=model_name)
    state = torch.load(path, map_location=get_dev())
    rs_model.user_factors = nn.Parameter(state['rs_model']['user_factors'])
    rs_model.item_factors = nn.Parameter(state['rs_model']['item_factors'])


def load_si_vectors_from_cnn():
    path = get_model_path(model_name=conf.si_vectors_name)
    state = torch.load(path[:-3], map_location=get_dev())
    return state['movies_vectors_after_fc']


def load_si_vectors(train_items):
    if 'CNN' in conf.si_vectors_name:
        si_fvs = load_si_vectors_from_cnn()
    else:
        si_vecs_filepath = os.path.join(side_info_vectors_dir, conf.si_vectors_name)
        all_movies_vectors = np.loadtxt(si_vecs_filepath)
        movielens_ids = [int(idx[0]) for idx in all_movies_vectors[:, :1].tolist()]
        movies_vector = all_movies_vectors[:, 1:]

        train_vectors = []
        for id, emb in zip(movielens_ids, movies_vector):
            if id in train_items:
                train_vectors.append(emb)

        si_fvs = torch.as_tensor(train_vectors, dtype=torch.float, device=get_dev())

    # Norm si_fvs to have norm = 2
    si_fvs = 2 * si_fvs / torch.norm(si_fvs, dim=1).reshape(-1, 1)
    si_fvs[si_fvs != si_fvs] = 0  # Replace nan values with zeroes

    return si_fvs


def init_rs_params(model_name, mode):
    model_train_params = {
        'BPR': {
            'epochs': 85,
            'with_si': False,
            'si_vectors_name': None
        },
        'SubCF_cnn': {
            'epochs': 71,
            'with_si': True,
            'si_vectors_name': consts.cnn_model_name

        },
        'SubCF_longformer': {
            'epochs': 78,
            'with_si': True,
            'si_vectors_name': consts.longformer_reduced_vecs_subs_file

        },
        'SubCF_lda': {
            'epochs': 82,
            'with_si': True,
            'si_vectors_name': consts.lda_vecs_subs_file

        }
    }

    conf.n_epochs = model_train_params[model_name]['epochs']
    conf.WITH_SI = model_train_params[model_name]['with_si']
    conf.si_vectors_name = model_train_params[model_name]['si_vectors_name']
    if mode == 'train':
        consts.load_validation_data = True
    else:
        consts.load_validation_data = False
    consts.model_name = model_name
