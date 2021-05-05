import os

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from transformers import LongformerModel, LongformerTokenizer

import Consts.consts as consts
from DB.DB_handler import load_movielens_imdb_index_table
from DB.db_consts import side_info_vectors_dir
from DB.subs_utils import read_all_subs_text
from general_utils import log_f


def encode_subs(model, tokenizer, all_subs_text):
    list_of_emb = []
    max_tokens = 4096
    for i in tqdm(range(len(all_subs_text))):
        SAMPLE_TEXT = all_subs_text[i]  # long input document
        input_ids = torch.tensor(tokenizer.encode(SAMPLE_TEXT))
        from_idx = input_ids.shape[0] - max_tokens - 1000
        to_idx = -1000
        if from_idx < 0:
            to_idx = to_idx - from_idx
            from_idx = 0
        input_ids = input_ids[from_idx:to_idx]
        input_ids = input_ids.unsqueeze(0)  # batch of size 1
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)
        attention_mask[:, [0, -1]] = 2

        with torch.no_grad():

            outputs = model(input_ids, attention_mask=attention_mask)
            hidden_states = outputs[2]

        token_embeddings = torch.stack(hidden_states, dim=0)
        token_embeddings = torch.squeeze(token_embeddings, dim=1)
        token_embeddings = token_embeddings.permute(1, 0, 2)

        token_vecs_sum = []
        for token in token_embeddings:
            sum_vec = torch.sum(token[-4:], dim=0)
            token_vecs_sum.append(sum_vec)

        h = 0
        for i in range(len(token_vecs_sum)):
            h += token_vecs_sum[i]

        list_of_emb.append(np.array(h))
    return list_of_emb


def create_longformer_feats():
    print('Starting')
    index_df = load_movielens_imdb_index_table(subs_filtered=True)
    model = LongformerModel.from_pretrained('allenai/longformer-base-4096', output_hidden_states=True)
    tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
    model.eval()

    all_subs_text = read_all_subs_text(index_df, lda_extraction=False)

    bsize = 128
    filepath = os.path.join(side_info_vectors_dir, consts.longformer_vecs_subs_file)

    for i in range(0, len(all_subs_text), bsize):
        log_f(f'appending Chunks from Movie {i} / {len(all_subs_text)}')
        batch_subs = all_subs_text[i:i + bsize]
        subs_encoded = encode_subs(model, tokenizer, [subs[1] for subs in batch_subs])
        movielens_ids = np.array([int(s[0]) for s in batch_subs])

        movie_vec = np.insert(np.array(subs_encoded), 0, movielens_ids, axis=1)
        with open(filepath, 'a') as f:
            np.savetxt(f, movie_vec)
    print('Finished')


def reduce_longformer_feats():
    filepath = os.path.join(side_info_vectors_dir, consts.longformer_vecs_subs_file)
    with open(filepath, 'r') as f:
        movie_vec = np.loadtxt(f)

    movie_vec_ids = movie_vec[:, 0]
    movie_vec_feats = movie_vec[:, 1:]

    print('Scaling longformer vectors')
    scaler = StandardScaler()
    movie_vec_feats_normed = scaler.fit_transform(movie_vec_feats)

    print('PCA FIT Transform')
    pca = PCA(n_components=64)
    movie_vec_reduced = pca.fit_transform(movie_vec_feats_normed)

    print('Scaling longformer output vectors')
    scaler = StandardScaler()
    movie_vec_reduced_normed = scaler.fit_transform(movie_vec_reduced)

    print('Finished scaling - Writing to file')
    movie_vec = np.insert(movie_vec_reduced_normed, 0, movie_vec_ids, axis=1)

    filepath = os.path.join(side_info_vectors_dir, consts.longformer_reduced_vecs_subs_file)
    with open(filepath, 'a') as f:
        np.savetxt(f, movie_vec)
    print('Finished Reduce Longformer feats')
