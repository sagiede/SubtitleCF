import json
import os

import numpy as np

import SI.CNN.cnn_config as conf
from DB.DB_handler import load_movielens_imdb_index_table
from DB.subs_utils import read_all_subs_text
from general_utils import log_f

from sentence_transformers import SentenceTransformer


def load_movie_bert_embeddings(movielens_id):
    filename = f'movieid-{movielens_id}.npy'
    emb_dir = conf.movies_bert_sentences_split_dir
    embeddings_path = os.path.join(emb_dir, filename)
    with open(embeddings_path, 'rb') as f:
        embeddings = np.load(f)
    return embeddings


def combine_sentences(sentences):
    window_size = 5
    overlap = 1
    combined_sents = []
    for i in range(0, len(sentences) - window_size, overlap):
        curr_sents = sentences[i: i + window_size]
        combined_sents.append('\n'.join(curr_sents))
    return combined_sents


def extract_bert_sentences():
    bert_model = SentenceTransformer('bert-base-nli-max-tokens')
    index_df = load_movielens_imdb_index_table(subs_filtered=True)
    all_subs_text = read_all_subs_text(index_df, lda_extraction=False)
    raw_sentences = []
    bert_sents_dir = conf.movies_bert_sentences_split_dir

    for movie_idx, (movielens_id, subs_text) in enumerate(all_subs_text):
        sentences = subs_text
        filename = f'movieid-{movielens_id}.npy'

        sentences = combine_sentences(sentences)

        sentences_embs = bert_model.encode(sentences)
        sentences_embs = np.asarray(sentences_embs)
        embeddings_path = os.path.join(bert_sents_dir, filename)
        with open(embeddings_path, 'wb') as f:
            np.save(f, sentences_embs)

        raw_sentences.append([movielens_id, sentences])

        print(f'Saved movielens_id {movielens_id} \t movie {movie_idx} out of {len(all_subs_text)}')
        log_f(f'Saved movielens_id {movielens_id} \t movie {movie_idx} out of {len(all_subs_text)}')

    with open(os.path.join(bert_sents_dir, 'movies_sentences.json'), 'w') as f:
        json.dump({'raw_sentences': raw_sentences}, f)

    print('save sentences to file')
    print('All Bert Sentences Embeddings Saved')
