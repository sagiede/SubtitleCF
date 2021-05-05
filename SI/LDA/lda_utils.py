import os
import string
from heapq import nlargest
from operator import itemgetter

import gensim
import nltk
import numpy as np
import regex as re
import sklearn
from gensim.corpora import Dictionary
from tqdm import tqdm

import Consts.consts as consts
from DB.DB_handler import load_movielens_imdb_index_table
from DB.db_consts import models_dir_path, side_info_vectors_dir
from DB.subs_utils import read_all_subs_text
from SI.LDA.stopwords_list import stopwords

# n_tfidf_words = 10000
n_tfidf_words = 12000


def create_lda_movies_vectors(all_subs_text, filename):
    print('Create LDA Vectors')
    movies_vectors = create_lda_topic_vectors([subs[1] for subs in all_subs_text])

    all_movies_vectors = []
    for i in range(len(all_subs_text)):
        movielens_id = int(all_subs_text[i][0])
        movie_vec = np.insert(movies_vectors[i], 0, movielens_id, axis=0)
        all_movies_vectors.append(movie_vec)

    np.savetxt(os.path.join(side_info_vectors_dir, filename), np.asarray(all_movies_vectors))


def preprocess(movies_subs):
    movies_subs = [s.lower() for s in movies_subs]
    table = str.maketrans('', '', string.punctuation)
    movies_subs = [s.translate(table) for s in movies_subs]
    movies_subs = [re.sub(r'\d+', 'num', s) for s in movies_subs]
    movies_subs = [[word for word in s.split() if word not in stopwords] for s in movies_subs]

    stemmer = nltk.stem.PorterStemmer()
    movies_subs = [" ".join([stemmer.stem(word) for word in s]) for s in movies_subs]
    movies_subs = filter_words_by_tfidf(movies_subs)
    return movies_subs


def filter_words_by_tfidf(movies_subs):
    print('Vectorizing')
    tfidf_vec = sklearn.feature_extraction.text.TfidfVectorizer()
    movies_vectors = tfidf_vec.fit_transform(movies_subs)
    print('Finished Vectorizing')

    words_score = []
    for i, word in tqdm(zip(range(movies_vectors.shape[1]), tfidf_vec.get_feature_names())):
        word_tfidf_scores = movies_vectors[:, i].toarray()
        wq9995 = np.quantile(word_tfidf_scores, 0.9995)  # leave only words with higher then 10 scores.
        words_score.append([word, wq9995])
    words_max_tfidf_scores, n_words = words_score, n_tfidf_words

    print('filter_words_by_tfidf')
    top_words_tfidf_scores = nlargest(n_words, words_max_tfidf_scores, key=itemgetter(1))
    filtered_words = [ws[0] for ws in top_words_tfidf_scores]

    with open('SI/namesList.txt') as f:
        names = f.readlines()
    names = [x.strip().lower() for x in names]

    movies_subs_filtered = [[w for w in movie_sub.split(' ') if w in filtered_words and w not in names] for movie_sub in
                            movies_subs]
    return movies_subs_filtered


def create_lda_topic_vectors(movies_subs):
    n_topics = 64
    print('Pre-processing subs')
    movies_subs = preprocess(movies_subs)

    print('Starting doc2Bow')
    movies_dict = Dictionary(movies_subs)
    movies_corpus = [movies_dict.doc2bow(movie_subs) for movie_subs in movies_subs]

    print('Training LDA MODEL')
    lda = gensim.models.LdaModel(movies_corpus, id2word=movies_dict, alpha='auto', num_topics=n_topics, passes=40,
                                 random_state=42)

    print('Save model to disk')
    print('Save model to disk')

    lda.save(os.path.join(models_dir_path, 'LDA_gensim_model', consts.lda_subs_model_file))

    print('Finished Training LDA MODEL')
    movies_vectors = []
    for movie_subs in movies_subs:
        topics = lda[movies_dict.doc2bow(movie_subs)]
        topics_vec = np.zeros(n_topics)
        for topic_idx, topic_val in topics:
            topics_vec[topic_idx] = topic_val
        movies_vectors.append(topics_vec)

    return movies_vectors


def create_lda_embedding_vectors():
    print('Starting')
    index_df = load_movielens_imdb_index_table(subs_filtered=True)
    all_subs_text = read_all_subs_text(index_df)
    create_lda_movies_vectors(all_subs_text, consts.lda_vecs_subs_file)
