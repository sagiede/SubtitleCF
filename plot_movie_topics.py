import os

import gensim
import numpy as np
from gensim.test.utils import datapath
from matplotlib import pyplot as plt
from wordcloud import WordCloud

from Consts.name_consts import lda_subs_model_file, lda_vecs_subs_file
from DB.DB_handler import load_movielens_imdb_index_table, load_movielens_20m_movies
from DB.db_consts import data_path
from FE.stopwords_list import stopwords


def get_axis_sizes_by_n_topics(n_topics_to_print):
    if n_topics_to_print <= 4:
        axis_x = 2
        axis_y = 2
    elif n_topics_to_print <= 6:
        axis_x = 2
        axis_y = 3
    elif n_topics_to_print <= 8:
        axis_x = 2
        axis_y = 4
    elif n_topics_to_print <= 9:
        axis_x = 3
        axis_y = 3
    elif n_topics_to_print <= 12:
        axis_x = 3
        axis_y = 4
    elif n_topics_to_print <= 16:
        axis_x = 4
        axis_y = 4
    elif n_topics_to_print <= 20:
        axis_x = 4
        axis_y = 5
    elif n_topics_to_print <= 25:
        axis_x = 5
        axis_y = 6
    elif n_topics_to_print <= 36:
        axis_x = 6
        axis_y = 6
    elif n_topics_to_print <= 42:
        axis_x = 6
        axis_y = 7
    else:
        axis_x = 8
        axis_y = 8

    return axis_x, axis_y


def censore_words(all_topics):
    words_cens_dict = {'fuck': 'f**k',
                   'shit': 'sh*t',
                   'bitch': 'b**ch',
                   'asshol': 'a***ole',
                   'ass': 'a**',
                   'motherfuck': 'mothe***ck',
                   'bullshit': 'bul***it',
                   'fuckin': 'f***ing',
                   'fucker': 'f***er',
                   'pussi': 'p****',
                   'cunt': 'c***',
                   'suck': 's**k'}
    for t in all_topics:
        for w_idx, w in enumerate(t[1]):
            if w[0] in words_cens_dict:
                t[1][w_idx] = (words_cens_dict[w[0]], w[1])


def plot_lda_topics_wordclouds(lda_model, topics_to_print_ids=None, num_words=20, title_name='lda_d64'):
    # stop_words = set(nltk.corpus.stopwords.words('english') + ['reuter', '\x03'])
    i = 0
    # cols = [color for name, color in mcolors.XKCD_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'
    cols = ['#952e8f', '#0a5f38', '#2242c7', '#b2996e', '#ff0789', '#894585', '#533cc6', '#017374'] * 10

    cloud = WordCloud(stopwords=stopwords,
                      background_color='white',
                      width=2500,
                      height=1800,
                      max_words=num_words,
                      colormap='tab10',
                      color_func=lambda *args, **kwargs: cols[i],
                      prefer_horizontal=1.0)

    # topics = lda_model.show_topics(formatted=False)
    # topics = lda_model.show_topics(num_topics=25, num_words=15, formatted=False)

    all_topics = lda_model.show_topics(num_topics=64, num_words=num_words, formatted=False)
    if not topics_to_print_ids:
        topics_to_print_ids = [0, 1, 2, 3, 5, 6, 7, 8, 11, 12, 13, 15, 17, 18, 19, 20, 21, 22, 25, 26, 28, 32, 33, 34,
                               35, 36, 39, 42, 45, 46, 47, 48, 50, 51, 53, 55, 57, 59, 60, 63]
    topics = [all_topics[t_id] for t_id in topics_to_print_ids]

    n_topics_to_print = len(topics_to_print_ids)

    axis_x_size, axis_y_size = get_axis_sizes_by_n_topics(n_topics_to_print)
    fig, axes = plt.subplots(axis_x_size, axis_y_size, figsize=(65, 55), sharex=True, sharey=True)

    for i, ax in enumerate(axes.flatten()):
        if i >= len(topics):
            break
        fig.add_subplot(ax)
        topic_words = dict(topics[i][1])
        cloud.generate_from_frequencies(topic_words, max_font_size=300)
        plt.gca().imshow(cloud)
        plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
        plt.gca().axis('off')

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.axis('off')
    plt.margins(x=0, y=0)
    plt.tight_layout()

    plt.savefig(f'{title_name}_topic_words.png')
    plt.show()


def plot_movie_topics_by_movielense_id(imdb_id, n_topics, n_words_per_topic):
    imdb_id = int(imdb_id)
    index_df = load_movielens_imdb_index_table(subs_filtered=True)
    movies_df = load_movielens_20m_movies(subs_filtered=True)
    if imdb_id in index_df.imdb_id.tolist():
        movielense_id = int(index_df[index_df.imdb_id == imdb_id]['movielens_id'])
        movie_name = movies_df[movies_df.movielens_id == movielense_id]['movie_name'].values[0]
        si_vecs_filepath = os.path.join(data_path, 'models', 'FE', lda_vecs_subs_file)
        all_movies_vectors = np.loadtxt(si_vecs_filepath)
        movie_lda_vector = all_movies_vectors[all_movies_vectors[:, 0] == movielense_id][:, 1:].tolist()[0]
        movie_topic_scores = [[topic_id, topic_score] for topic_id, topic_score in enumerate(movie_lda_vector)]
        movie_topic_scores = sorted(movie_topic_scores, key=lambda tup: tup[1], reverse=True)
        movie_topic_ids = [topic[0] for topic in movie_topic_scores if topic[1] > 0][:n_topics]
        lda_model = gensim.models.LdaModel.load(datapath(lda_subs_model_file))
        print(f'Plotting Word-cloud for movie: {movie_name}')
        plot_lda_topics_wordclouds(lda_model, topics_to_print_ids=movie_topic_ids, num_words=n_words_per_topic, title_name=movie_name)
    else:
        print('Movie not exist in databse')

