import argparse
import os

import gensim
import numpy as np
from matplotlib import pyplot as plt
from wordcloud import WordCloud

import Consts.consts as consts
from DB.DB_handler import load_movielens_imdb_index_table, load_movielens_20m_movies
from DB.db_consts import data_path, models_dir_path
from SI.LDA.stopwords_list import stopwords


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
    elif n_topics_to_print <= 40:
        axis_x = 8
        axis_y = 5
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


def plot_lda_topics_wordclouds(lda_model, chosen_topics_to_print_ids, num_words=20, title_name='lda_d64'):
    i = 0
    cols = ['#952e8f', '#0a5f38', '#2242c7', '#b2996e', '#ff0789', '#894585', '#533cc6', '#017374', '#c0392b'] * 10

    cloud = WordCloud(stopwords=stopwords,
                      background_color='white',
                      width=2500,
                      height=1800,
                      max_words=num_words,
                      colormap='tab10',
                      color_func=lambda *args, **kwargs: cols[i],
                      prefer_horizontal=1.0)

    all_topics = lda_model.show_topics(num_topics=64, num_words=num_words, formatted=False)
    censore_words(all_topics)

    # if not chosen_topics_to_print_ids:
    topics = [all_topics[t_id] for t_id in chosen_topics_to_print_ids]

    n_topics_to_print = len(topics)

    axis_x_size, axis_y_size = get_axis_sizes_by_n_topics(n_topics_to_print)
    fig, axes = plt.subplots(axis_x_size, axis_y_size, figsize=(60, 60), sharex=True, sharey=True)

    for i, ax in enumerate(axes.flatten()):
        if i >= len(topics):
            break
        fig.add_subplot(ax)
        topic_words = dict(topics[i][1])
        cloud.generate_from_frequencies(topic_words, max_font_size=300)
        plt.gca().imshow(cloud)
        plt.gca().set_title('Topic ' + str(i + 1), fontdict=dict(size=30))
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
        si_vecs_filepath = os.path.join(data_path, 'models', 'Side_info_vectors', consts.lda_vecs_subs_file)
        all_movies_vectors = np.loadtxt(si_vecs_filepath)
        movie_lda_vector = all_movies_vectors[all_movies_vectors[:, 0] == movielense_id][:, 1:].tolist()[0]
        movie_topic_scores = [[topic_id, topic_score] for topic_id, topic_score in enumerate(movie_lda_vector)]
        movie_topic_scores = sorted(movie_topic_scores, key=lambda tup: tup[1], reverse=True)
        movie_topic_ids = [topic[0] for topic in movie_topic_scores if topic[1] > 0][:n_topics]
        lda_model = gensim.models.LdaModel.load(os.path.join(models_dir_path, 'LDA_gensim_model',
                                                             consts.lda_subs_model_file))
        print(f'Plotting Word-cloud for movie: {movie_name}')
        plot_lda_topics_wordclouds(lda_model, chosen_topics_to_print_ids=movie_topic_ids,
                                   num_words=n_words_per_topic, title_name=movie_name)
    else:
        print('Movie not exist in databse')


def main():
    parser = argparse.ArgumentParser(description="Running The Code arguments:")
    parser.add_argument("-m", "--movie", help="Batman movie Example: 0468569 or 468569 ", required=True, default="")
    parser.add_argument("-t", "--topics", help="Example: 8 (max of 64) (recommended range -[3,8])", required=True,
                        default="")
    parser.add_argument("-w", "--words", help="Example: 25 recommended range -[25,40]", required=True, default="")

    argument = parser.parse_args()
    err_msg = 'Invalid Input'

    if not (argument.movie and argument.topics and argument.words):
        print(err_msg)
        return

    imdb_id, n_topics, n_words = argument.movie, int(argument.topics), int(argument.words)
    plot_movie_topics_by_movielense_id(imdb_id, n_topics=n_topics, n_words_per_topic=n_words)


if __name__ == '__main__':
    main()
