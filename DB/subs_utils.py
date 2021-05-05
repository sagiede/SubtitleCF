import os

import pandas as pd
import pysrt

from DB.db_consts import subtitles_dir_path, subtitles_csv_titles
from general_utils import log_f


def read_subtitles(movie_id):
    movie_id_str = str(movie_id)
    while len(movie_id_str) < 7:
        movie_id_str = '0' + movie_id_str
    movie_subtitles_folder_path = os.path.join(subtitles_dir_path, movie_id_str)
    subtitles_file = None
    for r, d, f in os.walk(movie_subtitles_folder_path):
        for file in f:
            if '.srt' in file:
                subtitles_file = file
    if subtitles_file:
        subs = pysrt.open(os.path.join(movie_subtitles_folder_path, subtitles_file), encoding='iso-8859-1')
        rows = [
            [row.start.ordinal, row.end.ordinal, row.duration.ordinal, row.characters_per_second, row.text_without_tags]
            for row in subs.data]
        subs_df = pd.DataFrame(rows, columns=subtitles_csv_titles)
        return subs_df
    return None


def read_all_subs_text(df_joined, lda_extraction=True):
    n_rows = len(df_joined)
    all_subs_text = []
    for row_idx, row in df_joined.iterrows():
        # if row_idx > 10:
        #     break
        log_f(f'Reading Movie {row_idx}')
        print('Reading movie {} out of {}'.format(row_idx + 1, n_rows))
        imdb_id = int(row['imdb_id'])
        movielens_id = str(int(row['movielens_id']))
        subs_df = read_subtitles(imdb_id)
        subs_text = list(subs_df['text'])

        if lda_extraction:
            subs_text = ' \n '.join(subs_text)
        all_subs_text.append([movielens_id, subs_text])
    return all_subs_text
