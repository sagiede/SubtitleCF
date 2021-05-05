import os
from datetime import datetime

# File names params
import Consts.consts as consts

data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'DATA')
movielens_20m_dir_path = os.path.join(data_path, 'ml-20m')

subtitles_dir_path = os.path.join(data_path, 'subtitles')

subtitles_csv_titles = ['start', 'end', 'duration', 'cps', 'text']

# File path for model
models_dir_path = os.path.join(data_path, 'models')
side_info_vectors_dir = os.path.join(models_dir_path, 'Side_info_vectors')

user_c_name = 'user'
item_c_name: str = 'item'
rating_c_name = 'rate'

ml_20m_dataset_name = 'ml-20m'
preprocess_test_size = 0.4


def get_preprocessed_data_path():
    preprocessed_data_path = os.path.join(data_path, 'preprocessed', 'recsys_60-20-20-o2stars_ml-20m.pickle')
    return preprocessed_data_path


def get_results_path():
    now = datetime.now()
    time = now.strftime("%m%d_%H-%M-%S")
    results = os.path.join(data_path, 'results',
                           '{}_Results_{}.csv'.format(consts.Experiment_suffix, time))
    return results
