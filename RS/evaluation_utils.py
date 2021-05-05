import json
import os

import numpy as np
import pandas as pd
import torch

import Consts.consts as consts
from DB.db_consts import data_path
from RS.Core.rs_config import batch_size
from general_utils import get_dev, log_f


def create_items_set_users(n_items, users_items_pairs):
    items_set_users = []
    for i in range(n_items):
        items_set_users.append([])
    for u, i in users_items_pairs:
        items_set_users[i].append(u)
    return items_set_users


def create_non_observed_mask(train_user_list, user_i, num_of_users, num_of_items, batch):
    # Create already observed mask - for testing only non-observed items

    dev = get_dev()
    mask = torch.ones([min([batch, num_of_users - user_i]), num_of_items], device=dev)

    for j in range(batch):
        if user_i + j >= num_of_users:
            break
        index = torch.as_tensor(list(train_user_list[user_i + j]), dtype=torch.long, device=dev)
        value = torch.tensor(0.0, device=dev)
        mask[j].scatter_(dim=0, index=index, value=value)
    return mask


def calc_batch_users_rank_results(model, mask, user_batch):
    # Calculate prediction value (chose batch rows and multiply user_emb with item_emb_transpose at the same rows)
    users_results = model.calc_user_ranking(user_batch)

    # Make zero for already observed item
    users_results = torch.mul(mask, users_results)
    return users_results


def get_user_batch(num_of_users, user_i, batch):
    if consts.use_cuda:
        user_batch = torch.tensor(list(range(user_i, user_i + min(batch, num_of_users - user_i)))).cuda()
    else:
        user_batch = torch.tensor(list(range(user_i, user_i + min(batch, num_of_users - user_i))))
    return user_batch


def eval_all_topk_preds(model, num_of_users, num_of_items, users_train_set_items, batch=batch_size):
    # Calculate max k value
    max_k = 20

    # Compute all pair of training and test record
    all_topk_preds = None
    users_mpr_results = []

    # for user_i in tqdm(range(0, num_of_users, batch)):
    for user_i in range(0, num_of_users, batch):
        mask = create_non_observed_mask(users_train_set_items, user_i, num_of_users, num_of_items, batch)
        user_batch = get_user_batch(num_of_users, user_i, batch)
        users_results = calc_batch_users_rank_results(model, mask, user_batch)

        _, topk_result = torch.topk(users_results, k=max_k, dim=1)

        all_topk_preds = topk_result if all_topk_preds is None else torch.cat((all_topk_preds, topk_result), dim=0)

    all_topk_preds = all_topk_preds.cpu()
    return all_topk_preds, users_mpr_results


def calc_coverage(recommendations, cold_items, num_of_items):
    if cold_items:
        n_recommended_items = len(set((recommendations.flatten())).intersection(cold_items))
        coverage = n_recommended_items / len(cold_items)

    else:
        n_recommended_items = len(set((recommendations.flatten())))
        coverage = n_recommended_items / num_of_items
    return coverage


def reciprocal_rank(hit_values):
    for i, hit_value in enumerate(hit_values):
        if hit_value:
            return 1 / (i + 1)
    return 0


def eval_measures_on_topk(all_topk_preds, num_of_users, num_of_items, users_test_set_items, cold_items):
    coverage = calc_coverage(all_topk_preds, cold_items, num_of_items)

    # Sort indices and get test_pred_topk

    reciprocal_rank_results, precisions, recalls = [], [], []
    k = 20
    for user_i in range(num_of_users):
        test_items = set(users_test_set_items[user_i])

        if test_items:
            pred = all_topk_preds[user_i].tolist()
            predictions_bool = [item in test_items for item in pred]
            reciprocal_rank_results.append(reciprocal_rank(predictions_bool))

            pred_k = set(all_topk_preds[user_i, :k].tolist())
            val_k = len(test_items & pred_k)
            precisions.append(val_k / k)
            recalls.append(val_k / len(test_items))

    precision = np.around(np.mean(precisions), 4)
    recall = np.around(np.mean(recalls), 4)
    mrr = np.around(np.mean(reciprocal_rank_results), 4)
    coverage = np.around(coverage, 2)

    return coverage, recall, precision, mrr


def filter_test_data_to_group(users_test_set_items, cold_items_t5, cold_items_t10, cold_items_t20):
    read_from_file = True
    if read_from_file:
        items_cold_split_test_data_path = os.path.join(data_path, 'preprocessed', 'items_cold_splitting_test_data_o2stars.json')
        with open(items_cold_split_test_data_path, 'r') as f:
            test_items_dict = json.load(f)
        return test_items_dict
    else:
        users_set_items_t5_cold = []
        users_set_items_t10_cold = []
        users_set_items_t20_cold = []

        print('Create cold items dict')
        log_f('Create cold items dict')

        for u_set_items in users_test_set_items:
            u_set_items_filtered_t5_cold = []
            u_set_items_filtered_t10_cold = []
            u_set_items_filtered_t20_cold = []
            for item in u_set_items:
                if item in cold_items_t5:
                    u_set_items_filtered_t5_cold.append(item)
                if item in cold_items_t10:
                    u_set_items_filtered_t10_cold.append(item)
                if item in cold_items_t20:
                    u_set_items_filtered_t20_cold.append(item)

            users_set_items_t5_cold.append(u_set_items_filtered_t5_cold)
            users_set_items_t10_cold.append(u_set_items_filtered_t10_cold)
            users_set_items_t20_cold.append(u_set_items_filtered_t20_cold)

        print('Finished Create cold items dict')
        log_f('Finished Create cold items dict')

        test_items_dict = {'t5_cold': users_set_items_t5_cold,
                           't10_cold': users_set_items_t10_cold,
                           't20_cold': users_set_items_t20_cold}

        items_cold_split_test_data_path = os.path.join(data_path,
                                                       'preprocessed/BPR/items_cold_splitting_test_data_o2stars.json')
        with open(items_cold_split_test_data_path, 'w') as f:
            json.dump(test_items_dict, f)

        return test_items_dict


def evaluate_ranking_model(rs_model, results_file_path, setup_params):
    rs_model.eval()

    titles = {'Group': [], 'Recall @20': [], 'Precision @20': [], 'MRR @20': [],
              'Catalog Coverage': [], 'Catalog Hits': []}

    users_train_set_items = setup_params[consts.users_train_set_items_setup]
    users_test_set_items = setup_params[consts.users_test_set_items_setup]

    results_df = pd.DataFrame(data=titles)
    results_df.to_csv(results_file_path, index=False)
    cold_items_t5 = setup_params[consts.cold_items_t5_setup]
    cold_items_t10 = setup_params[consts.cold_items_t10_setup]
    cold_items_t20 = setup_params[consts.cold_items_t20_setup]
    test_items_dict = filter_test_data_to_group(users_test_set_items, cold_items_t5, cold_items_t10, cold_items_t20)
    test_items_dict = {**test_items_dict, 'all': users_test_set_items}
    setup_params['test_items_dict'] = test_items_dict
    num_of_users = rs_model.n_users
    num_of_items = rs_model.n_items

    print('Evaluating users all topk recommendations')
    all_topk_preds, _ = eval_all_topk_preds(rs_model, num_of_users, num_of_items, users_train_set_items)
    all_topk_preds = all_topk_preds.numpy()

    def count_catalog_hits(all_topk_preds, test_items):
        items_l = []
        u_set_test_items = []
        for u_preds, u_test_set in zip(all_topk_preds, test_items):
            if u_test_set:
                u_set_test_items.append(set(u_test_set))
                for i in u_test_set:
                    if i in u_preds:
                        items_l.append(i)

        catalog_hits = np.around(len(set(items_l)) / len(set.union(*u_set_test_items)), 4)
        return catalog_hits

    def eval_group(test_items, group_title):

        if group_title == 't5_cold':
            cold_items = setup_params[consts.cold_items_t5_setup]
        elif group_title == 't10_cold':
            cold_items = setup_params[consts.cold_items_t10_setup]
        elif group_title == 't20_cold':
            cold_items = setup_params[consts.cold_items_t20_setup]
        else:
            cold_items = None

        coverage, mean_recall, mean_precision, mrr = eval_measures_on_topk(all_topk_preds, num_of_users,
                                                                           num_of_items, test_items, cold_items)

        catalog_hits = count_catalog_hits(all_topk_preds, test_items)

        results_df.loc[len(results_df)] = [group_title, mean_recall, mean_precision, mrr, coverage, catalog_hits]

    for g_name, g_test_data in setup_params['test_items_dict'].items():
        print(f'Evaluating group {g_name}')
        eval_group(g_test_data, g_name)
    results_df.to_csv(results_file_path, index=False, mode='a', header=False)
