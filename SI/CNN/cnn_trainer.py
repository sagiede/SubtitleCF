import numpy as np
import pandas as pd
import torch
from torch import optim

import Consts.consts as consts
import SI.CNN.cnn_config as cnn_conf
from RS.Core.rs_config import dim
from SI.CNN.CNNModel import CNNModel
from SI.CNN.cnn_utils import cnn_expr_setup, load_base_model_embeddings, \
    get_cnn_dataloader, save_cnn_model
from general_utils import log_f, print_cuda_memory_status, init_seeds


def build_model(setup_params, items_embs):
    n_items = setup_params[consts.n_items_setup]

    train_movies_data = setup_params[consts.train_movies_cnn_data_setup]

    embs_dim = dim
    dev = setup_params[consts.dev_setup]

    items_embs = items_embs.to(dev)
    model = CNNModel(n_items, items_embs, train_movies_data, embs_dim, dev)
    model = model.to(dev)

    params = [{'params': model.convs.parameters()}, {'params': model.text_fc.parameters()}]
    optimizer = optim.Adam(params, lr=cnn_conf.cnn_lr, weight_decay=cnn_conf.cnn_wd)
    model.optimizer = optimizer

    return model


def cnn_train(model, epoch, items_id_mapping, train_reliable_movies_data):
    movies_data_train = train_reliable_movies_data
    if cnn_conf.split_cnn_train_test:
        movies_data_train, movies_data_test = train_reliable_movies_data['train'], train_reliable_movies_data['test']

    cnn_train_loader = get_cnn_dataloader(movies_data_train, items_id_mapping)

    model.train()
    print('Train Loss:')
    mean_nlp_loss_train = cnn_training_loop(model, cnn_train_loader, epoch, train_mode=True)

    if cnn_conf.split_cnn_train_test:
        cnn_test_loader = get_cnn_dataloader(movies_data_test, items_id_mapping)
        model.eval()
        print('Test Loss:')
        mean_nlp_loss_test = cnn_training_loop(model, cnn_test_loader, epoch, train_mode=False)
    else:
        mean_nlp_loss_test = None

    return mean_nlp_loss_train, mean_nlp_loss_test


def cnn_training_loop(model, cnn_loader, epoch, train_mode):
    all_nlp_loss = []

    for idx, (i_batch, i_batch_words_embds) in enumerate(cnn_loader):

        torch.cuda.empty_cache()

        model.optimizer.zero_grad()

        nlp_loss, nlp_loss_for_log = model.forward(i_batch, i_batch_words_embds)

        losses = [t.item() for t in nlp_loss_for_log]
        all_nlp_loss.extend(losses)

        if train_mode:
            nlp_loss.backward()
            model.optimizer.step()

    mean_nlp_loss = np.nanmean(all_nlp_loss)
    if train_mode:
        mode_str = 'Train'
    else:
        mode_str = 'Test'
    print(f'Epoch {epoch} - {mode_str} Items Nlp_loss: {mean_nlp_loss}, std: {np.nanstd(all_nlp_loss)}')
    log_f(f'Epoch {epoch} - {mode_str} CNN Items Nlp_loss: {mean_nlp_loss}, std: {np.nanstd(all_nlp_loss)}')
    return mean_nlp_loss


def cnn_experiment(results_file_path):
    setup_params = cnn_expr_setup()

    items_embs = load_base_model_embeddings()
    model = build_model(setup_params, items_embs)

    print(f'Starting Training loop with {cnn_conf.n_epochs} epochs')
    log_f(f'Starting Training loop with {cnn_conf.n_epochs} epochs')

    titles = {'Epoch': [], consts.train_nlp_loss_title: [], consts.test_nlp_loss_title: []}
    results_df = pd.DataFrame(data=titles)

    init_seeds()
    print_cuda_memory_status('Train Model Start')

    for epoch in range(cnn_conf.n_epochs):
        model.train()

        mean_nlp_loss_train, mean_nlp_loss_test = cnn_train(model, epoch,
                                                            setup_params[consts.items_id_mapping_setup],
                                                            setup_params[consts.train_reliable_movies_cnn_data_setup])

        results_df.loc[len(results_df)] = [int(epoch), mean_nlp_loss_train, mean_nlp_loss_test]
        results_df.to_csv(results_file_path, index=False)

    movies_vectors_after_fc = model.aggregate_all_movies_vectors()
    save_cnn_model(consts.Experiment_suffix, movies_vectors_after_fc, model)

    print('Finished')
    log_f('Finished')
