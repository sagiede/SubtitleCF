import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader

import Consts.consts as consts
import RS.Core.rs_config as conf
from RS.Core.RSModel import RSModel
from RS.rs_utils import save_models_to_file, rs_expr_setup, load_si_vectors, bpr_collate_fn, TripletUniformPair
from general_utils import log_f, print_cuda_memory_status, init_seeds


def build_model(setup_params, si_fvs):
    model = RSModel(setup_params[consts.n_users_setup], setup_params[consts.n_items_setup], conf.dim,
                    setup_params[consts.dev_setup], si_fvs).to(setup_params[consts.dev_setup])

    lr, wd = conf.rs_lr, conf.rs_wd
    params = [{'params': model.user_factors}, {'params': model.item_factors}]
    optimizer = optim.Adam(params, lr=lr, weight_decay=wd)
    model.optimizer = optimizer

    return model


def train_model(model, train_loader, test_loader, setup_params):
    dev = setup_params[consts.dev_setup]
    print(f'Starting Training loop with {conf.n_epochs} epochs')
    log_f(f'Starting Training loop with {conf.n_epochs} epochs')

    init_seeds()
    print_cuda_memory_status('Train Model Start')

    for epoch in range(conf.n_epochs + 1):

        model.train()
        all_bpr_loss_nol2 = []

        for idx, (u_batch, i_batch_pos, i_batch_neg) in enumerate(train_loader):
            u_batch = u_batch.to(dev)
            i_batch_pos = i_batch_pos.to(dev)
            i_batch_neg = i_batch_neg.to(dev)

            model.optimizer.zero_grad()
            bpr_loss, bpr_loss_for_log_nol2 = model.forward(u_batch, i_batch_pos, i_batch_neg)

            bpr_loss.backward()
            model.optimizer.step()

            all_bpr_loss_nol2.extend([t.item() for t in bpr_loss_for_log_nol2])

            if (idx + 1) % 100 == 0:
                print('BPR loss at epoch {}: \t{:f}\t\t epoch_coverage {:f}'.format(epoch, bpr_loss,
                                                                                    idx / len(train_loader)))

        torch.cuda.empty_cache()
        model.eval()

        with torch.no_grad():

            bpr_test_loss_nol2 = []
            for idx, (u_batch, i_batch_pos, i_batch_neg) in enumerate(test_loader):
                _, bpr_loss_for_log_nol2 = model.forward(u_batch, i_batch_pos, i_batch_neg)
                bpr_test_loss_nol2.extend([t.item() for t in bpr_loss_for_log_nol2])

            mean_bpr_test_loss_nol2 = np.mean(bpr_test_loss_nol2)
            mean_bpr_train_loss_nol2 = np.mean(all_bpr_loss_nol2)

            print('BPR Total loss at epoch {}:  bprTrainLoss\t{:f}\tbprTestLoss {:f}\n'.format(epoch,
                                                                                               mean_bpr_train_loss_nol2,
                                                                                               mean_bpr_test_loss_nol2))

            log_f('BPR Total loss at epoch {}:  bprTrainLoss\t{:f}\tbprTestLoss {:f}\n'.format(epoch,
                                                                                               mean_bpr_train_loss_nol2,
                                                                                               mean_bpr_test_loss_nol2))

    save_models_to_file(model, conf.n_epochs)


def rs_experiment():
    setup_params = rs_expr_setup()
    if conf.WITH_SI:
        si_fvs = load_si_vectors(setup_params['train_items'])
    else:
        si_fvs = None
    model = build_model(setup_params, si_fvs)
    dataset = TripletUniformPair(setup_params[consts.n_items_setup], setup_params[consts.users_train_set_items_setup],
                                 setup_params[consts.user_item_train_pairs_setup])
    train_loader = DataLoader(dataset, batch_size=conf.batch_size, shuffle=True, collate_fn=bpr_collate_fn)

    users_combined_set_items = [u_train.union(u_test) for u_train, u_test in
                                zip(setup_params[consts.users_train_set_items_setup],
                                    setup_params[consts.users_test_set_items_setup])]
    dataset = TripletUniformPair(setup_params[consts.n_items_setup], users_combined_set_items,
                                 setup_params[consts.user_item_test_pairs_setup])
    test_loader = DataLoader(dataset, batch_size=conf.batch_size, shuffle=True, collate_fn=bpr_collate_fn)
    train_model(model, train_loader, test_loader, setup_params)

    print('Finished')
    log_f('Finished')
