import torch
from torch import nn as nn
from torch.nn import functional as F

import RS.Core.rs_config as bprws_consts


class RSModel(nn.Module):

    def __init__(self, n_users, n_items, dim, dev, si_fvs=None):
        super().__init__()
        self.dev = dev
        self.dim = dim
        self.n_items = n_items
        self.n_users = n_users
        self.with_si = bprws_consts.WITH_SI

        self.user_factors = nn.Parameter(torch.empty(n_users, dim))
        self.item_factors = nn.Parameter(torch.empty(n_items, dim))

        if self.with_si:
            self.si_fvs = si_fvs

        nn.init.xavier_normal_(self.user_factors.data)
        nn.init.xavier_normal_(self.item_factors.data)

        self.optimizer = None

    def forward(self, u_batch, i_batch_pos, i_batch_neg):

        u_factors = self.user_factors[u_batch]

        if self.with_si:
            i_pos_vectors = self.combine_item_emb_factors_with_si_vectors(i_batch_pos)
            i_neg_vectors = self.combine_item_emb_factors_with_si_vectors(i_batch_neg)
        else:
            i_pos_vectors = self.item_factors[i_batch_pos]
            i_neg_vectors = self.item_factors[i_batch_neg]

        bpr_loss = self.bpr_forward(u_factors, i_pos_vectors, i_neg_vectors)
        bpr_loss_for_log_nol2 = bpr_loss.detach()
        bpr_loss = bpr_loss + self.add_l2_reg(u_factors, i_pos_vectors, i_neg_vectors)

        bpr_mean_loss = bpr_loss.mean()

        return bpr_mean_loss, bpr_loss_for_log_nol2

    def calc_user_ranking(self, user_batch):
        with torch.no_grad():
            u_factors = self.user_factors[user_batch]

            if self.with_si:
                movies_vectors = self.si_fvs
                i_batch = torch.as_tensor(range(movies_vectors.shape[0]), dtype=torch.long, device=self.dev)
                users_results = self.bpr_with_si_eval(u_factors, i_batch)
            else:
                users_results = self.bpr_eval(u_factors, self.item_factors)

        return users_results

    def bpr_with_si_eval(self, u_factors, i_batch):
        i_summed = self.combine_item_emb_factors_with_si_vectors(i_batch)
        return self.bpr_eval(u_factors, i_summed)

    def combine_item_emb_factors_with_si_vectors(self, i_batch):
        i_emb_factors = self.item_factors[i_batch]
        i_si_factors = self.si_fvs[i_batch]
        i_vectors = i_emb_factors + i_si_factors
        return i_vectors

    @staticmethod
    def add_l2_reg(u_factors, i_emb_factors_pos, i_emb_factors_neg):

        u_norm_loss = u_factors.pow(2).sum(1)
        i_norms_loss = i_emb_factors_pos.pow(2).sum(1) + i_emb_factors_neg.pow(2).sum(1)
        norms_loss = (i_norms_loss * bprws_consts.rs_wd_i) + (u_norm_loss * bprws_consts.rs_wd_u)
        return norms_loss

    @staticmethod
    def bpr_forward(u_factors, i_emb_factors_pos, i_emb_factors_neg):
        x_ui = torch.mul(u_factors, i_emb_factors_pos).sum(dim=1)
        x_uj = torch.mul(u_factors, i_emb_factors_neg).sum(dim=1)
        bpr_emb_loss = bpr_loss_function(x_ui, x_uj)
        return bpr_emb_loss

    @staticmethod
    def bpr_eval(u_factors, item_factors):
        x_ui_emb = torch.mm(u_factors, item_factors.t())
        users_results_emb = torch.sigmoid(x_ui_emb)
        return users_results_emb


def bpr_loss_function(x_ui, x_uj):
    """
    Args:
        x_ui: user-pos_item score
        x_uj: user-neg_item score

    Returns: loss value

    """
    return -F.logsigmoid(x_ui - x_uj)
