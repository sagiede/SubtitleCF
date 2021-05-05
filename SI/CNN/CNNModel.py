import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import SI.CNN.cnn_config as cnn_conf
from SI.CNN.BERT_utils import load_movie_bert_embeddings
from SI.CNN.cnn_utils import pad_bert_sentences_embeddings, calc_nlp_loss


class CNNModel(nn.Module):

    def __init__(self, n_items, items_embs, train_movies_data, out_dim, dev):
        super(CNNModel, self).__init__()
        self.dev = dev
        self.dim = out_dim
        self.item_factors = items_embs
        self.train_movies_data = train_movies_data
        self.n_items = n_items

        embedding_size = 768  # BERT emb size

        filter_size = embedding_size
        # 256 filters of size 3. As you can see in the comment, you can add filters of various sizes.
        kernel_sizes = cnn_conf.cnn_kernel_sizes

        self.convs = nn.ModuleList(
            [nn.Conv2d(1, n_filters, (kernel_size, filter_size)) for kernel_size, n_filters in kernel_sizes])

        conv_emb_len = sum(filters for _, filters in kernel_sizes)

        self.text_fc = nn.Sequential(
            nn.BatchNorm1d(conv_emb_len),
            nn.Dropout(cnn_conf.dropout_rate),
            nn.Linear(conv_emb_len, out_dim),
            nn.ReLU(),
            nn.BatchNorm1d(out_dim),
            nn.Dropout(cnn_conf.dropout_rate),
            nn.Linear(out_dim, out_dim),
            nn.Tanh()
        )

        self.optimizer = None

    def forward(self, i_batch, i_batch_content_embds):
        i_emb_factors_pos = self.item_factors[i_batch, :]

        i_si_factors_pos = self.text_fc_forward(self.conv_forward(i_batch_content_embds))

        nlp_loss, nlp_loss_for_log = calc_nlp_loss(i_emb_factors_pos, i_si_factors_pos)

        return nlp_loss, nlp_loss_for_log

    def conv_forward(self, b_words_embds):
        embeddings = b_words_embds.unsqueeze(1)
        # [batch, 1, words, dim]

        cnn_output = [F.relu(conv(embeddings.float())).squeeze(3) for conv in self.convs]
        # [batch, filter_size, words] * filters

        cnn_output = [torch.norm(i, p=cnn_conf.maxpool_norm_power, dim=2) for i in cnn_output]
        # [batch, filter_size] * filters

        output = torch.cat(cnn_output, 1)
        # [batch, cnn_concat]

        return output

    def text_fc_forward(self, convs_output):
        """
        Args:
            convs_output: [batch, cnn_concat]
        Returns:
            text_rep after fully connected: [batch, text_emb_size]
        """
        return self.text_fc(convs_output)

    def aggregate_all_movies_vectors(self):
        """
        in case of BERT case movies_cnn_data refers to movielens ids
        in case of W2V case movies_cnn_data refers to sentences list
        Returns:
        """

        movies_cnn_data = self.train_movies_data
        dev = self.dev

        with torch.no_grad():
            movies_vectors_after_fc = []
            for batch_index in tqdm(range(0, len(movies_cnn_data), cnn_conf.cnn_batch_size)):
                movies_cnn_data_batch = movies_cnn_data[
                                        batch_index:batch_index + cnn_conf.cnn_batch_size]

                movies_sentences_embeddings = [load_movie_bert_embeddings(movieid) for movieid in
                                               movies_cnn_data_batch]
                cnn_embeddings_pad = pad_bert_sentences_embeddings(movies_sentences_embeddings)

                i_si_convs_embs = self.conv_forward(cnn_embeddings_pad)
                i_si_factors = self.text_fc_forward(i_si_convs_embs)

                movies_vectors_after_fc.extend(list(i_si_factors))
            movies_vectors_after_fc = torch.stack(movies_vectors_after_fc).to(dev)
            return movies_vectors_after_fc
