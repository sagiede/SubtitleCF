from DB.db_consts import models_dir_path

movies_bert_sentences_dir = models_dir_path + f'/Sents_BERT_Embeddings'
movies_bert_sentences_split_dir = movies_bert_sentences_dir + f'/Bert_embeddings_s5o1'

# # # # # # # #CNN HyperParams # # # # # # # # # #

base_model_name = 'BPR'

split_cnn_train_test = False
n_epochs = 51

reliable_groups_threshold = 5  # learn from top 50% movies
cnn_batch_size = 32
cnn_lr = 1e-4
cnn_wd = 1e-3
dropout_rate = 0.4
cnn_kernel_sizes = [(1, 100)]
maxpool_norm_power = 6
