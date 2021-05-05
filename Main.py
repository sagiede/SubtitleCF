import argparse
import traceback

import Consts.consts as consts
import RS.Core.rs_config as conf
from DB.db_consts import get_results_path, ml_20m_dataset_name
from DB.preprocess import run_preprocess
from RS.Core.rs_trainer import rs_experiment, build_model
from RS.evaluation_utils import evaluate_ranking_model
from RS.rs_utils import rs_expr_setup, load_si_vectors, update_model_from_file, init_rs_params
from SI.CNN.BERT_utils import extract_bert_sentences
from SI.CNN.cnn_trainer import cnn_experiment
from SI.LDA.lda_utils import create_lda_embedding_vectors
from SI.Longformer.longformer_utils import create_longformer_feats, reduce_longformer_feats
from general_utils import log_f, init_seeds

train_rs_exp = 'train_rs'
test_rs_exp = 'test_rs'
create_si_vectors = 'create_si_vectors'
preprocess_data = 'preprocess_data'
create_sentences_bert_embeddings = 'create_sentences_bert_embeddings'
train_cnn_model = 'train_cnn_model'

run_names = [train_rs_exp,
             test_rs_exp,
             create_si_vectors,
             preprocess_data,
             create_sentences_bert_embeddings,
             train_cnn_model]

rs_model_names = [consts.bpr_rs_model_name,
                  consts.lda_rs_model_name,
                  consts.cnn_rs_model_name,
                  consts.longformer_rs_model_name]

si_model_names = ['lda', 'longformer']

set_args_from_code = False


def set_args(argument):
    argument.run = train_rs_exp
    argument.rsname = consts.longformer_rs_model_name
    argument.siname = 'lda'


def main():
    parser = argparse.ArgumentParser(description="Running The Code arguments:")
    parser.add_argument("-r", "--run",
                        help=f"Options: {', '.join(run_names)}", required=True,
                        default="")
    parser.add_argument("-rsn", "--rsname", help=f"RS Options: {', '.join(rs_model_names)}",
                        required=False, default="")
    parser.add_argument("-sin", "--siname", help=f"SI Options (for creating_si_vectors): {', '.join(si_model_names)}",
                        required=False, default="")
    parser.add_argument("-c", "--cuda", help="T / F (T for cuda , F for cpu) - default to cuda", required=False,
                        default="")

    argument = parser.parse_args()
    err_msg = 'Invalid Input'

    if set_args_from_code:
        set_args(argument)

    try:
        experiment = argument.run
        if argument.cuda == 'F':
            consts.use_cuda = False

        print(
            'Running {} {} experiment on {} Dataset'.format(experiment, consts.Experiment_suffix, ml_20m_dataset_name))
        log_f(f'Running experiment {consts.Experiment_suffix}')
        init_seeds()

        if experiment == preprocess_data:
            run_preprocess()

        if experiment == create_sentences_bert_embeddings:
            extract_bert_sentences()

        if experiment == train_cnn_model:
            results_file_path = get_results_path()
            cnn_experiment(results_file_path=results_file_path)

        if experiment in [train_rs_exp, test_rs_exp]:
            if not argument.rsname:
                print(err_msg)
                return
            else:
                model_name = argument.rsname

            if experiment == train_rs_exp:
                if model_name not in rs_model_names:
                    print('Bad model name for train_rs_exp')
                    return

                init_rs_params(model_name, mode='train')
                rs_experiment()

            if experiment == test_rs_exp:
                if model_name not in rs_model_names:
                    print('Bad model name for test_rs_exp')
                    return

                init_rs_params(model_name, mode='test')
                results_file_path = get_results_path()
                setup_params = rs_expr_setup()
                if conf.WITH_SI:
                    si_fvs = load_si_vectors(setup_params['train_items'])
                else:
                    si_fvs = None
                rs_model = build_model(setup_params, si_fvs)
                update_model_from_file(model_name, rs_model)
                evaluate_ranking_model(rs_model=rs_model, results_file_path=results_file_path,
                                       setup_params=setup_params)

            if experiment == create_si_vectors:

                if not argument.sisname:
                    print(err_msg)
                    return
                else:
                    model_name = argument.sisname

                if model_name not in si_model_names:
                    print('Bad model name for create_si_vectors')
                    return

                if model_name == 'lda':
                    create_lda_embedding_vectors()
                elif model_name == 'longformer':
                    create_longformer_feats()
                    reduce_longformer_feats()
                else:
                    print('Unknown model in create_si_vectors')
                    return

    except Exception as e:
        log_f(traceback.format_exc())
        raise e

    print('Finished Expr')
    log_f('Finished Expr')


if __name__ == '__main__':
    main()
