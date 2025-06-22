####################################

# Example usage: python run_single.py  --mc '{"names": "lrhmmci", "seeds": 6205, "num_states": 2, "transition_matrix_stickiness": 10}' --path "general"
# OR
# python run_single.py --mc '{"names": "chance"}' --path "general"

####################################

import argparse
import os.path
import sys

import joblib
import json
import numpy as np

from hmms.LRHMMIndFemaleFly import LRHMMIndFemaleFly
from hmms.LRFemaleFly import LRFemaleFly
from hmms.LRHMMCustomInitFemaleFly import LRHMMCustomInitFemaleFly
from utilities import utils
import run_global


def create_cli_parser():
    parser = argparse.ArgumentParser(
        description="Model config."
    )
    parser.add_argument(
        "--mc",
        type=str,
        required=True,
        help="model config",
    )
    return parser


def fit_individual(s, global_params, data_varlen, data_config, mc, ind_dump_filepath, trained_bool):

    print(f"Session {s}: Fitting..")

    emissions, inputs = data_varlen['emissions'], data_varlen['inputs']
    copulation_bools = data_varlen['copulation_bools']
    all_session_keys = data_config['session_keys']

    s_i = all_session_keys.index(s)

    e = emissions[s_i][None]
    i = inputs[s_i][None]
    cb = copulation_bools[s_i]
    model_prefix = mc['name']

    if model_prefix == 'lrhmmci':
        imodel = LRHMMIndFemaleFly(data_config, mc)
    elif model_prefix == 'lr':
        imodel = LRFemaleFly(data_config, mc)
    else:
        raise Exception(f'Unsupported model "{model_prefix}" for individual data fits.')

    imodel.fit(global_params, e, i)
    print(f"Session {s_i}: Fit done.", len(imodel.learned_lps))
    try:
        print(f"Session {s_i} r2: ", imodel.score(e, i))
        dump_filepath_single = os.path.join(ind_dump_filepath, f'session{s_i}')
        utils.save_single(imodel, e, i, cb, trained_bool, dump_filepath_single)
        utils.generate_figures_single(dump_filepath_single)
        print(f"Session {s_i}: Figures generated. Saved at {dump_filepath_single}")
    except ValueError:
        print(f"Session {s_i}: NaNed.")
    print("==========")
    return imodel


def run(mc):

    if not len(mc):
        return  # if empty dict is passed

    print(">> Fitting global fit")
    global_mc = {
        "name": mc['name'],
        "seed": mc['seed'],
        "num_states": mc['num_states'],
        "transition_matrix_stickiness": 100,
        "path": mc['path'],
        "data_path": mc['data_path_fixlen'],
    }

    # Fit with fixed length data
    global_dump_filepath = run_global.run(global_mc)
    print("Global model dumped at:", global_dump_filepath)

    # Now load the sessions used for training and testing in global fit
    # model_pkl, _, _ = utils.load_specific_path(global_dump_filepath)
    # train_session_keys = model_pkl['train_data']['train_session_keys']
    # test_session_keys = model_pkl['test_data']['test_session_keys']
    # global_params = model_pkl['learned_params']

    data_varlen = joblib.load(mc['data_path_varlen'])
    # data_config = data_varlen['data_config']
    #
    # ind_dump_filepath_train = os.path.join(global_dump_filepath, 'individual_train')
    # ind_dump_filepath_test = os.path.join(global_dump_filepath, 'individual_test')
    # mc['data_path'] = mc['data_path_varlen']
    #
    # print(">> Fitting each train session separately:")
    # for s in train_session_keys:
    #     fit_individual(s, global_params, data_varlen, data_config, mc, ind_dump_filepath_train, trained_bool=True)
    # print(">> Individual train fits done.")
    # utils.generate_figures_all_singles_merged(ind_dump_filepath_train)
    #
    # print(">> Fitting each test session separately:")
    # for s in data_config['session_keys']:   # sessions for fixed are less than total, so we have more sessions held out than just in test_session_keys, especially the shorter ones
    #     if s in train_session_keys:
    #         continue
    #     fit_individual(s, global_params, data_varlen, data_config, mc, ind_dump_filepath_test, trained_bool=False)
    # print(">> Individual test fits done.")
    # utils.generate_figures_all_singles_merged(ind_dump_filepath_test)
    return


if __name__ == '__main__':

    dataset = 'wt'
    run({
        "name": 'lr',
        "seed": 2244,
        "num_states": 1,
        # "transition_matrix_stickiness": 100,
        "path": f'general_{dataset}',

        "data_path_fixlen": f'data/{dataset}_fly_data_cos=4_ortho_o=5_smoothed.pkl',
        # "data_path_varlen": f'data/{dataset}_fly_data_cos=4_ortho_o=5_fixlen=False.pkl',
    })

