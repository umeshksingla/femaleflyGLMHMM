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


# def create_cli_parser():
#     parser = argparse.ArgumentParser(
#         description="Model config."
#     )
#     parser.add_argument(
#         "--mc",
#         type=str,
#         required=True,
#         help="model config",
#     )
#     return parser


# def fit_individual(s, global_params, data_varlen, data_config, mc, ind_dump_filepath, trained_bool):
#
#     print(f"Session {s}: Fitting..")
#
#     emissions, inputs = data_varlen['emissions'], data_varlen['inputs']
#     copulation_bools = data_varlen['copulation_bools']
#     all_session_keys = data_config['session_keys']
#
#     s_i = all_session_keys.index(s)
#
#     e = emissions[s_i][None]
#     i = inputs[s_i][None]
#     cb = copulation_bools[s_i]
#     model_prefix = mc['name']
#
#     if model_prefix == 'lrhmmci':
#         imodel = LRHMMIndFemaleFly(data_config, mc)
#     elif model_prefix == 'lr':
#         imodel = LRFemaleFly(data_config, mc)
#     else:
#         raise Exception(f'Unsupported model "{model_prefix}" for individual data fits.')
#
#     imodel.fit(global_params, e, i)
#     print(f"Session {s_i}: Fit done.", len(imodel.learned_lps))
#     try:
#         print(f"Session {s_i} r2: ", imodel.score(e, i))
#         dump_filepath_single = os.path.join(ind_dump_filepath, f'session{s_i}')
#         utils.save_single(imodel, e, i, cb, trained_bool, dump_filepath_single)
#         utils.generate_figures_single(dump_filepath_single)
#         print(f"Session {s_i}: Figures generated. Saved at {dump_filepath_single}")
#     except ValueError:
#         print(f"Session {s_i}: NaNed.")
#     print("==========")
#     return imodel


if __name__ == '__main__':
    dataset = 'wt'
    mc = {
        "name": 'lrhmmci',
        "seed": 64378,
        "num_states": 30,
        "transition_matrix_stickiness": 100,
        "path": f'general_{dataset}',
        "data_path": f'data/{dataset}_fly_data_cos=4_ortho_o=5_smoothed_stdset.pkl',
        # "data_path_varlen": f'data/{dataset}_fly_data_cos=4_ortho_o=5_fixlen=False.pkl',
    }

    print(">> Fitting global fit")
    global_dump_filepath = run_global.run(mc)
    print("Global model dumped at:", global_dump_filepath)
