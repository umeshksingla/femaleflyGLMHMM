####################################

# Example usage: python run_single.py  --mc '{"names": "lrhmmci", "seeds": 6205, "num_states": 2, "transition_matrix_stickiness": 10}' --path "general"
# OR
# python run_single.py --mc '{"names": "chance"}' --path "general"

####################################

import argparse
import os.path

import joblib
import json
import numpy as np

from hmms.LRHMMIndFemaleFly import LRHMMIndFemaleFly
from utilities import utils


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


def run(mc):
    if not len(mc): return  # if empty dict is passed

    path = mc['path']
    data_path = mc['data_path']
    # model_prefix = mc['names']
    # print(f"Fitting {model_prefix} with model_config: {mc}")

    data = joblib.load(data_path)
    emissions, inputs, output_mn_std = data['emissions'], data['inputs'], data['output_mn_std']
    copulation_bools = data['copulation_bools']

    data_config = data['data_config']
    # print('Inputs:', data_config['input_labels'])
    # print('Emissions:', data_config['emission_labels'])

    num_batches = data_config['num_sessions']
    num_train_batches = int(num_batches * 0.8)
    train_session_indices = np.arange(num_train_batches).astype(int)
    test_session_indices = np.arange(num_train_batches, num_batches).astype(int)

    # train_emissions = emissions[train_session_indices]
    # test_emissions = emissions[test_session_indices]
    # train_inputs = inputs[train_session_indices]
    # test_inputs = inputs[test_session_indices]
    # train_output_mn_std = output_mn_std[train_session_indices]

    print("# Train sessions:", train_session_indices, len(train_session_indices))
    print("# Test sessions:", test_session_indices, len(test_session_indices))
    return

    # print(">> Fitting global fit")
    # global_model = LRHMMCustomInitFemaleFly(data_config, mc)
    # global_model.fit(train_emissions, train_inputs, train_output_mn_std)
    # print("LRHMM global W and b computed.")
    # print(">> Global fit done.")
    # global_params = global_model.learned_params
    # print("global_params", global_params)
    # print("global_lps", global_model.learned_lps)
    # joblib.dump(global_params, 'global_params.pkl')

    global_params = joblib.load('global_params.pkl')

    dump_filepath = utils.getafilepath(f'{path}/lrhmmci2_4')

    print(">> Fitting each session separately:")
    for s_i in range(num_batches):
        print(f"Session {s_i}: Fitting..")

        e = emissions[s_i][None]
        i = inputs[s_i][None]
        cb = copulation_bools[s_i]
        used_for_global_training = s_i in train_session_indices

        imodel = LRHMMIndFemaleFly(data_config, mc)
        imodel.fit(global_params, e, i)
        print(f"Session {s_i}: Fit done.", len(imodel.learned_lps))
        try:
            print(f"Session {s_i} r2: ", imodel.score(e, i))
            dump_filepath_single = os.path.join(dump_filepath, f'session{s_i}')
            utils.save_single(imodel, e, i, cb, used_for_global_training, dump_filepath_single)
            # utils.generate_figures_single(dump_filepath_single)
            print(f"Session {s_i}: Figures generated. Saved at {dump_filepath_single}")
        except ValueError:
            print(f"Session {s_i}: NaNed.")
            continue
        print("=====")
    print(">> Individual fits done.")

    utils.generate_figures_all_singles_merged(dump_filepath)
    return


if __name__ == '__main__':

    ## If from command line
    parser = create_cli_parser()
    args = parser.parse_args()
    print("Args:", vars(args))
    model_config = json.loads(args.mc)
    run(model_config)
