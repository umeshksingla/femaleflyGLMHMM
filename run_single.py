####################################

# Example usage: python run_single.py  --mc '{"names": "lrhmmci", "seeds": 6205, "num_states": 2, "transition_matrix_stickiness": 10}' --path "general"

####################################

import argparse
import joblib
import os
import json

from hmms.BaseFemaleFly import BaseFemaleFly
from hmms.LRHMMFemaleFly import LRHMMFemaleFly
from hmms.LRFemaleFly import LRFemaleFly
from hmms.GHMMFemaleFly import GHMMFemaleFly
from hmms.ChanceFemaleFly import ChanceFemaleFly
from hmms.LRHMMCustomInitFemaleFly import LRHMMCustomInitFemaleFly
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
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="models/{path} argument",
    )
    return parser


def run(mc):
    if not len(mc): return  # if empty dict is passed

    model_prefix = mc['names']
    print(f"Fitting {model_prefix} with model_config: {mc}")
    data = joblib.load(f'../data/fly_data_cos=4_ortho_o=15.pkl')
    data_config, emissions, inputs = data['data_config'], data['emissions'], data['inputs']
    session_keys = data_config['session_keys']
    output_indices = data['output_indices']
    num_batches = data_config['num_sessions']
    num_train_batches = int(num_batches * 0.8)
    train_emissions, train_inputs, train_session_keys = emissions[:num_train_batches], inputs[:num_train_batches], session_keys[ :num_train_batches]
    test_emissions, test_inputs, test_session_keys = emissions[num_train_batches:], inputs[num_train_batches:], session_keys[num_train_batches:]

    if model_prefix == 'lrhmm':
        model = LRHMMFemaleFly(data_config, mc)
    elif model_prefix == 'ghmm':
        model = GHMMFemaleFly(data_config, mc)
    elif model_prefix == 'lrhmmci':
        model = LRHMMCustomInitFemaleFly(data_config, mc)
    else:
        raise Exception('Unsupported model for cross validation.')

    model.fit(train_emissions, train_inputs)
    dump_filepath = utils.getafilepath(f'{path}/{model.prefix}_{model.model_config["num_states"]}_cv')
    print(">> Saving at:", dump_filepath)
    utils.save(model, train_emissions, train_inputs, train_session_keys, test_emissions, test_inputs, test_session_keys,
               output_indices, dump_filepath)
    print("Saved.\n")
    print(">> Making figures:")
    utils.generate_figures(dump_filepath, savefig=True, display=False)
    print("Done.\n")

    print(model.score(train_emissions, train_inputs))
    print(model.score(test_emissions, test_inputs))
    print(model.score_by_z_and_o(test_emissions, test_inputs))
    print(model.correlation_by_o(test_emissions, test_inputs))
    print("Finished.\n")
    return


if __name__ == '__main__':

    ## If from command line
    parser = create_cli_parser()
    args = parser.parse_args()
    print("Args:", vars(args))
    model_config_str = args.mc
    path = args.path
    model_config = json.loads(model_config_str)

    ## If model_config specified here
    # model_config = {}

    run(model_config)
