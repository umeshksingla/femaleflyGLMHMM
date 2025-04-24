####################################

# Example usage: python run_single.py  --mc '{"names": "lrhmmci", "seeds": 6205, "num_states": 2, "transition_matrix_stickiness": 10}' --path "general"
# OR
# python run_single.py --mc '{"names": "chance"}' --path "general"

####################################

import argparse
import joblib
import json

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
    return parser


def run(mc):
    if not len(mc): return  # if empty dict is passed

    path = mc['path']
    data_path = mc['data_path']
    model_prefix = mc['names']

    print(f"Fitting {model_prefix} with model_config: {mc}")
    data = joblib.load(data_path)
    data_config, emissions, inputs = data['data_config'], data['emissions'], data['inputs']
    print('Inputs:', data_config['input_labels'])
    print('Emissions:', data_config['emission_labels'])
    session_keys = data_config['session_keys']
    output_indices = data['output_indices']
    num_batches = data_config['num_sessions']
    num_train_batches = int(num_batches * 0.8)
    train_emissions, train_inputs, train_session_keys = emissions[:num_train_batches], inputs[:num_train_batches], session_keys[ :num_train_batches]
    test_emissions, test_inputs, test_session_keys = emissions[num_train_batches:], inputs[num_train_batches:], session_keys[num_train_batches:]
    print("# Train sessions:", len(train_session_keys))
    print("# Test sessions:", len(test_session_keys))

    if model_prefix == 'lrhmm':
        model = LRHMMFemaleFly(data_config, mc)
    elif model_prefix == 'ghmm':
        model = GHMMFemaleFly(data_config, mc)
    elif model_prefix == 'lrhmmci':
        model = LRHMMCustomInitFemaleFly(data_config, mc)
    elif model_prefix == 'chance':
        model = ChanceFemaleFly(data_config, mc)
    elif model_prefix == 'lr':
        model = LRFemaleFly(data_config, mc)
    else:
        raise Exception(f'Unsupported model "{model_prefix}" for cross validation.')

    print(">> Fitting")
    model.fit(train_emissions, train_inputs)
    print(">> Fit done.")

    dump_filepath = utils.getafilepath(f'{path}/{model.prefix}_{model.model_config["num_states"]}_cv')

    print(">> Saving basic checkpoint at:", dump_filepath)
    utils.save(model, train_emissions, train_inputs, train_session_keys, test_emissions, test_inputs, test_session_keys,
               output_indices, dump_filepath)   # save model parameters and data used for train and test
    print(">> Saved.\n")

    print(">> Calculating overall r2 scores for this fit:")
    print("train r2: ", model.score(train_emissions, train_inputs))
    print("test r2: ", model.score(test_emissions, test_inputs))

    print(">> Saving enhanced checkpoint:")
    utils.enhance(dump_filepath)     # add prediction statistics etc. to the same checkpoint
    print(">> Saved.\n")

    if model.prefix == 'chance': return

    print(">> Making figures:")
    utils.generate_figures(dump_filepath, savefig=True, display=False)
    print(">> Done with figures.\n")

    print("Finished.\n")
    return


if __name__ == '__main__':

    ## If from command line
    parser = create_cli_parser()
    args = parser.parse_args()
    print("Args:", vars(args))
    model_config = json.loads(args.mc)
    run(model_config)
