####################################

# Example usage:
# python run_global.py --mc '{"name": "lr", "path": "general", "data_path": "data/wt_fly_data_cos=4_ortho_o=5.pkl"}'
# python run_global.py --mc '{"name": "lr", "path": "general", "data_path": "data/wt_fly_data_cos=4_ortho_o=5.pkl"}' --enhance
# python run_global.py --mc '{"name": "lr", "path": "general", "data_path": "data/wt_fly_data_cos=4_ortho_o=5.pkl"}' --enhance --genfig

####################################

import argparse
import joblib
import json
import numpy as np

from hmms.LRHMMFemaleFly import LRHMMFemaleFly
from hmms.LogRHMMFemaleFly import LogRHMMFemaleFly
from hmms.LRFemaleFly import LRFemaleFly
from hmms.LogRFemaleFly import LogRFemaleFly
from hmms.GHMMFemaleFly import GHMMFemaleFly
from hmms.ChanceFemaleFly import ChanceFemaleFly
from hmms.LRHMMCustomInitFemaleFly import LRHMMCustomInitFemaleFly
from hmms.LogRHMMCustomInitFemaleFly import LogRHMMCustomInitFemaleFly
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
        "--enhance",
        dest="enhance",
        action="store_true",
        help="Enable enhanced analysis (default is off)",
    )
    parser.add_argument(
        "--genfig",
        dest="genfig",
        action="store_true",
        help="Enable enhanced analysis and generate figures (default is off)",
    )
    return parser


def run(mc, enhance=False, genfig=False):
    if not len(mc): return  # if empty dict is passed

    print(f"Fitting {mc['name']} with model_config: {mc}")

    data = joblib.load(mc['data_path'])
    emissions, inputs, output_mn_std = data['emissions'], data['inputs'], data['output_mn_std']

    data_config = data['data_config']
    print('Inputs:', data_config['input_labels'])
    print('Emissions:', data_config['emission_labels'])

    num_batches = data_config['num_sessions']
    datasplit_seed = mc.get('datasplit_seed', 0)
    num_train_batches = int(num_batches * 0.8)

    # Set up reproducible RNG and shuffle indices
    rng = np.random.default_rng(datasplit_seed)
    all_indices = np.arange(num_batches)
    rng.shuffle(all_indices)

    train_session_indices = all_indices[:num_train_batches].astype(int)
    test_session_indices = all_indices[num_train_batches:].astype(int)

    train_emissions = [emissions[e] for e in train_session_indices]
    train_inputs = [inputs[e] for e in train_session_indices]
    train_output_mn_std = [output_mn_std[e] for e in train_session_indices]

    print("# Train sessions:", train_session_indices, "total=", len(train_session_indices))
    print("# Test sessions:", test_session_indices, "total=", len(test_session_indices))

    model_prefix = mc['name']
    if model_prefix == 'lrhmm':
        model = LRHMMFemaleFly(data_config, mc)
    elif model_prefix == 'logrhmm':
        model = LogRHMMFemaleFly(data_config, mc)
    elif model_prefix == 'ghmm':
        model = GHMMFemaleFly(data_config, mc)
    elif model_prefix == 'lrhmmci':
        model = LRHMMCustomInitFemaleFly(data_config, mc)
    elif model_prefix == 'logrhmmci':
        model = LogRHMMCustomInitFemaleFly(data_config, mc)
    elif model_prefix == 'chance':
        model = ChanceFemaleFly(data_config, mc)
    elif model_prefix == 'lr':
        model = LRFemaleFly(data_config, mc)
    elif model_prefix == 'logr':
        model = LogRFemaleFly(data_config, mc)
    else:
        raise Exception(f'Unsupported model "{model_prefix}" for cross validation.')

    dump_filepath = utils.getafilepath(f'{mc["path"]}/{model.prefix}_{model.model_config["num_states"]}_cv')

    print(">> Fitting")
    model.fit(train_emissions, train_inputs, train_output_mn_std)
    print(">> Fit done.")

    print(">> Saving basic checkpoint at:", dump_filepath)
    utils.save(model, data, train_session_indices, test_session_indices, dump_filepath)   # save model parameters and data used for train and test
    print(">> Saved.\n")

    if enhance or genfig:
        print(">> Saving enhanced checkpoint:")
        utils.enhance(dump_filepath)     # add prediction statistics etc. to the same checkpoint
        print(">> Saved.\n")

        if model.prefix == 'chance': return

        if genfig:
            print(">> Making figures:")
            utils.generate_figures(dump_filepath, savefig=True, display=False)
            print(">> Done with figures.\n")

            print(">> Generating trajectories:")
            utils.generate_trajs(dump_filepath, savefig=True, display=False, gen_corr_video=False)
            print(">> Done with trajectories.\n")

            print(">> Generating videos:")
            utils.generate_state_traces(dump_filepath, savefig=True, display=False)
            utils.generate_state_clips(dump_filepath, savefig=True, display=False, gen_corr_video=True)
            print(">> Done with videos.\n")

    print("Finished.\n")
    return dump_filepath


if __name__ == '__main__':

    ## If from command line
    parser = create_cli_parser()
    args = parser.parse_args()
    print("Args:", vars(args))
    model_config = json.loads(args.mc)
    print(model_config)
    run(model_config, args.enhance, args.genfig)
