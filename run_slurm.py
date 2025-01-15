import argparse
import joblib

from hmms.LRHMMFemaleFly import LRHMMFemaleFly
from hmms.GHMMFemaleFly import GHMMFemaleFly
from utils import utils


def create_cli_parser():
    """Create an argument parser for the command line interface."""
    parser = argparse.ArgumentParser(
        description="Model config."
    )
    parser.add_argument(
        "-mc",
        type=str,
        required=True,
        help="model config",
    )
    return parser


if __name__ == '__main__':

    parser = create_cli_parser()
    args = parser.parse_args()
    print("Args:", vars(args))
    print()

    model_config_str = args.mc
    model_config = model_config_str

    model_prefix = model_config['names']
    print(f"Fitting {model_prefix} with model_config: {model_config}")
    data = joblib.load(f'../data/fly_data_cos=4_ortho_o=15.pkl')
    data_config, emissions, inputs = data['data_config'], data['emissions'], data['inputs']
    session_keys = data_config['session_keys']
    output_indices = data['output_indices']
    num_batches = data_config['num_sessions']
    num_train_batches = int(num_batches * 0.8)
    train_emissions, train_inputs, train_session_keys = emissions[:num_train_batches], inputs[ :num_train_batches], session_keys[ :num_train_batches]
    test_emissions, test_inputs, test_session_keys = emissions[num_train_batches:], inputs[ num_train_batches:], session_keys[ num_train_batches:]

    if model_prefix == 'lrhmm':
        model = LRHMMFemaleFly(data_config, model_config)
    elif model_prefix == 'ghmm':
        model = GHMMFemaleFly(data_config, model_config)
    else:
        raise Exception('Unsupported model for cross validation.')
    model.fit(train_emissions, train_inputs)
    dump_filepath = utils.getafilepath(f'cv/{model.prefix}_{model.model_config["num_states"]}_cv')
    print(">> Saving at:", dump_filepath)
    utils.save(model, train_emissions, train_inputs, train_session_keys, test_emissions, test_inputs, test_session_keys, output_indices, dump_filepath)
    print("Finished.\n")

