####################################
# Used in run_slurm.sh
####################################

import argparse
import json
from run_ladder import run_ladder


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
    if not len(mc):
        return  # if empty dict is passed
    print(f"Fitting {mc['name']} with model_config: {mc}")
    paths = run_ladder(mc)
    return paths


if __name__ == '__main__':

    ## If from command line
    parser = create_cli_parser()
    args = parser.parse_args()
    print("Args:", vars(args))
    model_config = json.loads(args.mc)
    print(model_config)
    run(model_config, args.enhance, args.genfig)
