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


if __name__ == '__main__':
    dataset = 'wt_fred'
    if dataset == 'wt':
        data_path = f'data/wt_fly_data_cos=4_ortho_o=5_smoothed_stdset_auxem_0723.pkl'
    elif dataset == 'wt_shuffled':
        data_path = f'data/wt_fly_data_cos=4_ortho_o=5_smoothed_stdset_auxem_shuffled.pkl'
    elif dataset == 'wt_male':
        data_path = f'data/wt_fly_data_cos=4_ortho_o=5_smoothed_stdset_auxem_MALE.pkl'
    elif dataset == 'wt_fred':
        data_path = f'data/wt_fred_fly_data_cos=4_ortho_o=2_smoothed_stdset_auxem_0723.pkl'
    else:
        raise Exception(f'Wrong dataset {dataset} specified.')
    mc = {
        "name": 'lrhmmci',
        "seed": 3242,
        "num_states": 5,
        "transition_matrix_stickiness": 100,
        "path": f'general_{dataset}_lr_temp',
        "data_path": data_path,
    }

    print(">> Fitting global fit")
    global_dump_filepath = run_global.run(mc, genfig=True)
    print("Global model dumped at:", global_dump_filepath)
