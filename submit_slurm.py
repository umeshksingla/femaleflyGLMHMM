####################################

# Usage: python submit_slurm.py

####################################

import sys
import json
import random
import itertools
import subprocess


def create_array_file(model_configs):

    arrays = []

    keys = list(model_configs.keys())
    values = [model_configs[k] for k in keys]
    for combo in itertools.product(*values):
        combo_dict = dict(zip(keys, combo))
        arrays.append(json.dumps(combo_dict) + '\n')

    with open('array_args.txt', 'w') as w:
        w.writelines(arrays)

    return arrays


if __name__ == '__main__':

    src = sys.argv[1]

    if src == 'wt':
        data_path = '/scratch/gpfs/MMURTHY/usingla/data/wt_fly_data_cos=4_ortho_o=5_smoothed_stdset.pkl'
    elif src == 'wt_fred':
        data_path = '/scratch/gpfs/MMURTHY/usingla/data/wt_fred_fly_data_cos=4_ortho_o=2_smoothed_stdset.pkl'
    else:
        raise Exception(f'Incorrect data source specified "{src}".')

    path = f'june24_{src}'
    model_configs = {
        'name': ['lrhmmci'],
        'seed': [random.randint(1, 10000) for _ in range(5)],
        'num_states': [
            2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25, 30
        ],
        'transition_matrix_stickiness': [100],
        'data_path': [data_path],
        'path': [path],
    }
    job_configs = create_array_file(model_configs)
    NUM_ARRAY_JOBS = len(job_configs)
    JOB_SCRIPT = 'run_slurm.sh'

    command = [
        "sbatch",
        "-a",
        f"1-{NUM_ARRAY_JOBS}",
        "-J nstates",
        f"{JOB_SCRIPT}",
        f"array_args.txt",
    ]
    print(">>> SLURM COMMAND ran:", " ".join(command))
    subprocess.run(command)

