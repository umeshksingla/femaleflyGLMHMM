####################################

# Usage: python submit_slurm.py wt female cv
# OR
# Usage: python submit_slurm.py wt female all

####################################

import sys
import json
import random
import itertools
import subprocess


def create_array_file(model_configs, arrayfilename):

    arrays = []

    keys = list(model_configs.keys())
    values = [model_configs[k] for k in keys]
    for combo in itertools.product(*values):
        combo_dict = dict(zip(keys, combo))
        arrays.append(json.dumps(combo_dict) + '\n')

    with open(arrayfilename, 'w') as w:
        w.writelines(arrays)

    return arrays


if __name__ == '__main__':

    src = sys.argv[1]       # Specify src = 'wt' or 'wt_fred'
    animal = sys.argv[2]    # Specify animal = 'female' or 'male'
    fold = sys.argv[3]      # Specify fold = 'cv' or 'all'

    init_seeds = [5427, 4787, 7896, 5627, 5131]
    if src == 'wt':
        if animal == 'female':
            data_path = '/scratch/gpfs/MMURTHY/usingla/data/wt_fly_data_cos=4_ortho_o=5_today=jan1.pkl'
        elif animal == 'male':
            data_path = '/scratch/gpfs/MMURTHY/usingla/data/wt_male_fly_data_cos=4_ortho_o=5_today=jan1.pkl'
    elif src == 'wt_fred':
        if animal == 'female':
            data_path = '/scratch/gpfs/MMURTHY/usingla/data/wt_fred_fly_data_cos=4_ortho_o=2_today=jan1.pkl'
        elif animal == 'male':
            data_path = '/scratch/gpfs/MMURTHY/usingla/data/wt_fred_male_fly_data_cos=4_ortho_o=2_today=jan1.pkl'
    else:
        raise Exception(f'Incorrect data source specified "{src}".')

    model_name = 'ladder'
    path = f'octladder_cv_{src}_{animal}'

    if fold == 'cv':            # Use this for each CV fold on train/test flies.
        model_configs = {
            'name': [model_name],
            'init_seeds': [init_seeds[:5]],
            'fold': list(range(10)),
            'num_states': [
                2, 3, 4, 5, 6, 7, 8
            ],
            'transition_matrix_stickiness': [100],
            'emission_l2_grid': [[1e2, 1e4, 3e4, 1e5, 3e5, 1e6, 1e8]],
            'l2_grid': [[1e2, 1e3, 1e4, 1e5, 1e6, 1e8]],
            'l1_grid': [[1]],
            'data_path': [data_path],
            'path': [path]
        }
    elif fold == 'all':         # Use this for the final fold on all flies.
        model_configs = {
            'name': [model_name],
            'init_seeds': [init_seeds[:5]],
            'fold': ['all'],
            'num_states': [
                5
            ],
            'transition_matrix_stickiness': [100],
            'emission_l2_penalty': [3e5],
            'l2_penalty': [1e3],
            'l1_penalty': [1],
            'data_path': [data_path],
            'path': [path],
            'tune': [False],
        }

    arrayfilename = f'{path}_{model_name}_{fold}_array_args.txt'        # CAUTION!! Running this script will overwrite existing array_args file.
    job_configs = create_array_file(model_configs, arrayfilename=arrayfilename)
    NUM_ARRAY_JOBS = len(job_configs)
    JOB_SCRIPT = 'run_slurm.sh'

    command = [
        "sbatch",
        "-a",
        f"1-{NUM_ARRAY_JOBS}",
        f"-J {src}_{animal}",
        f"{JOB_SCRIPT}",
        f"{arrayfilename}",
    ]
    print(">>> SLURM COMMAND ran:", " ".join(command))
    subprocess.run(command)


#SBATCH --mem=64GB
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
