####################################

# Usage: python submit_slurm.py wt female kfold

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
    cvtype = sys.argv[3]    # Specify cvtype = 'kfold' or 'initseeds'

    if src == 'wt':
        if animal == 'female':
            data_path = '/scratch/gpfs/MMURTHY/usingla/data/wt_fly_data_cos=4_ortho_o=5_today=jan1.pkl'
            init_seeds = [5427, 4787, 7896, 5627, 5131, 1818, 65, 8206, 8471, 2734]
            datasplit_seeds = [1326, 6244, 6400, 3733, 2582, 8644, 3930, 7401, 8116, 4335]
        elif animal == 'male':
            data_path = '/scratch/gpfs/MMURTHY/usingla/data/wt_male_fly_data_cos=4_ortho_o=5_today=jan1.pkl'
            init_seeds = [5427, 4787, 7896, 5627, 5131, 1818, 65, 8206, 8471, 2734]
            datasplit_seeds = [1326, 6244, 6400, 3733, 2582, 8644, 3930, 7401, 8116, 4335]
    elif src == 'wt_fred':
        if animal == 'female':
            data_path = '/scratch/gpfs/MMURTHY/usingla/data/wt_fred_fly_data_cos=4_ortho_o=2_today=jan1.pkl'
            init_seeds = [1818, 65, 8206, 8471, 2734, 5427, 4787, 7896, 5627, 5131]
            datasplit_seeds = [8644, 3930, 7401, 8116, 4335, 1326, 6244, 6400, 3733, 2582]
        elif animal == 'male':
            data_path = '/scratch/gpfs/MMURTHY/usingla/data/wt_fred_male_fly_data_cos=4_ortho_o=2_today=jan1.pkl'
            init_seeds = [1818, 65, 8206, 8471, 2734, 5427, 4787, 7896, 5627, 5131]
            datasplit_seeds = [8644, 3930, 7401, 8116, 4335, 1326, 6244, 6400, 3733, 2582]
    else:
        raise Exception(f'Incorrect data source specified "{src}".')

    path = f'jan1_{cvtype}cv_{src}_{animal}'
    model_name = 'idglmhmmci'

    if cvtype == 'kfold':
        init_seeds = [0]        # fix init seed
    elif cvtype == 'initseeds':
        datasplit_seeds = [0]   # fix data split seed

    model_configs = {
        'name': [model_name],
        'seed': init_seeds,
        'datasplit_seed': datasplit_seeds,  #
        'num_states': [
            # 1      # uncomment for chance or lr
            2, 3, 4, 5, 6, 7, 8, 10, #12, 15, 20, 25, 30
            # 8, 10
        ],
        # 'transition_matrix_stickiness': [100],
        'data_path': [data_path],
        'path': [path],
    }
    arrayfilename = f'{path}_{model_name}_array_args.txt'        # CAUTION!! Running this script will overwrite existing array_args file.
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