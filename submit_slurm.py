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
    model_configs = {
        'names': ['lrhmm', 'ghmm'],
        'seeds': [random.randint(1, 10000) for _ in range(2)],
        'num_states': [
            2,
            #3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 40, 50
        ],
        'transition_matrix_stickiness': [10],
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
    # subprocess.run(command)


