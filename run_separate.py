####################################

# Example usage: python run_separate.py

####################################

import run_global


if __name__ == '__main__':
    dataset = 'wt'
    if dataset == 'wt':
        data_path = f'../data/wt_fly_data_cos=4_ortho_o=5_today=jan1.pkl'
    elif dataset == 'wt_acc':
        data_path = f'../data/wt_fly_data_cos=4_ortho_o=5_shuffle_inputs=False_accelerations_today=jan1.pkl'
    elif dataset == 'wt_shuffled':
        data_path = f'../data/wt_fly_data_cos=4_ortho_o=5_shuffle_inputs=True_today=jan1.pkl'
    elif dataset == 'wt_male':
        data_path = f'../data/wt_male_fly_data_cos=4_ortho_o=5_today=apr29.pkl'
    elif dataset == 'wt_fred':
        data_path = f'../data/wt_fred_fly_data_cos=4_ortho_o=2_today=jan1.pkl'
    elif dataset == 'wt_fred_male':
        data_path = f'../data/wt_fred_male_fly_data_cos=4_ortho_o=2_today=jan1.pkl'
    else:
        raise Exception(f'Wrong dataset {dataset} specified.')
    mc = {
        "name": 'idglmhmmci',
        "seed": 5427,
        "datasplit_seed": 0,
        "num_states": 5,
        "transition_matrix_stickiness": 100,
        "l2_penalty": 1000,
        'split': 1,
        "path": f'general_{dataset}',
        "data_path": data_path,
    }

    print(">> Fitting global fit")
    global_dump_filepath = run_global.run(mc, enhance=True, genfig=True)
    print("Global model dumped at:", global_dump_filepath)
