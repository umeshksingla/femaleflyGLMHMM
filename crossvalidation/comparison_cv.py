import glob

import matplotlib.pyplot as plt
import joblib

import numpy as np
from utils import utils
from hmms.LRHMMFemaleFly import LRHMMFemaleFly
from hmms.GHMMFemaleFly import GHMMFemaleFly


def run(model_prefix, model_config):
    print(f"Fitting with seed: {model_config['seed']}")
    if model_prefix == 'lrhmm':
        model = LRHMMFemaleFly(data_config, model_config)
    elif model_prefix == 'ghmm':
        model = GHMMFemaleFly(data_config, model_config)
    else:
        raise Exception('Unsupported model for cross validation.')
    model.fit(train_emissions, train_inputs)
    dump_filepath = utils.getafilepath(f'{model.prefix}_{model.model_config["num_states"]}_cv')
    print(">> Saving at:", dump_filepath)
    utils.save(model, train_emissions, train_inputs, test_emissions, test_inputs, dump_filepath)
    print("Finished.\n")
    return


def runCV(model_prefix, model_configs):

    model_config = {}
    for seed in model_configs['seeds']:
        model_config['seed'] = int(seed)
        for s in model_configs['num_states']:
            model_config['num_states'] = s
            for t in model_configs['transition_matrix_stickiness']:
                model_config['transition_matrix_stickiness'] = t
                run(model_prefix, model_config)
    return


def loadCV(model_prefix, num_states):
    model_pkls = glob.glob(f'models/{model_prefix}_{num_states}_cv/**/')
    train_lps = []
    test_lps = []
    for _ in model_pkls:
        pkl, _, _ = utils.load_specific_path(_)
        train_lps.append(pkl['train_data']['train_lp'].item())
        test_lps.append(pkl['test_data']['test_lp'].item())
        print(train_lps[-1], test_lps[-1])
    return np.array(train_lps), np.array(test_lps)


def plotCV_same_model(model_configs):

    chance_pkl, _, _ = utils.load_specific_path('models/chance/20241208_133339_lose')

    baseline = chance_pkl['test_data']['test_lp']
    effective_fps = 150 // data_config['predict_window_size']

    plt.figure()

    for i, s in enumerate(model_configs['num_states']):
        lrhmm_train_lps, lrhmm_test_lps = loadCV('lrhmm', s)
        print(f"num_states={s} Train: {lrhmm_train_lps} Test:{lrhmm_test_lps}")
        x = [s + np.random.uniform(-0.1, 0.1) for _ in lrhmm_train_lps]
        plt.plot(x, (lrhmm_train_lps - baseline)*effective_fps, 'b.', label='Train' if i == 0 else '')
        plt.plot(x, (lrhmm_test_lps - baseline)*effective_fps, 'r.', label='Test' if i == 0 else '')

    plt.ylabel('Normalized LL (bits/s)')
    plt.xlabel('Number of states')
    plt.xticks(model_configs['num_states'])
    plt.title('LR-HMM')
    plt.legend()
    plt.margins(0.1)
    plt.tight_layout()
    plt.savefig('models/lrhmm_cv.pdf', bbox_inches='tight', dpi=300)
    # plt.show()
    return


def plotCV_different_models(num_states):

    lrhmm_train_lps, lrhmm_test_lps = loadCV('lrhmm', num_states)
    ghmm_train_lps, ghmm_test_lps = loadCV('ghmm', num_states)

    print(f"LR-HMM num_states={num_states} Train: {lrhmm_train_lps} Test:{lrhmm_test_lps}")
    print(f"G-HMM num_states={num_states} Train: {ghmm_train_lps} Test:{ghmm_test_lps}")

    lr_pkl, _, lr_config = utils.load_specific_path('../models/lr/20241208_134048_hog')
    chance_pkl, _, _ = utils.load_specific_path('models/chance/20241208_133339_lose')

    baseline = chance_pkl['test_data']['test_lp']
    effective_fps = 150 // data_config['predict_window_size']

    plt.figure()

    # lr
    plt.plot(0, (lr_pkl['train_data']['train_lp'] - baseline)*effective_fps, 'b.', label='Train')
    plt.plot(0, (lr_pkl['test_data']['test_lp'] - baseline)*effective_fps, 'r.', label='Train')

    # lrhmm
    x = [1 + np.random.uniform(-0.1, 0.1) for _ in lrhmm_train_lps]
    plt.plot(x, (lrhmm_train_lps - baseline)*effective_fps, 'b.', label='Train')
    plt.plot(x, (lrhmm_test_lps - baseline)*effective_fps, 'r.', label='Test')

    # ghmm
    x = [2 + np.random.uniform(-0.1, 0.1) for _ in ghmm_train_lps]
    plt.plot(x, (ghmm_train_lps - baseline)*effective_fps, 'b.')
    plt.plot(x, (ghmm_test_lps - baseline)*effective_fps, 'r.')

    plt.xticks([1, 2], ['LR-HMM', 'G-HMM'])
    plt.ylabel('Normalized LL (bits/s)')
    plt.title(f'Multiple seeds (num_states={num_states})')
    plt.legend()
    plt.margins(0.1)
    plt.tight_layout()
    # plt.savefig(f'models/cv_{num_states}.pdf', bbox_inches='tight', dpi=300)
    plt.show()
    return


if __name__ == '__main__':

    data = joblib.load(f'data/fly_data_cos=4_o=15_LRfull.pkl')
    data_config, emissions, inputs = data['data_config'], data['emissions'], data['inputs']
    num_batches = data_config['num_sessions']
    num_train_batches = int(num_batches * 0.8)
    train_emissions, train_inputs = emissions[:num_train_batches], inputs[:num_train_batches]
    test_emissions, test_inputs = emissions[num_train_batches:], inputs[num_train_batches:]

    model_configs = {
        'seeds': np.random.randint(10000, size=4),
        'num_states': [2, 3, 4, 5],
        'transition_matrix_stickiness': [10],
    }
    # runCV('lrhmm', model_configs)
    # runCV('ghmm', model_configs)

    plotCV_different_models(num_states=5)   # plot lrhmm vs ghmm for specific <num_states>
    # plotCV_same_model(model_configs)    # plot lrhmm for all num_states


#
# # lrhmm_pkl, data_config, lrhmm_config = load_latest('lrhmm')
# lrhmm_pkl, data_config, lrhmm_config = load_specific_path('models/lrhmm/20241206_150838_few')
# chance_pkl, _, chance_config = load_latest('chance')
# lr_pkl, _, lr_config = load_latest('lr')
# # ghmm_pkl, _, ghmm_config = load_latest('ghmm')
# ghmm_pkl, _, ghmm_config = load_specific_path('models/ghmm/20241206_152151_spin')
#
# comparison_models = [lr_pkl, lrhmm_pkl, ghmm_pkl]
# model_labels = [m['prefix'].upper() for m in comparison_models]
#
# for m in comparison_models:
#     print(m['prefix'], m['train_data']['train_lp'], m['test_data']['test_lp'])
#
# # plot lp
# plt.figure()
# ticks = range(len(comparison_models))
# effective_fps = 150//data_config['predict_window_size']
# baseline = chance_pkl['test_data']['test_lp']
# plt.plot(ticks, [(m['train_data']['train_lp']-baseline)*effective_fps for m in comparison_models], 'o-', label='Train')
# plt.plot(ticks, [(m['test_data']['test_lp']-baseline)*effective_fps for m in comparison_models], 'o-', label='Test')
# plt.xticks(ticks, model_labels)
# plt.title('')
# plt.legend(loc='upper left')
# plt.ylabel('Normalized LL (bits/s)')
# plt.tight_layout()
# plt.show()
