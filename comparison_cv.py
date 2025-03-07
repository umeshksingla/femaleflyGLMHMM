####################################

# Usage: python comparison_cv.py

####################################

import glob
import joblib

import matplotlib.pyplot as plt
import numpy as np

from utilities import utils


def loadCV(path, model_prefix, num_states):
    model_pkls = sorted(glob.glob(f'models/{path}/{model_prefix}_{num_states}_cv/**/'))
    train_lps = []
    test_lps = []
    for _ in model_pkls:
        pkl, _, _ = utils.load_specific_path(_)
        if pkl is None:
            continue
        train_lps.append(pkl['train_data']['train_lp'].item())
        test_lps.append(pkl['test_data']['test_lp'].item())
        print(train_lps[-1], test_lps[-1])
    return np.array(train_lps), np.array(test_lps)


def plotCV_same_model(path, model_prefix, num_states_configs):

    effective_fps = 150 // data_config['predict_window_size']

    chance_pkl, _, _ = utils.load_specific_path(CHANCE_MODEL_PATH)
    lr_pkl, _, _ = utils.load_specific_path(LR_MODEL_PATH)

    baseline = chance_pkl['test_data']['test_lp']
    # lr_baseline = lr_pkl['test_data']['test_lp']

    plt.figure(figsize=(20, 10), constrained_layout=True)

    # Plot for num_states=1 i.e. linear regression
    plt.plot(1, (lr_pkl['train_data']['train_lp'] - baseline)*effective_fps, 'b.', label='Train')
    plt.plot(1, (lr_pkl['test_data']['test_lp'] - baseline)*effective_fps, 'r*', label='Test (corr to best train)', markersize=15)
    for i, s in enumerate(num_states_configs):
        hmm_train_lps, hmm_test_lps = loadCV(path, model_prefix, s)
        print(f"{model_prefix}: num_states={s} Train: {hmm_train_lps} Test:{hmm_test_lps}")
        x = [s + np.random.uniform(-0.2, 0.2) for _ in hmm_train_lps]
        plt.plot(x, (hmm_train_lps - baseline)*effective_fps, 'b.')
        # plt.plot(x, (hmm_test_lps - baseline)*effective_fps, 'r.', label='Test' if i == 0 else '')

        if len(hmm_train_lps):
            # plt.plot(s, (hmm_train_lps[np.argmax(hmm_train_lps)] - baseline)*effective_fps, 'b*', markersize=15)
            plt.plot(s, (hmm_test_lps[np.argmax(hmm_train_lps)] - baseline)*effective_fps, 'r*', markersize=15)

    plt.ylabel('Normalized LL (bits/s)')
    plt.xlabel('Number of states')
    plt.xticks([1] + num_states_configs)
    plt.title(model_prefix.upper())
    plt.legend(loc='upper left')
    plt.margins(0.1)
    plt.grid()
    # plt.tight_layout()
    if savefig:
        plt.savefig(f'models/{path}/{model_prefix}_cv.pdf', bbox_inches='tight', dpi=300)
    if display:
        plt.show()
    return


def plotCV_different_models(num_states):

    lrhmm_train_lps, lrhmm_test_lps = loadCV('lrhmm', num_states)
    ghmm_train_lps, ghmm_test_lps = loadCV('ghmm', num_states)

    print(f"LR-HMM num_states={num_states} Train: {lrhmm_train_lps} Test:{lrhmm_test_lps}")
    print(f"G-HMM num_states={num_states} Train: {ghmm_train_lps} Test:{ghmm_test_lps}")

    lr_pkl, _, _ = utils.load_specific_path(LR_MODEL_PATH)
    chance_pkl, _, _ = utils.load_specific_path(CHANCE_MODEL_PATH)

    baseline = chance_pkl['test_data']['test_lp']
    effective_fps = 150 // data_config['predict_window_size']

    plt.figure(constrained_layout=True)

    # lr
    plt.plot(0, (lr_pkl['train_data']['train_lp'] - baseline)*effective_fps, 'b.', label='Train')
    plt.plot(0, (lr_pkl['test_data']['test_lp'] - baseline)*effective_fps, 'r.', label='Train')

    # lrhmm
    x = [1 + np.random.uniform(-0.1, 0.1) for _ in lrhmm_train_lps]
    plt.plot(x, (lrhmm_train_lps - baseline)*effective_fps, 'b.')
    plt.plot(x, (lrhmm_test_lps - baseline)*effective_fps, 'r.')

    # ghmm
    x = [2 + np.random.uniform(-0.1, 0.1) for _ in ghmm_train_lps]
    plt.plot(x, (ghmm_train_lps - baseline)*effective_fps, 'b.')
    plt.plot(x, (ghmm_test_lps - baseline)*effective_fps, 'r.')

    plt.xticks([0, 1, 2], ['LR', 'LR-HMM', 'G-HMM'])
    plt.ylabel('Normalized LL (bits/s)')
    plt.title(f'Multiple seeds (num_states={num_states})')
    plt.legend()
    plt.margins(0.1)
    if savefig:
        plt.savefig(f'models/cv_{num_states}.pdf', bbox_inches='tight', dpi=300)
    if display:
        plt.show()
    return


if __name__ == '__main__':

    data_config = joblib.load(f'../data/fly_data_cos=4_ortho_o=15.pkl')['data_config']
    # emissions, inputs = data['emissions'], data['inputs']
    # session_keys = data_config['session_keys']
    # output_indices = data['output_indices']
    # num_batches = data_config['num_sessions']

    # num_train_batches = int(num_batches * 0.8)
    # train_emissions, train_inputs, train_session_keys = emissions[:num_train_batches], inputs[:num_train_batches], session_keys[:num_train_batches]
    # test_emissions, test_inputs, test_session_keys = emissions[num_train_batches:], inputs[num_train_batches:], session_keys[num_train_batches:]

    savefig = True
    display = False
    num_states_configs = [
        # 2, 5, 15, 20, 25, 27, 30
        2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18, 20, 23, 25, 27, 30,
        #   16, 18, 23, 28, 33, 15, 20, 25, 30, 40, 50
        ]

    CHANCE_MODEL_PATH = 'models/chance_1/20250117_135807_octave'
    LR_MODEL_PATH = 'models/lr_1/20250117_135840_lane'

    path = 'cv6'
    plotCV_same_model(path, 'lrhmmci', num_states_configs)
    # plotCV_same_model('ghmm', num_states_configs)
    # for ns in num_states_configs: plotCV_different_models(num_states=ns)
