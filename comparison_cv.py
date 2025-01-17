import glob
import joblib

import matplotlib.pyplot as plt
import numpy as np

from utilities import utils


def loadCV(model_prefix, num_states):
    model_pkls = glob.glob(f'models/cv/{model_prefix}_{num_states}_cv/**/')
    train_lps = []
    test_lps = []
    for _ in model_pkls:
        pkl, _, _ = utils.load_specific_path(_)
        train_lps.append(pkl['train_data']['train_lp'].item())
        test_lps.append(pkl['test_data']['test_lp'].item())
        print(train_lps[-1], test_lps[-1])
    return np.array(train_lps), np.array(test_lps)


def plotCV_same_model(model_prefix, num_states_configs):

    chance_pkl, _, _ = utils.load_specific_path(CHANCE_MODEL_PATH)

    baseline = chance_pkl['test_data']['test_lp']
    effective_fps = 150 // data_config['predict_window_size']

    plt.figure(figsize=(15, 10), constrained_layout=True)

    for i, s in enumerate(num_states_configs):
        hmm_train_lps, hmm_test_lps = loadCV(model_prefix, s)
        print(f"{model_prefix}: num_states={s} Train: {hmm_train_lps} Test:{hmm_test_lps}")
        x = [s + np.random.uniform(-0.1, 0.1) for _ in hmm_train_lps]
        plt.plot(x, (hmm_train_lps - baseline)*effective_fps, 'b.', label='Train' if i == 0 else '')
        plt.plot(x, (hmm_test_lps - baseline)*effective_fps, 'r.', label='Test' if i == 0 else '')

    plt.ylabel('Normalized LL (bits/s)')
    plt.xlabel('Number of states')
    plt.xticks(num_states_configs)
    plt.title(model_prefix.upper())
    plt.legend(loc='upper left')
    plt.margins(0.1)
    # plt.tight_layout()
    if savefig:
        plt.savefig(f'models/{model_prefix}_cv.pdf', bbox_inches='tight', dpi=300)
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

    data = joblib.load(f'../data/fly_data_cos=4_ortho_o=15.pkl')
    data_config, emissions, inputs = data['data_config'], data['emissions'], data['inputs']
    # session_keys = data_config['session_keys']
    # output_indices = data['output_indices']
    # num_batches = data_config['num_sessions']

    # num_train_batches = int(num_batches * 0.8)
    # train_emissions, train_inputs, train_session_keys = emissions[:num_train_batches], inputs[:num_train_batches], session_keys[:num_train_batches]
    # test_emissions, test_inputs, test_session_keys = emissions[num_train_batches:], inputs[num_train_batches:], session_keys[num_train_batches:]

    savefig = True
    display = False
    num_states_configs = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 18, 23, 28, 33, 15, 20, 25, 30, 40, 50]

    CHANCE_MODEL_PATH = 'models/chance_1/20250117_135807_octave'
    LR_MODEL_PATH = 'models/lr_1/20250117_135840_lane'
    # plotCV_same_model('lrhmm', num_states_configs)
    plotCV_same_model('ghmm', num_states_configs)
    # for ns in num_states_configs: plotCV_different_models(num_states=ns)
