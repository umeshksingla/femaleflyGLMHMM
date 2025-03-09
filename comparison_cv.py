####################################

# Usage: python comparison_cv.py

####################################

import glob
import joblib

import matplotlib.pyplot as plt
import numpy as np

from utilities import utils


def loadCV_LLs(path, model_prefix, num_states):
    model_pkl_paths = sorted(glob.glob(f'models/{path}/{model_prefix}_{num_states}_cv/**/'))
    train_lps = []
    test_lps = []
    for _ in model_pkl_paths:
        pkl, _, _ = utils.load_specific_path(_)
        if pkl is None:
            continue
        train_lps.append(pkl['train_data']['train_lp'].item())
        test_lps.append(pkl['test_data']['test_lp'].item())
        print(train_lps[-1], test_lps[-1])
    return np.array(train_lps), np.array(test_lps)


def loadCV_R2s(path, model_prefix, num_states):
    model_pkl_paths = sorted(glob.glob(f'models/{path}/{model_prefix}_{num_states}_cv/**/'))
    train_r2s = []
    test_r2s = []
    for _ in model_pkl_paths:
        pkl, _, _ = utils.load_specific_path(_)
        if pkl is None:
            continue
        train_r2s.append(pkl['train_data']['train_score'])
        test_r2s.append(pkl['test_data']['test_score'])
        print(train_r2s[-1], test_r2s[-1])
    return np.array(train_r2s), np.array(test_r2s)


def plotCV_same_model_LL(path, model_prefix, num_states_configs):

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
        hmm_train_lps, hmm_test_lps = loadCV_LLs(path, model_prefix, s)
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
        plt.savefig(f'models/{path}/{model_prefix}_ll_cv.pdf', bbox_inches='tight', dpi=300)
    if display:
        plt.show()
    return


def plotCV_same_model_R2(path, model_prefix, num_states_configs):

    chance_pkl, _, _ = utils.load_specific_path(CHANCE_MODEL_PATH)
    lr_pkl, _, _ = utils.load_specific_path(LR_MODEL_PATH)

    plt.figure(figsize=(20, 10), constrained_layout=True)

    # Plot for chance model
    plt.plot(0, chance_pkl['train_data']['train_score']*100, 'b.', label='Train', markersize=15)
    plt.plot(0, chance_pkl['test_data']['test_score'] * 100, 'r.', label='Test', markersize=15)

    # Plot for num_states=1 i.e. linear regression
    plt.plot(1, lr_pkl['train_data']['train_score']*100, 'b.', markersize=15)
    plt.plot(1, lr_pkl['test_data']['test_score']*100, 'r.', markersize=15)

    for i, s in enumerate(num_states_configs):
        hmm_train_lps, hmm_test_lps = loadCV_R2s(path, model_prefix, s)
        print(f"{model_prefix}: num_states={s} Train: {hmm_train_lps} Test:{hmm_test_lps}")
        x = [s + np.random.uniform(-0.1, 0.1) for _ in hmm_train_lps]
        plt.plot(x, hmm_train_lps*100, 'b.', markersize=15)
        plt.plot(x, hmm_test_lps*100, 'r.', markersize=15)

        # if len(hmm_train_lps):
        #     plt.plot(s, (hmm_train_lps[np.argmax(hmm_train_lps)] - baseline)*effective_fps, 'b*', markersize=15)
        #     plt.plot(s, (hmm_test_lps[np.argmax(hmm_train_lps)] - baseline)*effective_fps, 'r*', markersize=15)

    plt.ylabel('Var explained (%)')
    plt.xlabel('Number of states')
    plt.xticks([0, 1] + num_states_configs)
    plt.title(model_prefix.upper())
    plt.legend(loc='upper left')
    plt.margins(0.1)
    plt.grid()
    if savefig:
        plt.savefig(f'models/{path}/{model_prefix}_r2_cv.pdf', bbox_inches='tight', dpi=300)
    if display:
        plt.show()
    return


def generate_figures_same_model(path, model_prefix, num_states_configs):

    for s in num_states_configs:
        model_pkl_paths = sorted(glob.glob(f'models/{path}/{model_prefix}_{s}_cv/**/'))
        for model_pkl_path in model_pkl_paths:
            utils.generate_figures(model_pkl_path, savefig=True, display=False)
    return


def plotCV_different_models(path, num_states):

    lrhmm_train_lps, lrhmm_test_lps = loadCV_LLs(path, 'lrhmm', num_states)
    ghmm_train_lps, ghmm_test_lps = loadCV_LLs(path, 'ghmm', num_states)

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

    data_config = joblib.load(f'data/wt_fred_fly_data_cos=4_ortho_o=2.pkl')['data_config']

    savefig = True
    display = False
    num_states_configs = [
        # 2, 5, 15, 20, 25, 27, 30
        2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18, 20, 23, 25, 27, 30,
        #   16, 18, 23, 28, 33, 15, 20, 25, 30, 40, 50
        ]

    CHANCE_MODEL_PATH = 'models/general_fred/chance_1_cv/20250307_175312_monopoly'
    LR_MODEL_PATH = 'models/general_fred/lr_1_cv/20250307_175346_pancreas'

    path = 'general_fred'
    # plotCV_same_model_LL(path, 'lrhmmci', num_states_configs)
    generate_figures_same_model(path, 'lrhmmci', num_states_configs[:2])
    # plotCV_same_model_R2(path, 'lrhmmci', num_states_configs[:2])
    # plotCV_same_model_LL(path, 'lrhmmci', num_states_configs[:2])

    # for ns in num_states_configs: plotCV_different_models(path, num_states=ns)
