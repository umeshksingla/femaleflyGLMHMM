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
        # print(train_lps[-1], test_lps[-1])
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
        # print(train_r2s[-1], test_r2s[-1])
    return np.array(train_r2s), np.array(test_r2s)


def loadCV_Corrs(path, model_prefix, num_states):
    model_pkl_paths = sorted(glob.glob(f'models/{path}/{model_prefix}_{num_states}_cv/**/'))
    train_r2s = []
    test_r2s = []
    for _ in model_pkl_paths:
        pkl, _, _ = utils.load_specific_path(_)
        if pkl is None:
            continue
        train_r2s.append(pkl['train_data']['train_correlation_by_o'][0])
        test_r2s.append(pkl['test_data']['test_correlation_by_o'][0])
        # print(train_r2s[-1], test_r2s[-1])
    return np.array(train_r2s), np.array(test_r2s)


def loadCV_R2adjs(path, model_prefix, num_states):
    model_pkl_paths = sorted(glob.glob(f'models/{path}/{model_prefix}_{num_states}_cv/**/'))
    train_r2adjs = []
    test_r2adjs = []
    for _ in model_pkl_paths:
        pkl, _, _ = utils.load_specific_path(_)
        if pkl is None:
            continue

        k = num_states*40

        tr_r2 = pkl['train_data']['train_score']
        tr_n = pkl['train_data']['train_emissions'].shape[0] * pkl['train_data']['train_emissions'].shape[1]
        tr_r2adj =  1 - (1-tr_r2) * ((tr_n-1)/(tr_n-k-1))
        print(num_states, k, tr_n, tr_r2, tr_r2adj)
        train_r2adjs.append(tr_r2adj)

        te_r2 = pkl['test_data']['test_score']
        te_n = pkl['test_data']['test_emissions'].shape[0] * pkl['test_data']['test_emissions'].shape[1]
        te_r2adj = 1 - (1-te_r2) * ((te_n-1)/(te_n-k-1))
        print(num_states, k, te_n, te_r2, te_r2adj)
        test_r2adjs.append(te_r2adj)

    return np.array(train_r2adjs), np.array(test_r2adjs)


def plotCV_same_model_LL(path, model_prefix, num_states_configs):

    chance_pkl, data_config_pkl, _ = utils.load_specific_path(CHANCE_MODEL_PATH)
    lr_pkl, _, _ = utils.load_specific_path(LR_MODEL_PATH)

    effective_fps = (data_config_pkl['input_raw_each_dim']//3) // data_config_pkl['predict_window_size']

    baseline = chance_pkl['test_data']['test_lp']

    plt.figure(figsize=(20, 12), constrained_layout=True)

    # Plot for num_states=1 i.e. linear regression
    plt.plot(1, (lr_pkl['train_data']['train_lp'] - baseline)*effective_fps, 'b.', label='Train')
    plt.plot(1, (lr_pkl['test_data']['test_lp'] - baseline)*effective_fps, 'r*', label='Test (corr to best train)', markersize=15)
    for i, s in enumerate(num_states_configs):
        hmm_train_lps, hmm_test_lps = loadCV_LLs(path, model_prefix, s)
        print(f"{model_prefix}: num_states={s} Train: {len(hmm_train_lps)} Test:{len(hmm_test_lps)}")
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

    plt.figure(figsize=(20, 12), constrained_layout=True)

    # # Plot for chance model
    # plt.plot(0, chance_pkl['train_data']['train_score']*100, 'b.', label='Train', markersize=15)
    # plt.plot(0, chance_pkl['test_data']['test_score']*100, 'r.', label='Test', markersize=15)

    # Plot for num_states=1 i.e. linear regression
    plt.plot(1, lr_pkl['train_data']['train_score']*100, 'b.', label='Train', markersize=15)
    plt.plot(1, lr_pkl['test_data']['test_score']*100, 'r.', label='Test', markersize=15)
    print("LR:", "train", lr_pkl['train_data']['train_score'], "test", lr_pkl['test_data']['test_score'])

    for i, s in enumerate(num_states_configs):
        hmm_train_r2s, hmm_test_r2s = loadCV_R2s(path, model_prefix, s)
        print("lrhmm_s: hmm_train_r2s", hmm_train_r2s, "hmm_test_r2s", hmm_test_r2s)
        print(f"{model_prefix}: num_states={s} Train: {len(hmm_train_r2s)} Test:{len(hmm_test_r2s)}")
        x = [s + np.random.uniform(-0.1, 0.1) for _ in hmm_train_r2s]
        plt.plot(x, hmm_train_r2s*100, 'b.', markersize=15)
        plt.plot(x, hmm_test_r2s*100, 'r.', markersize=15)

        if len(hmm_train_r2s):
            plt.plot(s, hmm_train_r2s[np.argmax(hmm_train_r2s)]*100, 'b*', markersize=15)
            plt.plot(s, hmm_test_r2s[np.argmax(hmm_train_r2s)]*100, 'r*', markersize=15)

    plt.ylabel('Var explained (%)')
    plt.xlabel('Number of states')
    plt.xticks([1] + num_states_configs)
    plt.title(model_prefix.upper())
    plt.legend(loc='upper left')
    plt.margins(0.1)
    plt.grid()
    if savefig:
        plt.savefig(f'models/{path}/{model_prefix}_r2_cv.pdf', bbox_inches='tight', dpi=300)
    if display:
        plt.show()
    return


def plotCV_same_model_Corr(path, model_prefix, num_states_configs):

    chance_pkl, _, _ = utils.load_specific_path(CHANCE_MODEL_PATH)
    lr_pkl, _, _ = utils.load_specific_path(LR_MODEL_PATH)

    plt.figure(figsize=(20, 12), constrained_layout=True)

    # Plot for num_states=1 i.e. linear regression
    plt.plot(1, lr_pkl['train_data']['train_correlation_by_o'][0], 'b.', label='Train', markersize=15)
    plt.plot(1, lr_pkl['test_data']['test_correlation_by_o'][0], 'r.', label='Test', markersize=15)
    print("LR:", "train", lr_pkl['train_data']['train_correlation_by_o'], "test", lr_pkl['test_data']['test_correlation_by_o'])

    for i, s in enumerate(num_states_configs):
        hmm_train_r2s, hmm_test_r2s = loadCV_Corrs(path, model_prefix, s)
        print("lrhmm_s: hmm_train_r2s", hmm_train_r2s, "hmm_test_r2s", hmm_test_r2s)
        print(f"{model_prefix}: num_states={s} Train: {len(hmm_train_r2s)} Test:{len(hmm_test_r2s)}")
        x = [s + np.random.uniform(-0.1, 0.1) for _ in hmm_train_r2s]
        plt.plot(x, hmm_train_r2s, 'b.', markersize=15)
        plt.plot(x, hmm_test_r2s, 'r.', markersize=15)

        if len(hmm_train_r2s):
            plt.plot(s, hmm_train_r2s[np.argmax(hmm_train_r2s)], 'b*', markersize=15)
            plt.plot(s, hmm_test_r2s[np.argmax(hmm_train_r2s)], 'r*', markersize=15)

    plt.ylabel('Correlation (lag=0) score for emissions[0]')
    plt.xlabel('Number of states')
    plt.xticks([1] + num_states_configs)
    plt.title(model_prefix.upper())
    plt.legend(loc='upper left')
    plt.margins(0.1)
    plt.grid()
    if savefig:
        plt.savefig(f'models/{path}/{model_prefix}_corr0_cv.pdf', bbox_inches='tight', dpi=300)
    if display:
        plt.show()
    return


def plotCV_same_model_R2adj(path, model_prefix, num_states_configs):

    lr_pkl, _, _ = utils.load_specific_path(LR_MODEL_PATH)

    plt.figure(figsize=(20, 12), constrained_layout=True)

    # Plot for num_states=1 i.e. linear regression
    lr_tr_r2 = lr_pkl['train_data']['train_score']
    lr_tr_n = lr_pkl['train_data']['train_emissions'].shape[0] * lr_pkl['train_data']['train_emissions'].shape[1]
    lr_te_r2 = lr_pkl['test_data']['test_score']
    lr_te_n = lr_pkl['test_data']['test_emissions'].shape[0] * lr_pkl['test_data']['test_emissions'].shape[1]
    k = 40
    plt.plot(1, 1 - (1-lr_tr_r2) * ((lr_tr_n-1)/(lr_tr_n-k-1)), 'b.', label='Train', markersize=15)
    plt.plot(1, 1 - (1-lr_te_r2) * ((lr_te_n-1)/(lr_te_n-k-1)), 'r.', label='Test', markersize=15)

    for i, s in enumerate(num_states_configs):
        hmm_train_r2s, hmm_test_r2s = loadCV_R2adjs(path, model_prefix, s)
        print(f"{model_prefix}: num_states={s} Train: {len(hmm_train_r2s)} Test:{len(hmm_test_r2s)}")
        x = [s + np.random.uniform(-0.1, 0.1) for _ in hmm_train_r2s]
        plt.plot(x, hmm_train_r2s*100, 'b.', markersize=15)
        plt.plot(x, hmm_test_r2s*100, 'r.', markersize=15)

        # if len(hmm_train_r2s):
        #     plt.plot(s, hmm_train_r2s[np.argmax(hmm_train_r2s)]*100, 'b*', markersize=15)
        #     plt.plot(s, hmm_test_r2s[np.argmax(hmm_train_r2s)]*100, 'r*', markersize=15)

    plt.ylabel('Adjusted Var explained (%)')
    plt.xlabel('Number of states')
    plt.xticks([1] + num_states_configs)
    plt.title(model_prefix.upper())
    plt.legend(loc='upper left')
    plt.margins(0.1)
    plt.grid()
    if savefig:
        plt.savefig(f'models/{path}/{model_prefix}_r2adj_cv.pdf', bbox_inches='tight', dpi=300)
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

    lr_pkl, data_config_pkl, _ = utils.load_specific_path(LR_MODEL_PATH)
    chance_pkl, _, _ = utils.load_specific_path(CHANCE_MODEL_PATH)

    baseline = chance_pkl['test_data']['test_lp']
    effective_fps = (data_config_pkl['input_raw_each_dim']//3) // data_config_pkl['predict_window_size']

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

    savefig = True
    display = False
    num_states_configs = [
        # 2, 5, 15, 20, 25, 27, 30
        2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18, 20, 23, 25, 27, 30,
        #   16, 18, 23, 28, 33, 15, 20, 25, 30, 40, 50
        ]

    # CHANCE_MODEL_PATH = 'models/cv6_fred/chance_1_cv/20250310_195917_dysfunction'
    # LR_MODEL_PATH = 'models/cv6_fred/lr_1_cv/20250310_195654_rediscovery'
    # path = 'cv6_fred'

    CHANCE_MODEL_PATH = 'models/chance_1/20250117_135807_octave'
    LR_MODEL_PATH = 'models/lr_1/20250117_135840_lane'
    path = 'cv6'

    # generate_figures_same_model(path, 'lrhmmci', num_states_configs[:2])
    # plotCV_same_model_LL(path, 'lrhmmci', num_states_configs)
    # plotCV_same_model_R2(path, 'lrhmmci', num_states_configs)
    plotCV_same_model_Corr(path, 'lrhmmci', num_states_configs)
    # plotCV_same_model_R2adj(path, 'lrhmmci', num_states_configs)

    # for ns in num_states_configs: plotCV_different_models(path, num_states=ns)
