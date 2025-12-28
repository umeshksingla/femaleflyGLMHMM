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
    return np.array(train_lps), np.array(test_lps)


def loadCV_LLs_by_fly(path, model_prefix, num_states):
    model_pkl_paths = sorted(glob.glob(f'models/{path}/{model_prefix}_{num_states}_cv/**/'))
    for _ in model_pkl_paths[:1]:   # any one split is good enough to plot for LLs by fly
        pkl, _, _ = utils.load_specific_path(_)
        if pkl is None:
            continue
        return pkl['train_data']['train_lps_by_fly'], pkl['test_data']['test_lps_by_fly']


def loadCV_R2s(path, model_prefix, num_states):
    model_pkl_paths = sorted(glob.glob(f'models/{path}/{model_prefix}_{num_states}_cv/**/'))
    train_r2s = []
    test_r2s = []
    for _ in model_pkl_paths:
        pkl, _, _ = utils.load_specific_path(_)
        if pkl is None:
            continue
        # print(pkl['train_data'].keys())
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


def plotCV_same_model_LL(path, model_prefix, num_states_configs, plot_all_test=False, filesuffix=''):

    _, data_config_pkl, _ = utils.load_specific_path(CHANCE_MODEL_PATH)
    # lr_pkl, data_config_pkl, _ = utils.load_specific_path(LR_MODEL_PATH)

    factor_bits_per_sec = data_config_pkl['effective_fps']/np.log(2)

    # baseline = chance_pkl['test_data']['test_lp']

    plt.figure(figsize=(10, 6), constrained_layout=True)
    ax=plt.gca()
    ms = 5

    # Plot for num_states=0 i.e. chance
    chance_train_lps, chance_test_lps = loadCV_LLs(path, 'chance', 1)
    train_jitter = np.random.uniform(-0.25, 0.25, size=len(chance_train_lps))
    plt.plot(train_jitter, [0.]*len(chance_train_lps), 'ko', mfc='none', markersize=ms)  # plot 0s for chance as chance_lps stored are not relative to chance.

    # Plot for num_states=1 i.e. linear regression
    lr_train_lps, lr_test_lps = loadCV_LLs(path, 'lr', 1)
    train_jitter = np.random.uniform(-0.25, 0.25, size=len(lr_train_lps))
    plt.plot(1+train_jitter, lr_train_lps*factor_bits_per_sec, 'ko', mfc='none', markersize=ms, label='Train')
    if plot_all_test:
        test_jitter = np.random.uniform(-0.25, 0.25, size=len(lr_test_lps))
        plt.plot(1+test_jitter, lr_test_lps*factor_bits_per_sec, 'ko', markersize=ms, label='Held-out')
    else:
        plt.plot(1, (lr_test_lps[np.argmax(lr_train_lps)])*factor_bits_per_sec, 'ko', markersize=ms, label='Held-out')   # plot for the max train one

    # Plot for num_states > 1 now
    for i, s in enumerate(num_states_configs):
        hmm_train_lps, hmm_test_lps = loadCV_LLs(path, model_prefix, s)
        print(f"{model_prefix}: num_states={s} Train: {len(hmm_train_lps)} Test:{len(hmm_test_lps)}")
        train_jitter = np.random.uniform(-0.25, 0.25, size=len(hmm_train_lps))
        plt.plot(s+train_jitter, hmm_train_lps*factor_bits_per_sec, 'ko', mfc='none', markersize=ms)
        # plt.errorbar(s + 0.4, np.mean(hmm_train_lps*factor_bits_per_sec), yerr=np.std(hmm_train_lps*factor_bits_per_sec), color='k', fmt='o', capsize=0)


        if plot_all_test:
            test_jitter = np.random.uniform(-0.25, 0.25, size=len(hmm_test_lps))
            plt.plot(s+test_jitter, hmm_test_lps*factor_bits_per_sec, 'ko', markersize=ms)
        else:
            if len(hmm_train_lps):
                plt.plot(s, (hmm_test_lps[np.argmax(hmm_train_lps)])*factor_bits_per_sec, 'ko', markersize=ms)   # plot for the max train one

    plt.ylabel('Normalized LL (bits/s)')
    plt.xlabel('Number of states')
    plt.xticks([0, 1] + num_states_configs, labels=['Chance', 'GLM'] + num_states_configs)

    # Rotate only the first 2 tick labels
    for i, label in enumerate(ax.get_xticklabels()):
        if i < 2:
            label.set_rotation(90)
    plt.title(model_prefix.upper())
    plt.legend(loc='lower right')
    plt.margins(0.1)
    plt.grid(alpha=0.15)
    # plt.tight_layout()
    if savefig:
        plt.savefig(f'models/{path}/{model_prefix}_{path}_ll_cv{filesuffix}.pdf', bbox_inches='tight', dpi=300)
    if display:
        plt.show()
    return


def plotCV_same_model_LL_by_fly(path, model_prefix, num_states_configs, filesuffix=''):

    _, data_config_pkl, _ = utils.load_specific_path(CHANCE_MODEL_PATH)
    # lr_pkl, data_config_pkl, _ = utils.load_specific_path(LR_MODEL_PATH)

    factor_bits_per_sec = data_config_pkl['effective_fps']/np.log(2)

    # baseline = chance_pkl['test_data']['test_lp']

    plt.figure(figsize=(10, 6), constrained_layout=True)
    ax=plt.gca()

    # Plot for num_states=0 i.e. chance
    # plt.plot(0, 0., 'ko', mfc='none', markersize=7)
    chance_train_lps, chance_test_lps = loadCV_LLs_by_fly(path, 'chance', 1)    
    # train_jitter = np.random.uniform(-0.2, 0.2, size=len(chance_train_lps))
    test_jitter = np.random.uniform(-0.2, 0.2, size=len(chance_test_lps))
    # plt.plot(0+train_jitter, [0.]*len(chance_train_lps), 'ko', mfc='none', markersize=3, label='Train')
    plt.plot(0+test_jitter, [0.]*len(chance_test_lps), 'ko', markersize=3, label='Held-out')    # chance test lps are not calc relative to chance, so plot 0s here.

    # Plot for num_states=1 i.e. linear regression
    lr_train_lps, lr_test_lps = loadCV_LLs_by_fly(path, 'lr', 1)
    print(f"LR: Train: {len(lr_train_lps)} Test:{len(lr_test_lps)}")
    # train_jitter = np.random.uniform(-0.2, 0.2, size=len(lr_train_lps))
    test_jitter = np.random.uniform(-0.2, 0.2, size=len(lr_test_lps))
    # plt.plot(1+train_jitter, lr_train_lps*factor_bits_per_sec, 'ko', mfc='none', markersize=3, label='Train')
    plt.plot(1+test_jitter, lr_test_lps*factor_bits_per_sec, 'ko', markersize=3)
    plt.errorbar(1 + 0.4, np.mean(lr_test_lps*factor_bits_per_sec), yerr=np.std(lr_test_lps*factor_bits_per_sec), color='k', fmt='o', capsize=0)

    # Plot for num_states > 1 now
    for i, s in enumerate(num_states_configs):
        hmm_train_lps, hmm_test_lps = loadCV_LLs_by_fly(path, model_prefix, s)
        # train_jitter = np.random.uniform(-0.2, 0.2, size=len(hmm_train_lps))
        test_jitter = np.random.uniform(-0.2, 0.2, size=len(hmm_test_lps))
        print(f"{model_prefix}: num_states={s} Train: {len(hmm_train_lps)} Test:{len(hmm_test_lps)}")
        # plt.plot(s+train_jitter, (hmm_train_lps)*factor_bits_per_sec, 'ko', mfc='none', markersize=3)
        plt.plot(s+test_jitter, hmm_test_lps*factor_bits_per_sec, 'ko', markersize=3)
        plt.errorbar(s + 0.4, np.mean(hmm_test_lps*factor_bits_per_sec), yerr=np.std(hmm_test_lps*factor_bits_per_sec), color='k', fmt='o', capsize=0)

    plt.ylabel('Normalized LL (bits/s)')
    plt.xlabel('Number of states')
    plt.xticks([0, 1] + num_states_configs, labels=['Chance', 'GLM'] + num_states_configs)
    # Rotate only the first 2 tick labels
    for i, label in enumerate(ax.get_xticklabels()):
        if i < 2:
            label.set_rotation(90)
    plt.title(model_prefix.upper())
    plt.legend(loc='lower right')
    plt.margins(0.1)
    plt.grid(alpha=0.15)
    # plt.tight_layout()
    if savefig:
        plt.savefig(f'models/{path}/{model_prefix}_{path}_ll_cv_by_fly{filesuffix}.pdf', bbox_inches='tight', dpi=300)
    if display:
        plt.show()
    return


def plotCV_same_model_R2(path, model_prefix, num_states_configs, plot_all_test=False, filesuffix=''):

    plt.figure(figsize=(10, 6), constrained_layout=True)
    ax=plt.gca()
    ms = 5

    # skip chance model for r2

    # Plot for num_states=1 i.e. linear regression
    lr_train_lps, lr_test_lps = loadCV_R2s(path, 'lr', 1)
    train_jitter = np.random.uniform(-0.25, 0.25, size=len(lr_train_lps))
    plt.plot(1+train_jitter, lr_train_lps*100, 'ko', mfc='none', markersize=ms, label='Train')
    if plot_all_test:
        test_jitter = np.random.uniform(-0.25, 0.25, size=len(lr_test_lps))
        plt.plot(1+test_jitter, lr_test_lps*100, 'ko', markersize=ms, label='Held-out')
    else:
        plt.plot(1, (lr_test_lps[np.argmax(lr_train_lps)])*100, 'ko', markersize=ms, label='Held-out')   # plot for the max train one

    # Plot for num_states > 1 now
    for i, s in enumerate(num_states_configs):
        hmm_train_lps, hmm_test_lps = loadCV_R2s(path, model_prefix, s)
        print(f"{model_prefix}: num_states={s} Train: {len(hmm_train_lps)} Test:{len(hmm_test_lps)}")
        train_jitter = np.random.uniform(-0.25, 0.25, size=len(hmm_train_lps))
        plt.plot(s+train_jitter, hmm_train_lps*100, 'ko', mfc='none', markersize=ms)
        # plt.errorbar(s + 0.4, np.mean(hmm_train_lps*factor_bits_per_sec), yerr=np.std(hmm_train_lps*factor_bits_per_sec), color='k', fmt='o', capsize=0)

        if plot_all_test:
            test_jitter = np.random.uniform(-0.25, 0.25, size=len(hmm_test_lps))
            plt.plot(s+test_jitter, hmm_test_lps*100, 'ko', markersize=ms)
        else:
            if len(hmm_train_lps):
                plt.plot(s, (hmm_test_lps[np.argmax(hmm_train_lps)])*100, 'ko', markersize=ms)   # plot for the max train one

    plt.ylabel('Var Explained (%)')
    plt.xlabel('Number of states')
    plt.xticks([1] + num_states_configs, labels=['GLM'] + num_states_configs)

    # Rotate only the first 2 tick labels
    for i, label in enumerate(ax.get_xticklabels()):
        if i < 2:
            label.set_rotation(90)
    plt.title(model_prefix.upper())
    plt.legend(loc='lower right')
    plt.margins(0.1)
    plt.grid(alpha=0.15)
    # plt.tight_layout()
    if savefig:
        plt.savefig(f'models/{path}/{model_prefix}_{path}_r2_cv{filesuffix}.pdf', bbox_inches='tight', dpi=300)
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

    path = 'june25_kfoldcv_wt'
    CHANCE_MODEL_PATH = f'models/{path}/chance_1_cv/20250625_001932_chord'

    # path = 'june25_kfoldcv_wt_fred'
    # CHANCE_MODEL_PATH = f'models/{path}/chance_1_cv/20250625_012323_coil'
    plot_all_test = True

    num_states_configs = [ 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20]
    plotCV_same_model_LL(path, 'glm-hmm', num_states_configs, plot_all_test=plot_all_test, filesuffix='')
    # plotCV_same_model_LL_by_fly(path, 'glm-hmm', num_states_configs, filesuffix='')
    plotCV_same_model_R2(path, 'glm-hmm', num_states_configs, plot_all_test=plot_all_test, filesuffix='')

    num_states_configs = num_states_configs + [ 25, 30 ]
    plotCV_same_model_LL(path, 'glm-hmm', num_states_configs, plot_all_test=plot_all_test, filesuffix='_extended')
    # plotCV_same_model_LL_by_fly(path, 'glm-hmm', num_states_configs, filesuffix='_extended')
    plotCV_same_model_R2(path, 'glm-hmm', num_states_configs, plot_all_test=plot_all_test, filesuffix='_extended')

    # plotCV_same_model_Corr(path, 'lrhmmci', num_states_configs)   # maybe plot other metrics too
    # plotCV_same_model_R2adj(path, 'lrhmmci', num_states_configs)
