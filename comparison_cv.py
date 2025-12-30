####################################

# Usage: python comparison_cv.py

####################################

import glob
import joblib

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from utilities import utils

# -- Fonts --
mpl.rcParams['font.size'] = 18  # Panel label
# mpl.rcParams['font.family'] = 'Arial'
# mpl.rcParams['font.sans-serif'] = 'Arial'
mpl.rcParams['text.color'] = 'black'
mpl.rcParams['axes.labelcolor'] = 'black'

# -- Axes --
mpl.rcParams['axes.spines.bottom'] = True
mpl.rcParams['axes.spines.left'] = True
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.grid'] = False
mpl.rcParams['axes.grid.axis'] = 'y'
mpl.rcParams['grid.color'] = 'black'
mpl.rcParams['grid.linewidth'] = 0.5
mpl.rcParams['axes.axisbelow'] = True
mpl.rcParams['axes.linewidth'] = 0.5
mpl.rcParams['axes.ymargin'] = 0
mpl.rcParams["axes.labelsize"] = 20
mpl.rcParams["xtick.labelsize"] = 20
mpl.rcParams["ytick.labelsize"] = 20
mpl.rcParams["legend.fontsize"] = 20
plt.rcParams['axes.titlesize'] = 20

# -- Ticks and tick labels --
mpl.rcParams['axes.edgecolor'] = 'black'
mpl.rcParams['xtick.bottom'] = True
mpl.rcParams['ytick.left'] = True
mpl.rcParams['xtick.color'] = 'black'
mpl.rcParams['ytick.color'] = 'black'
mpl.rcParams['xtick.major.width'] = 1
mpl.rcParams['ytick.major.width'] = 1
mpl.rcParams['xtick.major.size'] = 4
mpl.rcParams['ytick.major.size'] = 4
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'

# -- Figure size --
# plt.rcParams['figure.figsize'] = (6, 4)
# plt.rcParams['figure.dpi'] = 300
# mpl.rcParams['legend.frameon'] = False

# -- Saving Options --
# plt.rcParams['savefig.bbox'] = 'tight'
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
# rcParams['savefig.transparent'] = True

# -- Plot Styles --
mpl.rcParams['lines.linewidth'] = 1
#################


# def loadCV_LLs(path, model_prefix, num_states):
#     model_pkl_paths = sorted(glob.glob(f'models/{path}/{model_prefix}_{num_states}_cv/**/'))
#     train_lps = []
#     test_lps = []
#     for _ in model_pkl_paths:
#         pkl, _, _ = utils.load_specific_path(_)
#         if pkl is None:
#             continue
#         train_lps.append(pkl['train_data']['train_lp'].item())
#         test_lps.append(pkl['test_data']['test_lp'].item())
#     return np.array(train_lps), np.array(test_lps)


def loadCV_Scores(path, model_prefix, num_states, score_type):
    """
    :param score_type: 'r2' or 'pearson' or 'll'
    """
    model_pkl_paths = sorted(glob.glob(f'models/{path}/{model_prefix}_{num_states}_cv/**/'))
    train_scores = []
    test_scores = []
    for _ in model_pkl_paths:
        pkl, data_config_pkl, _ = utils.load_specific_path(_)
        if pkl is None:
            continue
        if score_type == 'r2':
            train_score = pkl['train_data']['train_score'] * 100
            test_score = pkl['test_data']['test_score'] * 100
        elif score_type == 'pearson':
            train_score = pkl['train_data']['train_pearson']
            test_score = pkl['test_data']['test_pearson']
        elif score_type == 'll':
            factor_bits_per_sec = data_config_pkl['effective_fps']/np.log(2)
            train_score = pkl['train_data']['train_lp'].item() * factor_bits_per_sec
            test_score = pkl['test_data']['test_lp'].item() * factor_bits_per_sec
        else:
            raise Exception(f'Unsupported score type "{score_type}".')
        train_scores.append(train_score)
        test_scores.append(test_score)
    return np.array(train_scores), np.array(test_scores)


def plotCV_same_model_LL(path, model_prefix, num_states_configs, plot_all_test=False, filesuffix=''):

    plt.figure(figsize=(10, 6), constrained_layout=True)
    ax=plt.gca()
    ms = 5

    # Plot for num_states=0 i.e. chance
    # chance_train_lps, chance_test_lps = loadCV_LLs(path, 'chance', 1)
    # train_jitter = np.random.uniform(-0.25, 0.25, size=len(chance_train_lps))
    # plt.plot([0], [0.], 'ko', mfc='none', markersize=ms)  # plot 0s for chance as chance_lps stored are not relative to chance.

    # Plot for num_states=0 i.e. chance
    s = 0
    chance_train_lps, chance_test_lps = loadCV_Scores(path, 'chance', 0, score_type='ll')
    train_jitter = np.random.uniform(-0.25, 0.25, size=len(chance_train_lps))
    plt.plot(s+train_jitter, np.zeros_like(chance_train_lps), 'ko', mfc='none', markersize=ms)  # plot 0s for chance as chance_lps stored are not relative to chance.
    if plot_all_test:
        test_jitter = np.random.uniform(-0.25, 0.25, size=len(chance_test_lps))
        plt.plot(s+test_jitter, np.zeros_like(chance_test_lps), 'ko', markersize=ms)
    else:
        plt.plot(s, (0.), 'ko', markersize=ms)   # plot for the max train one

    # Plot for num_states=1 i.e. linear regression
    s = 1
    lr_train_lps, lr_test_lps = loadCV_Scores(path, 'lr', 1, score_type='ll')
    train_jitter = np.random.uniform(-0.25, 0.25, size=len(lr_train_lps))
    plt.plot(s+train_jitter, lr_train_lps, 'ko', mfc='none', markersize=ms, label='Train')
    if plot_all_test:
        test_jitter = np.random.uniform(-0.25, 0.25, size=len(lr_test_lps))
        plt.plot(s+test_jitter, lr_test_lps, 'ko', markersize=ms, label='Held-out')
    else:
        plt.plot(s, (lr_test_lps[np.argmax(lr_train_lps)]), 'ko', markersize=ms, label='Held-out')   # plot for the max train one

    # Plot for num_states > 1 now
    for i, s in enumerate(num_states_configs):
        hmm_train_lps, hmm_test_lps = loadCV_Scores(path, model_prefix, s, score_type='ll')
        print(f"{model_prefix}: num_states={s} Train: {len(hmm_train_lps)} Test:{len(hmm_test_lps)}")
        train_jitter = np.random.uniform(-0.25, 0.25, size=len(hmm_train_lps))
        plt.plot(s+train_jitter, hmm_train_lps, 'ko', mfc='none', markersize=ms)
        # plt.errorbar(s + 0.4, np.mean(hmm_train_lps), yerr=np.std(hmm_train_lps), color='k', fmt='o', capsize=0)

        if plot_all_test:
            test_jitter = np.random.uniform(-0.25, 0.25, size=len(hmm_test_lps))
            plt.plot(s+test_jitter, hmm_test_lps, 'ko', markersize=ms)
        else:
            if len(hmm_train_lps):
                plt.plot(s, (hmm_test_lps[np.argmax(hmm_train_lps)]), 'ko', markersize=ms)   # plot for the max train one

    plt.ylabel('Normalized LL (bits/s)')
    plt.xlabel('Number of states')
    plt.xticks([0, 1] + num_states_configs, labels=['Chance', 'GLM'] + num_states_configs)

    # Rotate only the first 2 tick labels
    for i, label in enumerate(ax.get_xticklabels()):
        if i < 2:
            label.set_rotation(90)
    # plt.title(model_prefix.upper())
    plt.title('GLM-HMM')
    plt.legend(loc='lower right')
    plt.margins(0.1)
    plt.grid(alpha=0.15)
    # plt.tight_layout()
    if savefig:
        plt.savefig(f'models/{path}/{model_prefix}_{path}_ll_cv{filesuffix}.pdf', bbox_inches='tight', dpi=300)
    if display:
        plt.show()
    return


def plotCV_same_model_Score(path, model_prefix, num_states_configs, plot_all_test=False, filesuffix='', score_type='r2'):
    """
    :param score_type: 'r2' or 'pearson'
    """

    if score_type not in ['r2', 'pearson']:
        raise Exception(f'Unsupported score type "{score_type}".')

    plt.figure(figsize=(10, 6), constrained_layout=True)
    ax=plt.gca()
    ms = 5

    # skip chance model for r2 and pearson
    # Plot for num_states=1 i.e. linear regression
    lr_train_scores, lr_test_scores = loadCV_Scores(path, 'lr', 1, score_type=score_type)
    train_jitter = np.random.uniform(-0.25, 0.25, size=len(lr_train_scores))
    plt.plot(1+train_jitter, lr_train_scores, 'ko', mfc='none', markersize=ms, label='Train')
    if plot_all_test:
        test_jitter = np.random.uniform(-0.25, 0.25, size=len(lr_test_scores))
        plt.plot(1+test_jitter, lr_test_scores, 'ko', markersize=ms, label='Held-out')
    else:
        plt.plot(1, (lr_test_scores[np.argmax(lr_train_scores)]), 'ko', markersize=ms, label='Held-out')   # plot for the max train one

    # Plot for num_states > 1 now
    for i, s in enumerate(num_states_configs):
        hmm_train_scores, hmm_test_scores = loadCV_Scores(path, model_prefix, s, score_type=score_type)
        print(f"{model_prefix}: num_states={s} Train: {len(hmm_train_scores)} Test:{len(hmm_test_scores)}")
        train_jitter = np.random.uniform(-0.25, 0.25, size=len(hmm_train_scores))
        plt.plot(s+train_jitter, hmm_train_scores, 'ko', mfc='none', markersize=ms)
        # plt.errorbar(s + 0.4, np.mean(hmm_train_lps*factor_bits_per_sec), yerr=np.std(hmm_train_lps*factor_bits_per_sec), color='k', fmt='o', capsize=0)

        if plot_all_test:
            test_jitter = np.random.uniform(-0.25, 0.25, size=len(hmm_test_scores))
            plt.plot(s+test_jitter, hmm_test_scores, 'ko', markersize=ms)
        else:
            if len(hmm_train_scores):
                plt.plot(s, (hmm_test_scores[np.argmax(hmm_train_scores)]), 'ko', markersize=ms)   # plot for the max train one

    if score_type == 'r2':
        plt.ylabel('Var Explained (%)')
    elif score_type == 'pearson':
        plt.ylabel(r'Pearson $r$')
    else:
        raise Exception(f'Unsupported score type "{score_type}".')

    plt.xlabel('Number of states')
    plt.xticks([1] + num_states_configs, labels=['GLM'] + num_states_configs)

    # Rotate only the first 2 tick labels
    for i, label in enumerate(ax.get_xticklabels()):
        if i < 2:
            label.set_rotation(90)
    # plt.title(model_prefix.upper())
    plt.title('GLM-HMM')
    plt.legend(loc='lower right')
    plt.margins(0.1)
    plt.grid(alpha=0.15)
    # plt.tight_layout()
    if savefig:
        plt.savefig(f'models/{path}/{model_prefix}_{path}_{score_type}_cv{filesuffix}.pdf', bbox_inches='tight', dpi=300)
    if display:
        plt.show()
    return


if __name__ == '__main__':

    savefig = True
    display = False

    # path = 'dec25_initseedscv_wt_fred'
    path = 'dec25_kfoldcv_wt'
    plot_all_test = True

    num_states_configs = [ 2, 3, 4, 5, 6, 7, 8, 10 ]
    plotCV_same_model_LL(path, 'id-glm-hmm', num_states_configs, plot_all_test=plot_all_test, filesuffix='')
    plotCV_same_model_Score(path, 'id-glm-hmm', num_states_configs, plot_all_test=plot_all_test, filesuffix='', score_type='r2')
    plotCV_same_model_Score(path, 'id-glm-hmm', num_states_configs, plot_all_test=plot_all_test, filesuffix='', score_type='pearson')
