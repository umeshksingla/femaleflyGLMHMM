####################################

# Usage: python comparison_cv.py

####################################

import glob
import random
import joblib
from pprint import pprint
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy import stats
import pandas as pd

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


def loadCV_Scores(path, model_path_prefixes, num_states_configs, precomputed=False, score_type=None):
    """
    :param score_type: 'r2' or 'pearson' or 'll'
    """
    from utilities import utils

    if precomputed:
        scores_dict = joblib.load(f'{path}_all_scores_dict.pkl')
        train_scores = scores_dict[model_path_prefix][num_states]['train'][score_type]
        test_scores = scores_dict[model_path_prefix][num_states]['test'][score_type]
        # print('train', score_type, f'num_states={num_states}', train_scores)
        # print('test', score_type, f'num_states={num_states}', test_scores)
        return train_scores, test_scores

    # all_train_scores = {}
    # all_test_scores = {}
    # for _ in ['r2', 'pearson', 'll']:
    #     all_train_scores[_] = []
    #     all_test_scores[_] = []

    # r2_train_score = pkl['train_data']['train_score'] * 100
    # r2_test_score = pkl['test_data']['test_score'] * 100
    # pearson_train_score = pkl['train_data']['train_pearson']
    # pearson_test_score = pkl['test_data']['test_pearson']
    # factor_bits_per_sec = data_config_pkl['effective_fps']/np.log(2)
    # ll_train_score = pkl['train_data']['train_lp'].item() * factor_bits_per_sec
    # ll_test_score = pkl['test_data']['test_lp'].item() * factor_bits_per_sec
    # ll_fly_train_score = pkl['train_data']['train_lps_by_fly'] * factor_bits_per_sec
    # ll_fly_test_score = pkl['test_data']['test_lps_by_fly'] * factor_bits_per_sec
    
    # all_train_scores['r2'].append(r2_train_score)
    # all_test_scores['r2'].append(r2_test_score)
    # all_train_scores['pearson'].append(pearson_train_score)
    # all_test_scores['pearson'].append(pearson_test_score)
    # all_train_scores['ll'].append(ll_train_score)
    # all_test_scores['ll'].append(ll_test_score)

    # for _ in all_train_scores:
    #     all_train_scores[_] = np.array(all_train_scores[_])
    #     all_test_scores[_] = np.array(all_test_scores[_])
    # return all_train_scores, all_test_scores

    rows = []
    for model_path_prefix in model_path_prefixes:
        for num_states in num_states_configs:
            model_pkl_paths = sorted(glob.glob(f'models/{path}/{model_path_prefix}_{num_states}_cv/**/'))
            random.shuffle(model_pkl_paths)

            c = 0
            for _ in model_pkl_paths:
                pkl, data_config_pkl, model_config = utils.load_specific_path(_)
                if pkl is None: continue
                datasplit_seed = model_config['datasplit_seed']
                init_seed = model_config['seed']
                name = model_path_prefix.split('/')[-1]
                for group in ['train', 'test']:
                    r2_score = pkl[f'{group}_data'][f'{group}_score'] * 100
                    pearson_score = pkl[f'{group}_data'][f'{group}_pearson']
                    factor_bits_per_sec = data_config_pkl['effective_fps']/np.log(2)
                    ll_score = pkl[f'{group}_data'][f'{group}_lp'].item() * factor_bits_per_sec
                    row = [name, num_states, group, datasplit_seed, init_seed, r2_score, pearson_score, ll_score, _]
                    rows.append(row)
                c += 1
                # if c == 2: break
    scores_df = pd.DataFrame(rows, columns=['model', 'num_states', 'group', 'datasplit_seed', 'init_seed', 'r2_score', 'pearson_score', 'll_score', 'path'])
    print(scores_df)
    return scores_df


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


def plotCV_same_model_LL_WIP(path, model_prefixes, num_states_configs, group, precomputed=False, onlyerrbars=True):
    '''
    group = 'train' or 'test'
    '''

    plt.figure(figsize=(10, 6), constrained_layout=True)
    ax=plt.gca()
    ms = 2

    colors = {
        'lrhmmci_': 'r',
        'id-glm-hmm': 'b',
    }

    labels = {
        'lrhmmci_': 'w/o GLM Transitions',
        'id-glm-hmm': 'w GLM Transitions',
    }

    # Plot for num_states > 1
    mps = []
    for mi, mpath in enumerate(model_prefixes):
        mp = mpath.split('/')[-1]
        ss = []
        mns = []
        stds = []
        sems = []
        for i, s in enumerate(num_states_configs):
            hmm_train_lps, hmm_test_lps = loadCV_Scores(path, mpath, s, score_type='ll', precomputed=precomputed)
            # print(f"{mp}: num_states={s} Train: {len(hmm_train_lps)} Test:{len(hmm_test_lps)}")
            train_jitter = np.random.uniform(-0.25, 0.25, size=len(hmm_train_lps))

            if group == 'train':
                mn = np.mean(hmm_train_lps)
                std = np.std(hmm_train_lps)
                sem = stats.sem(hmm_train_lps)
            elif group == 'test':
                mn = np.mean(hmm_test_lps)
                std = np.std(hmm_test_lps)
                sem = stats.sem(hmm_test_lps)

            ss.append(s)  # s + 0.1 * pow(-1, mi+1)
            mns.append(mn)
            stds.append(std)
            sems.append(sem)
            print(f"model={mp} ({group}) \t\t states={s} LL = {mn} ± {sem}")

            if not onlyerrbars:
                plt.plot(s+train_jitter, hmm_train_lps, 'o', color=colors[mp], mfc='none', markersize=ms)
    
        plt.errorbar(ss, mns, yerr=sems, color=colors[mp], fmt='o-', markersize=ms, capsize=0, label=labels[mp])
        mps.append(mp)

    plt.ylabel('Normalized LL (bits/s)')
    plt.xlabel('Number of states')
    plt.xticks(num_states_configs, labels=num_states_configs)
    plt.title('GLM-HMM')
    plt.legend(loc='lower right')
    plt.margins(0.1)
    plt.grid(alpha=0.15)
    # plt.tight_layout()
    if savefig:
        plt.savefig(f'models/cv_figs/{mps}_ll_cv_{group}.pdf', bbox_inches='tight', dpi=300)
    if display:
        plt.show()
    return


def plot_ll_scores(
    df, 
    model_prefixes=['id-glm-hmm'], 
    groups=['train', 'test'], 
    num_states=[1, 2, 3, 4, 5, 6, 7, 8, 10], 
    plot_type='mean_sem',   # 'mean_sem' or 'all_points'
    # group_by_seed=None,     # None, 'datasplit_seed', or 'init_seed'
    figsize=(10, 6)
):
    """
    Plots ll_score vs num_states for specified models, groups, and seeds.
    """
    np.random.seed(0)

    # 1. Filter for specific states and groups
    plot_df = df[df['num_states'].isin(num_states)].copy()
    plot_df = plot_df[plot_df['group'].isin(groups)]
    
    # 2. Filter and extract model prefixes (to handle names like 'hmm_1', 'glm_base')
    def get_prefix(model_name):
        for prefix in model_prefixes:
            if model_name.startswith(prefix):
                return prefix
        return None
        
    plot_df['model_prefix'] = plot_df['model'].apply(get_prefix)
    plot_df = plot_df.dropna(subset=['model_prefix'])
    
    # 3. Determine how we are grouping the data for lines/colors
    group_cols = ['model_prefix', 'group']
    # if group_by_seed:
    #     group_cols.append(group_by_seed)
        
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    colors = {
        'lrhmmci_': 'r',
        'id-glm-hmm': 'b',
    }

    labels = {
        'lrhmmci_': 'w/o GLM Transitions',
        'id-glm-hmm': 'w GLM Transitions',
    }
    
    # 4. Group data and plot
    for name, group_df in plot_df.groupby(group_cols):
        # 'name' is a tuple representing the current group
        label = " | ".join(map(str, name))

        # Sort to ensure lines connect left-to-right properly
        group_df = group_df.sort_values('num_states')
        
        if plot_type == 'all_points':
            ax.scatter(group_df['num_states'] + np.random.uniform(-0.1, 0.1, size=len(group_df)),
                        group_df['ll_score'], color=colors[name[0]], edgecolors='none', s=5, alpha=0.4) #, label=label)

        # elif plot_type == 'mean_sem':
        # Calculate mean and standard error of the mean
        stats_df = group_df.groupby('num_states')['ll_score'].agg(['mean', 'sem']).reset_index()
        stats_df['sem'] = stats_df['sem'].fillna(0) # Handle single-point edge cases

        ax.errorbar(
            stats_df['num_states'], 
            stats_df['mean'], 
            yerr=stats_df['sem'], 
            marker='o', 
            capsize=0,
            # linewidth=2,
            markersize=2,
            label=labels[name[0]],
            color=colors[name[0]],
        )

    # 5. Formatting and Aesthetics
    ax.set_xticks(num_states)
    ax.set_xlabel('Number of States')
    ax.set_ylabel('Normalized LL (bits/s)')

    plt.title(f'GLM-HMM ({name[1].title()})')

    handles, labels = ax.get_legend_handles_labels()
    cleaned_handles = [ h[0] if type(h).__name__ == 'ErrorbarContainer' else h for h in handles]
    ax.legend(cleaned_handles, labels, loc='lower right')

    plt.margins(0.1)
    plt.grid(alpha=0.15)
    plt.ylim([90, 280])
    if savefig:
        filename = f'{model_prefixes}_{groups}_{plot_type}.pdf'
        print(f'Saved at: {filename}')
        plt.savefig(f'models/cv_figs/{filename}', bbox_inches='tight', dpi=300)
    if display:
        plt.show()
    return



def plotCV_same_model_LL_by_fly(path, model_prefix, num_states_configs, plot_only_test=False, filesuffix=''):

    plt.figure(figsize=(10, 6), constrained_layout=True)
    ax=plt.gca()
    ms = 5

    # Plot for num_states=0 i.e. chance
    s = 0
    chance_train_lps, chance_test_lps = loadCV_Scores(path, 'chance', 0, score_type='ll_fly')

    if not plot_only_test:
        train_jitter = np.random.uniform(-0.25, 0.25, size=len(chance_train_lps))
        plt.plot(s+train_jitter, np.zeros_like(chance_train_lps), 'ko', mfc='none', markersize=ms)  # plot 0s for chance as chance_lps stored are not relative to chance.

    test_jitter = np.random.uniform(-0.25, 0.25, size=len(chance_test_lps))
    plt.plot(s+test_jitter, np.zeros_like(chance_test_lps), 'ko', markersize=ms)

    # Plot for num_states=1 i.e. linear regression
    s = 1
    lr_train_lps, lr_test_lps = loadCV_Scores(path, 'lr', 1, score_type='ll_fly')

    if not plot_only_test:
        train_jitter = np.random.uniform(-0.25, 0.25, size=len(lr_train_lps))
        plt.plot(s+train_jitter, lr_train_lps, 'ko', mfc='none', markersize=ms, label='Train')

    test_jitter = np.random.uniform(-0.25, 0.25, size=len(lr_test_lps))
    plt.plot(s+test_jitter, lr_test_lps, 'ko', markersize=ms, label='Held-out')
    plt.errorbar(s + 0.4, np.mean(lr_test_lps), yerr=np.std(lr_test_lps), color='k', fmt='o', capsize=0)


    # Plot for num_states > 1 now
    for i, s in enumerate(num_states_configs):
        hmm_train_lps, hmm_test_lps = loadCV_Scores(path, model_prefix, s, score_type='ll_fly')
        print(f"{model_prefix}: num_states={s} Train: {len(hmm_train_lps)} Test:{len(hmm_test_lps)}")

        if not plot_only_test:
            train_jitter = np.random.uniform(-0.25, 0.25, size=len(hmm_train_lps))
            plt.plot(s+train_jitter, hmm_train_lps, 'ko', mfc='none', markersize=ms)
        # plt.errorbar(s + 0.4, np.mean(hmm_train_lps), yerr=np.std(hmm_train_lps), color='k', fmt='o', capsize=0)

        test_jitter = np.random.uniform(-0.25, 0.25, size=len(hmm_test_lps))
        plt.plot(s+test_jitter, hmm_test_lps, 'ko', markersize=ms)
        plt.errorbar(s + 0.4, np.mean(hmm_test_lps), yerr=np.std(hmm_test_lps), color='k', fmt='o', capsize=0)


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
        plt.savefig(f'models/{path}/{model_prefix}_{path}_ll_by_fly_cv{filesuffix}.pdf', bbox_inches='tight', dpi=300)
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


def scores_dump(path, model_prefixes, num_states_configs):
    hmm_scores_df = loadCV_Scores(path, model_prefixes, num_states_configs)
    print(hmm_scores_df)
    joblib.dump(hmm_scores_df, f'{path}_hmm_scores_df.pkl')
    return

    scores_dict = {}
    for mi, mp in enumerate(model_prefixes):
        scores_dict[mp] = {}
        for i, s in enumerate(num_states_configs):
            scores_dict[mp][s] = {}
            # hmm_train_scores_dict, hmm_test_scores_dict = loadCV_Scores(path, mp, s)
            # scores_dict[mp][s]['train'] = hmm_train_scores_dict
            # scores_dict[mp][s]['test'] = hmm_test_scores_dict
            hmm_scores_df = loadCV_Scores(path, mp, s)
    print(hmm_scores_df)
    joblib.dump(hmm_scores_df, f'{path}_hmm_scores_df.pkl')
    return


if __name__ == '__main__':
    import sys

    savefig = True
    display = False

    # path = 'jan1_initseedscv_wt_female'
    # path = 'apr6_bothcv_wt_female_2'

    num_states_configs = [ 1, 2, 3, 4, 5, 6, 7, 8, 10 ]
    # num_states_configs = [1, 2, 5]
    # scores_dump('', ['apr6_bothcv_wt_female_2/lrhmmci_', 'apr6_bothcv_wt_female_2/id-glm-hmm'], num_states_configs)
    # sys.exit(0)

    df = joblib.load('new_hmm_scores_df.pkl')
    print(df)

    # # 1. Isolate the training data
    train_df = df[df['group'] == 'train']

    # # 2. Find the index of the max train ll_score for each setup
    best_train_idx = train_df.groupby(['model', 'num_states', 'datasplit_seed'])['ll_score'].idxmax()

    # # 3. Extract the winning combinations of model, states, datasplit, and the resulting best init_seed
    winning_combinations = train_df.loc[best_train_idx, ['model', 'num_states', 'datasplit_seed', 'init_seed']]

    # # 4. Merge this back with the original dataframe to get both 'train' and 'test' rows for those specific seeds
    best_runs_df = pd.merge(
        df, 
        winning_combinations, 
        on=['model', 'num_states', 'datasplit_seed', 'init_seed'], 
        how='inner'
    )
    print(best_runs_df)
    print(winning_combinations)

    # plot_ll_scores(df, model_prefixes=['id-glm-hmm', 'lrhmmci_'], groups=['train'], plot_type='mean_sem')
    # plot_ll_scores(df, model_prefixes=['id-glm-hmm', 'lrhmmci_'], groups=['test'], plot_type='mean_sem')
    # plot_ll_scores(df, model_prefixes=['id-glm-hmm', 'lrhmmci_'], groups=['test'], plot_type='all_points')
    plot_ll_scores(best_runs_df, model_prefixes=['id-glm-hmm', 'lrhmmci_'], groups=['train'], plot_type='all_points')
    plot_ll_scores(best_runs_df, model_prefixes=['id-glm-hmm', 'lrhmmci_'], groups=['test'], plot_type='all_points')

    # plotCV_same_model_LL_WIP('', ['apr6_bothcv_wt_female_2/id-glm-hmm', 'jan1_kfoldcv_wt_female/lrhmmci_'], num_states_configs, group='train', precomputed=True, onlyerrbars=False)
    # plotCV_same_model_LL(path, 'lrhmmci_', num_states_configs, plot_all_test=True, filesuffix='')
    # plotCV_same_model_Score(path, 'lrhmmci_', num_states_configs, plot_all_test=True, filesuffix='', score_type='pearson')
    # plotCV_same_model_Score(path, 'lrhmmci_', num_states_configs, plot_all_test=True, filesuffix='', score_type='r2')
    # plotCV_same_model_LL_by_fly(path, 'lrhmmci_', num_states_configs, plot_only_test=True, filesuffix='plot_only_test')

