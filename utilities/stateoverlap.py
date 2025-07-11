import os.path

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utilities.io import load_specific_path


def f(model_dir1, model_dir2):
    model_ckp1, _, model_config1 = load_specific_path(model_dir1)
    model_ckp2, _, model_config2 = load_specific_path(model_dir2)
    if (model_ckp1 is None) or (model_ckp2 is None):
        return

    num_states1 = model_config1['num_states']
    num_states2 = model_config2['num_states']

    gamma_2 = [*model_ckp2['train_data']['train_stateseq'], *model_ckp2['test_data']['test_stateseq']]
    gamma_2 = np.concatenate(gamma_2, axis=0)
    gamma_1 = [*model_ckp1['train_data']['train_stateseq'], *model_ckp1['test_data']['test_stateseq']]
    gamma_1 = np.concatenate(gamma_1, axis=0)

    print(gamma_2.shape, gamma_1.shape)

    cond = np.empty((num_states1, num_states2))
    for j in range(num_states1):
        for i in range(num_states2):
            cond[j][i] = (gamma_1[gamma_2 == i] == j).mean()

    print(cond)
    plt.figure(figsize=(7, 4))
    ax = plt.gca()
    sns.heatmap(cond, annot=True, cmap='viridis', cbar=True, fmt=".2f",
                xticklabels=[f'{i + 1}' for i in range(num_states2)],
                yticklabels=[f'{i + 1}' for i in range(num_states1)], annot_kws={'size': 'small'}, vmin=0, vmax=1)
    plt.xlabel(f'{num_states2}-state model')
    plt.ylabel(f'{num_states1}-state model')
    plt.yticks(rotation=0)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(length=0)
    cbar.set_ticks([0, 1])
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir1, 'stateoverlap_hard.pdf'), dpi=300, bbox_inches='tight', transparent=True)
    # plt.show()
    return


def f2(model_dir1, model_dir2):
    model_ckp1, _, model_config1 = load_specific_path(model_dir1)
    model_ckp2, _, model_config2 = load_specific_path(model_dir2)
    if (model_ckp1 is None) or (model_ckp2 is None):
        return

    num_states1 = model_config1['num_states']
    num_states2 = model_config2['num_states']

    gamma_1 = [*model_ckp1['train_data']['train_state_probs'], *model_ckp1['test_data']['test_state_probs']]
    gamma_1 = np.concatenate(gamma_1, axis=0)
    gamma_2 = [*model_ckp2['train_data']['train_state_probs'], *model_ckp2['test_data']['test_state_probs']]
    gamma_2 = np.concatenate(gamma_2, axis=0)

    print(gamma_1.shape, gamma_2.shape)
    # Initialize matrix to hold conditional probabilities
    cond_prob = np.empty((num_states2, num_states1))  # rows: 5-state model, cols: 4-state model

    # For each state i in the 5-state model
    for i in range(num_states2):
        # Weight each timepoint by its probability of being in state i of 5-state model
        w_i = gamma_2[:, i]  # shape (T,)

        # Normalizing denominator
        total_w_i = w_i.sum()

        # For each state j in the 4-state model
        for j in range(num_states1):
            joint = (w_i * gamma_1[:, j]).sum()  # joint contribution
            cond_prob[i, j] = joint / total_w_i  # conditional probability P(4=j | 5=i)

    print(cond_prob.T)
    plt.figure(figsize=(7, 4))
    ax = plt.gca()
    sns.heatmap(cond_prob.T, annot=True, cmap='viridis', cbar=True, fmt=".2f",
                xticklabels=[f'{i+1}' for i in range(num_states2)],
                yticklabels=[f'{i+1}' for i in range(num_states1)], annot_kws={'size': 'small'}, vmin=0, vmax=1)
    plt.xlabel(f'{num_states2}-state model')
    plt.ylabel(f'{num_states1}-state model')
    plt.yticks(rotation=0)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(length=0)
    cbar.set_ticks([0, 1])
    # cbar.set_ticklabels(['0'] + ['']*9 + ['1'])
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir1, 'stateoverlap_soft.pdf'), dpi=300, bbox_inches='tight', transparent=True)
    # plt.show()
    return


if __name__ == '__main__':
    # f('../models/general_wt_lrauxem/glm-hmm_4_cv/20250704_232631_chicken', '../models/general_wt_lrauxem/glm-hmm_5_cv/20250704_234515_operating')
    # f2('../models/general_wt_lrauxem/glm-hmm_4_cv/20250704_232631_chicken', '../models/general_wt_lrauxem/glm-hmm_5_cv/20250704_234515_operating')
    #
    # f('../models/general_wt_lrauxem/glm-hmm_5_cv/20250704_234515_operating', '../models/general_wt_lrauxem/glm-hmm_6_cv/20250705_021012_borrow')
    # f2('../models/general_wt_lrauxem/glm-hmm_5_cv/20250704_234515_operating', '../models/general_wt_lrauxem/glm-hmm_6_cv/20250705_021012_borrow')

    # f('../models/general_wt_lrauxem/glm-hmm_3_cv/20250704_215843_termite', '../models/general_wt_lrauxem/glm-hmm_4_cv/20250704_232631_chicken')
    # f2('../models/general_wt_lrauxem/glm-hmm_3_cv/20250704_215843_termite', '../models/general_wt_lrauxem/glm-hmm_4_cv/20250704_232631_chicken')

    f('../models/general_wt_lrauxem/glm-hmm_6_cv/20250705_021012_borrow', '../models/general_wt_lrauxem/glm-hmm_10_cv/20250705_001313_convertible')
    f2('../models/general_wt_lrauxem/glm-hmm_6_cv/20250705_021012_borrow', '../models/general_wt_lrauxem/glm-hmm_10_cv/20250705_001313_convertible')

