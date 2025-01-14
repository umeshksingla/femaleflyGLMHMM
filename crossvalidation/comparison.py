import sys

import matplotlib.pyplot as plt
import os.path

import numpy as np
from plotting import plots
from utils import utils

m1 = 'lrhmm_3/20241213_142846_lane'
m2 = 'lrhmm_3/20241217_175018_watch'
m3 = 'lr/20241213_143304_uncle'

comparison_models = []
model_labels = []

for m in [m3, m1, m2]:
    m, dc, mc = utils.load_specific_path(f'models/{m}')
    comparison_models.append((m, dc, mc))
    ns = m.get('num_states', mc['num_states'])
    model_labels.append(m['prefix'].upper() + '_' + str(ns))

folder = '-vs-'.join([m.split("/")[-1] for m in [m1, m2, m3]])
print(folder)
os.makedirs(f'models/comparisons/{folder}/', exist_ok=True)


def generate_trajectory_plots(train=False):

    prefix = 'train' if train else 'test'
    data_key = f'{prefix}_data'
    emissions_key = f'{prefix}_emissions'
    predictions_key = f'{prefix}_predictions'
    # emission_labels = data_config['emission_labels']

    sessions = np.random.choice(range(len(comparison_models[0][0][data_key][emissions_key])), 5, replace=False)   # plotting 5 sessions is enough for now
    for btch in sessions:
        emissions = np.array([m[data_key][predictions_key][btch] for m, _, _ in comparison_models])
        true_emissions = comparison_models[0][0][data_key][emissions_key][btch]
        emission_labels = comparison_models[0][1]['emission_labels']
        plots.plot_hmm_data_whole_session_multiple_models(
            emissions, true_emissions, model_labels=model_labels, y_labels=emission_labels,
            title=f'Predicted female trajectory ({prefix.capitalize()}:{btch})')
        os.makedirs(f'models/comparisons/{folder}/trajs', exist_ok=True)
        plt.savefig(f'models/comparisons/{folder}/trajs/{prefix}_{btch}.pdf', dpi=300, bbox_inches='tight', transparent=True)
        plt.show()
        plt.close()

        for xlim in [(0, 1000), (10000, 11000), (15000, 16000), (0, 5000), (0, 200), (1500, 1600)]:
            plots.plot_hmm_data_whole_session_multiple_models(
                emissions, true_emissions, model_labels=model_labels, y_labels=emission_labels, xlim=xlim,
                title=f'Predicted female trajectory ({prefix.capitalize()}:{btch})')
            plt.savefig(f'models/comparisons/{folder}/trajs/{prefix}_{btch}_xlim={xlim}.pdf', dpi=300,
                        bbox_inches='tight', transparent=True)
            plt.show()
            plt.close()
            # break
        # break


def plot_r2_by_o():
    fig, ax = plt.subplots(1, len(comparison_models), sharey=True, figsize=(10, 10))
    for i, (m, dc, _) in enumerate(comparison_models):

        emission_labels = dc['emission_labels']

        score_by_o_dict = m['train_data'][f'train_score_by_o']
        ax[i].plot(list(score_by_o_dict.keys()), list(score_by_o_dict.values()), 'bo', label='Train')

        score_by_o_dict = m['test_data'][f'test_score_by_o']
        ax[i].plot(list(score_by_o_dict.keys()), list(score_by_o_dict.values()), 'ro', label='Test')

        ax[i].set_title(model_labels[i])
        ax[i].margins(0.1)
        if i == 0:
            ax[i].legend()
        ax[i].set_xticks(range(len(score_by_o_dict)), emission_labels, rotation=90)
    fig.supylabel('R2 scores')
    plt.suptitle('Fraction variance explained')
    plt.savefig(f'models/comparisons/{folder}/score_by_o.pdf', dpi=300, bbox_inches='tight', transparent=True)
    plt.show()
    plt.close()
    return


def correlation_by_o():
    fig, ax = plt.subplots(1, len(comparison_models), sharey=True, figsize=(10, 10))
    for i, (m, dc, _) in enumerate(comparison_models):

        emission_labels = dc['emission_labels']

        score_by_o_dict = m['train_data'][f'train_correlation_by_o']
        ax[i].plot(list(score_by_o_dict.keys()), list(score_by_o_dict.values()), 'bo', label='Train')

        score_by_o_dict = m['test_data'][f'test_correlation_by_o']
        ax[i].plot(list(score_by_o_dict.keys()), list(score_by_o_dict.values()), 'ro', label='Test')

        ax[i].set_title(model_labels[i])
        ax[i].margins(0.1)
        if i == 0:
            ax[i].legend()
        ax[i].set_xticks(range(len(score_by_o_dict)), emission_labels, rotation=90)
    fig.supylabel('Correlation scores [-1, 1]')
    plt.suptitle('Correlation (Lag = 0)')
    plt.savefig(f'models/comparisons/{folder}/correlation_by_o.pdf', dpi=300, bbox_inches='tight', transparent=True)
    plt.show()
    plt.close()
    return


plot_r2_by_o()
correlation_by_o()
generate_trajectory_plots(train=False)
generate_trajectory_plots(train=True)
sys.exit()

# plot data vs LRHMM distributions
for btch in range(len(comparison_models[0]['test_data']['test_emissions'])):
    d = 0
    for _ in data_config["emission_labels"]:
        plt.figure()
        model_pred = lrhmm_pkl['test_data']['test_predictions'][btch, :, d]
        data = lrhmm_pkl['test_data']['test_emissions'][btch, :, d]
        plt.scatter(model_pred, data, marker='.', s=0.5)

        bins = np.linspace(-10, 10, 300)
        binned_means = np.array([
            (np.mean(model_pred[(model_pred < c2) & (model_pred >= c1)]), np.mean(data[(data < c2) & (data >= c1)]))
            for c1, c2 in zip(bins[:-1], bins[1:])
        ])
        print(binned_means)
        plt.scatter(binned_means[:, 0], binned_means[:, 1], s=10, c='orange', alpha=1, label='mean')

        # for l in bins:
        #     plt.axhline(l, ls=':', alpha=0.5)
        #     plt.axvline(l, ls=':', alpha=0.5)

        plt.axhline(0, ls=':')
        plt.axvline(0, ls=':')
        plt.plot(np.linspace(-1, 3, 100), np.linspace(-1, 3, 100), '-.', c='k', label='y=x line')
        plt.xlabel('LR-HMM')
        plt.ylabel('Data')
        plt.axis('equal')
        plt.legend()
        ax = plt.gca()
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        # if d == 0:
        #     plt.xlim([-3, 8])
        #     plt.ylim([-3, 8])
        plt.title(f'{data_config["emission_labels"][_]}: data vs predictions (Test: {btch})')
        os.makedirs(f'models/comparisons/{folder}/verify')
        plt.savefig(f'models/comparisons/{folder}/verify/lrhmm_vs_data_{data_config["emission_labels"][_]}_test{btch}.pdf', dpi=300, bbox_inches='tight', transparent=True)
        # plt.show()
        plt.close()
        d += 1

