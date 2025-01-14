import joblib
import numpy as np
from scipy.stats import zscore
import scipy
import matplotlib.pyplot as plt


DATA = 'wt'
# BASE_FOLDER = f' notebooks/eda/{DATA}/'
sessions_features = joblib.load('../data/sessions_features_82_isong.pkl')


def get_output_mu_std(s, f_name):
    if f_name in ['fFV', 'fFS', 'fLS', 'fLV', 'fFA']:
        ts = sessions_features[s][f_name]
    elif f_name in ['dfTheta']:
        dfTheta = np.diff(sessions_features[s]['fTheta'])
        ts = np.where(np.abs(dfTheta) > 90, 0, dfTheta)
    return np.mean(ts), np.std(ts)


def run():

    num_timesteps = 100000

    fig, ax = plt.subplots(3, 2, sharex=True)

    for s_i, s in enumerate(sessions_features):
        if s_i == 0:
            print(sessions_features[s].keys())
        session_len = len(sessions_features[s]['mFV'])
        if session_len < num_timesteps:    # skip short sessions for now
            print(f"Session {s_i} length = {len(sessions_features[s]['mFV'])}. Too short. Skipped.")
            continue
        # if session_len > 250000:
        #     print(f"Session {s_i} length = {len(sessions_features[s]['mFV'])}. No copulation. Skipped.")
        #     continue

        noise = np.random.uniform(-0.5, 0.5)

        mu, std = get_output_mu_std(s, 'fFV')
        ax[0, 0].plot(0+noise, mu, 'r.')
        # ax[0, 0].errorbar(0 + noise, mu, yerr=std, fmt='r.')
        ax[0, 1].plot(0+noise, std, 'r.')

        mu, std = get_output_mu_std(s, 'fLV')
        ax[1, 0].plot(0+noise, mu, 'r.')
        ax[1, 1].plot(0+noise, std, 'r.')

        mu, std = get_output_mu_std(s, 'dfTheta')
        ax[2, 0].plot(0 + noise, mu, 'r.')
        ax[2, 1].plot(0 + noise, std, 'r.')

    ax[0, 0].set_title('fFV means')
    ax[0, 1].set_title('fFV stds')
    ax[1, 0].set_title('fLV means')
    ax[1, 1].set_title('fLV stds')
    ax[2, 0].set_title('dfTheta means')
    ax[2, 1].set_title('dfTheta stds')

    ax[0, 0].set_xlim(-2, 2)

    for _ in ax:
        for __ in _:
            __.set_xticks([])

    plt.show()
    return


if __name__ == '__main__':
    run()
