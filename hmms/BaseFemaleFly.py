import matplotlib.pyplot as plt
import scipy
import numpy as np
from sklearn.metrics import r2_score
from scipy.ndimage import gaussian_filter1d


def r2_score_custom(y_true, preds_per_state, gamma):
    """
    Compute R^2 per state, works for both scalar (1D) and vector (D>1) outputs.

    Parameters:
    - y_true: (T,) or (T, D)
    - preds_per_state: (T, K) or (T, K, D)
    - gamma: (T, K)

    Returns:
    - r2: (K,) array of R^2 values per state
    """
    # If y_true is 1D, make it (T, 1)
    if y_true.ndim == 1:
        y_true = y_true[:, None]  # (T, 1)

    T, K = gamma.shape
    D = y_true.shape[1]

    if preds_per_state.ndim == 2:
        preds_per_state = preds_per_state[:, :, None]  # (T, K, 1)

    gamma_sum = gamma.sum(axis=0)[:, None]  # (K, 1)
    mean_y_k = (gamma.T @ y_true) / gamma_sum  # (K, D)

    resid_sq = gamma[:, :, None] * (y_true[:, None, :] - preds_per_state) ** 2  # (T, K, D)
    total_sq = gamma[:, :, None] * (y_true[:, None, :] - mean_y_k[None, :, :]) ** 2  # (T, K, D)

    num = np.array(resid_sq.sum(axis=0).sum(axis=1))  # (K,)
    den = np.array(total_sq.sum(axis=0).sum(axis=1))  # (K,)

    r2 = 1.0 - num / den    # if both are non-zero, then regular r2
    r2[(den == 0) & (num == 0)] = 0.0  # if both are zero, then 0.0
    return r2


class BaseFemaleFly:
    def __init__(self):
        self.fit_success = None

    def fit(self, emissions, inputs, output_mn_std=None):
        raise NotImplementedError

    def update_status(self):
        self.fit_success = self.check_nan_in_fit_params()
        print("SUCCEEDED?", self.fit_success)
        return

    def check_nan_in_fit_params(self):
        raise NotImplementedError

    def predict(self, emissions, inputs):
        raise NotImplementedError

    def score(self, emissions, y_preds):
        y_preds = np.concatenate(y_preds, axis=0)
        emissions = np.concatenate(emissions, axis=0)
        r = r2_score(emissions, y_preds, multioutput='variance_weighted')
        # print("separately", r2_score(emissions, y_preds, multioutput='raw_values'))
        return r

    def scores_by_fly(self, emissions, y_preds):
        r2s = np.array([r2_score(emissions[i], y_preds[i], multioutput='variance_weighted') for i in range(len(emissions))])
        # print("r2s", r2s)
        return r2s

    def score_by_o(self, emissions, y_preds):
        y_preds = np.concatenate(y_preds, axis=0)
        emissions = np.concatenate(emissions, axis=0)
        r2_o = dict(zip(range(self.data_config["emission_dim"]), r2_score(emissions, y_preds, multioutput='raw_values')))
        # print("r2_o", r2_o)
        return r2_o

    def score_by_o_by_fly(self, emissions, y_preds):
        r2_by_o = {}
        for o in range(self.data_config["emission_dim"]):
            r2_by_o[o] = []
            for i in range(len(emissions)):
                if not len(emissions[i]):  # if the session is empty, can happen when emissions are passed after z_masking
                    r = 0.
                else:
                    r = r2_score(emissions[i][:, o], y_preds[i][:, o])
                r2_by_o[o].append(r)
            r2_by_o[o] = np.array(r2_by_o[o])
        # print("r2_by_o by_fly", r2_by_o)
        return r2_by_o

    def score_by_z(self, emissions, y_preds, z_seqs):
        y_preds = np.concatenate(y_preds, axis=0)
        emissions = np.concatenate(emissions, axis=0)
        z_seqs = np.concatenate(z_seqs, axis=0)
        r2_z = {}
        for z in range(self.num_states):
            z_mask = z_seqs == z
            r = r2_score(emissions[z_mask], y_preds[z_mask], multioutput='variance_weighted')
            r2_z[z] = r
        # print("r2_z", r2_z)
        return r2_z

    def score_by_z_soft(self, emissions, y_preds_per_state, z_probs):
        y_preds_per_state = np.concatenate(y_preds_per_state, axis=0)
        emissions = np.concatenate(emissions, axis=0)
        z_probs = np.concatenate(z_probs, axis=0)
        r = r2_score_custom(emissions, y_preds_per_state, z_probs)
        r2_z = dict(zip(range(self.num_states), r))
        # print("r2_z soft", r2_z)
        return r2_z

    def score_by_z_by_fly(self, emissions, y_preds, z_seqs):
        r2_z_by_fly = {}
        for z in range(self.num_states):
            r2_z_by_fly[z] = []
            for i in range(len(emissions)):
                z_mask = z_seqs[i] == z
                if np.sum(z_mask):
                    r = r2_score(emissions[i][z_mask], y_preds[i][z_mask], multioutput='variance_weighted')
                    r2_z_by_fly[z].append(r)
                else:
                    r2_z_by_fly[z].append(0.)
                    print(f'nothing for state {z} for session {i}')
            r2_z_by_fly[z] = np.array(r2_z_by_fly[z])
        # print("r2_z_by_fly", r2_z_by_fly)
        return r2_z_by_fly

    def score_by_z_by_fly_soft(self, emissions, y_preds_per_state, z_probs):
        r2_z_by_fly = {}
        for z in range(self.num_states):
            r2_z_by_fly[z] = []

        for i in range(len(emissions)):
            r = r2_score_custom(emissions[i], y_preds_per_state[i], z_probs[i])
            # print(i, r, np.var(emissions[i]))
            for z in range(self.num_states):
                r2_z_by_fly[z].append(r[z])

        for z in range(self.num_states):
            r2_z_by_fly[z] = np.array(r2_z_by_fly[z])
        # print("r2_z_by_fly soft", r2_z_by_fly)
        return r2_z_by_fly

    def score_by_z_and_o(self, emissions, y_preds, z_seqs):
        y_preds = np.concatenate(y_preds, axis=0)
        emissions = np.concatenate(emissions, axis=0)
        z_seqs = np.concatenate(z_seqs, axis=0)
        r2_z_o = {}
        for z in range(self.num_states):
            r2_z_o[z] = {}
            z_mask = z_seqs == z
            for o in range(self.data_config["emission_dim"]):
                if np.sum(z_mask):
                    r2_z_o[z][o] = r2_score(emissions[z_mask][:, o], y_preds[z_mask][:, o])
                else:
                    print(f'nothing for state {z} for emission {o}')
        # print("r2_z_o", r2_z_o)
        return r2_z_o

    def score_by_z_and_o_soft(self, emissions, y_preds_per_state, z_probs):
        y_preds_per_state = np.concatenate(y_preds_per_state, axis=0)
        emissions = np.concatenate(emissions, axis=0)
        z_probs = np.concatenate(z_probs, axis=0)

        r2_z_o = {}
        for z in range(self.num_states):
            r2_z_o[z] = {}

        for o in range(self.data_config["emission_dim"]):
            r = r2_score_custom(emissions[:, o], y_preds_per_state[..., o], z_probs)
            for z in range(self.num_states):
                r2_z_o[z][o] = r[z]

        # print("r2_z_o soft", r2_z_o)
        return r2_z_o

    def score_by_z_and_o_by_fly(self, emissions, y_preds, z_seqs):
        r2_z_o = {}
        for z in range(self.num_states):
            emissions_z = [emissions[i][z_seqs[i] == z] for i in range(len(emissions))]
            y_preds_z = [y_preds[i][z_seqs[i] == z] for i in range(len(emissions))]
            r2_z_o[z] = self.score_by_o_by_fly(emissions_z, y_preds_z)
        # print("r2_z_o by fly", r2_z_o)
        return r2_z_o

    def score_by_z_and_o_by_fly_soft(self, emissions, y_preds_per_state, z_probs):
        r2_z_o_by_fly = {}

        for z in range(self.num_states):
            r2_z_o_by_fly[z] = {}
            for o in range(self.data_config["emission_dim"]):
                r2_z_o_by_fly[z][o] = []

        for i in range(len(emissions)):
            for o in range(self.data_config["emission_dim"]):
                r = r2_score_custom(emissions[i][:, o], y_preds_per_state[i][..., o], z_probs[i])
                for z in range(self.num_states):
                    r2_z_o_by_fly[z][o].append(r[z])

        for z in range(self.num_states):
            for o in range(self.data_config["emission_dim"]):
                r2_z_o_by_fly[z][o] = np.array(r2_z_o_by_fly[z][o])

        # print("r2_z_o_by_fly soft", r2_z_o_by_fly)
        return r2_z_o_by_fly

    def correlation_by_o(self, emissions, y_preds):
        y_preds = np.concatenate(y_preds, axis=0)
        emissions = np.concatenate(emissions, axis=0)
        corrs_dict = {}
        for o in range(self.data_config["emission_dim"]):
            a = y_preds[:, o]
            b = emissions[:, o]
            c = scipy.signal.correlate(a, b, mode='valid') / (np.linalg.norm(a) * np.linalg.norm(b))
            corrs_dict[o] = c[0]     # zero lag correlation only at the moment
        return corrs_dict

    def correlation_by_o_by_fly(self, emissions, y_preds):
        corrs_dict = {}
        for o in range(self.data_config["emission_dim"]):
            corrs_dict[o] = []
            for i in range(len(emissions)):
                a = y_preds[i][:, o]
                b = emissions[i][:, o]
                c = scipy.signal.correlate(a, b, mode='valid') / (np.linalg.norm(a) * np.linalg.norm(b))
                corrs_dict[o].append(c[0])     # zero lag correlation only at the moment
            corrs_dict[o] = np.array(corrs_dict[o])
        return corrs_dict

    def correlation_max_by_o(self, emissions, y_preds):
        """To summarize: with the calculation done as above, a positive lag means the first series lags the second,
        or the second leads the first--peaks earlier in time, so at a location to the left on the time series plot.
        Source: https://currents.soest.hawaii.edu/ocn_data_analysis/_static/SEM_EDOF.html

        +ve lags at the peak will mean "a" lags "b", or "b" leads "a". So, a=model lags b=truth, i.e. model prediction
        is delayed — it aligns best with future ground truth, meaning it's lagging behind the behavior.
        """
        y_preds = np.concatenate(y_preds, axis=0)
        emissions = np.concatenate(emissions, axis=0)
        corrs_dict = {}
        lags_dict = {}
        for o in range(self.data_config["emission_dim"]):
            a = y_preds[:, o]       # a = model, predicted
            b = emissions[:, o]     # b = data, truth
            a = a - np.mean(a)  # ensure both are mean-centered
            b = b - np.mean(b)
            print(np.mean(a), np.mean(b))
            c = scipy.signal.correlate(a, b, mode='full') / (np.linalg.norm(a) * np.linalg.norm(b))
            lags = np.arange(-len(a) + 1, len(a))
            corrs_dict[o] = np.max(c)
            lags_dict[o] = lags[np.argmax(c)]
        return corrs_dict, lags_dict

    def correlation_max_by_o_by_fly(self, emissions, y_preds):
        corrs_dict = {}
        lags_dict = {}
        for o in range(self.data_config["emission_dim"]):
            corrs_dict[o] = []
            lags_dict[o] = []
            for i in range(len(emissions)):
                a = y_preds[i][:, o]
                b = emissions[i][:, o]
                a = a - np.mean(a)  # ensure both are mean-centered
                b = b - np.mean(b)
                print(i, np.mean(a), np.mean(b))
                c = scipy.signal.correlate(a, b, mode='full') / (np.linalg.norm(a) * np.linalg.norm(b))
                lags = np.arange(-len(a) + 1, len(a))
                corrs_dict[o].append(np.max(c))
                lags_dict[o].append(lags[np.argmax(c)])
            corrs_dict[o] = np.array(corrs_dict[o])
            lags_dict[o] = np.array(lags_dict[o])
        return corrs_dict, lags_dict
