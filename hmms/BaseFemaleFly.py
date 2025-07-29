import matplotlib.pyplot as plt
import scipy
import numpy as np
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
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


def pearsonr_custom(y_true, preds_per_state, gamma):
    """
    Compute pearsonr per state, works for **ONLY** scalar (1D) outputs. Does not handle multi-dimensional outputs.

    Parameters:
    - y_true: (T,)
    - preds_per_state: (T, K)
    - gamma: (T, K)

    Returns:
    - pearsonr: (K,) array of pearsonr values per state
    """

    y_true = y_true.squeeze()  # (T,)
    T, K = gamma.shape

    pr = np.zeros(K)

    for k in range(K):
        w = gamma[:, k]
        y_k = y_true
        yhat_k = preds_per_state[:, k]

        # Weighted means
        w_sum = np.sum(w)
        mu_y = np.sum(w * y_k) / w_sum
        mu_yhat = np.sum(w * yhat_k) / w_sum

        # Weighted covariance
        cov = np.sum(w * (y_k - mu_y) * (yhat_k - mu_yhat)) / w_sum

        # Weighted variances
        var_y = np.sum(w * (y_k - mu_y) ** 2) / w_sum
        var_yhat = np.sum(w * (yhat_k - mu_yhat) ** 2) / w_sum

        pr[k] = cov / (np.sqrt(var_y * var_yhat) + 1e-8)  # epsilon for stability

    return pr


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

    # def score_by_o(self, emissions, y_preds):
    #     y_preds = np.concatenate(y_preds, axis=0)
    #     emissions = np.concatenate(emissions, axis=0)
    #     r2_o = dict(zip(range(self.data_config["emission_dim"]), r2_score(emissions, y_preds, multioutput='raw_values')))
    #     # print("r2_o", r2_o)
    #     return r2_o

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

    # def score_by_z(self, emissions, y_preds, z_seqs):
    #     raise NotImplementedError
    #     y_preds = np.concatenate(y_preds, axis=0)
    #     emissions = np.concatenate(emissions, axis=0)
    #     z_seqs = np.concatenate(z_seqs, axis=0)
    #     r2_z = {}
    #     for z in range(self.num_states):
    #         z_mask = z_seqs == z
    #         r = r2_score(emissions[z_mask], y_preds[z_mask], multioutput='variance_weighted')
    #         r2_z[z] = r
    #     # print("r2_z", r2_z)
    #     return r2_z
    #
    # def score_by_z_soft(self, emissions, y_preds_per_state, z_probs):
    #     y_preds_per_state = np.concatenate(y_preds_per_state, axis=0)
    #     emissions = np.concatenate(emissions, axis=0)
    #     z_probs = np.concatenate(z_probs, axis=0)
    #     r = r2_score_custom(emissions, y_preds_per_state, z_probs)
    #     r2_z = dict(zip(range(self.num_states), r))
    #     # print("r2_z soft", r2_z)
    #     return r2_z

    def score_by_z_by_fly(self, emissions, y_preds, z_seqs):
        raise NotImplementedError
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

    # def score_by_z_and_o(self, emissions, y_preds, z_seqs):
    #     raise NotImplementedError
    #     y_preds = np.concatenate(y_preds, axis=0)
    #     emissions = np.concatenate(emissions, axis=0)
    #     z_seqs = np.concatenate(z_seqs, axis=0)
    #     r2_z_o = {}
    #     for z in range(self.num_states):
    #         r2_z_o[z] = {}
    #         z_mask = z_seqs == z
    #         for o in range(self.data_config["emission_dim"]):
    #             if np.sum(z_mask):
    #                 r2_z_o[z][o] = r2_score(emissions[z_mask][:, o], y_preds[z_mask][:, o])
    #             else:
    #                 print(f'nothing for state {z} for emission {o}')
    #     # print("r2_z_o", r2_z_o)
    #     return r2_z_o

    # def score_by_z_and_o_soft(self, emissions, y_preds_per_state, z_probs):
    #     y_preds_per_state = np.concatenate(y_preds_per_state, axis=0)
    #     emissions = np.concatenate(emissions, axis=0)
    #     z_probs = np.concatenate(z_probs, axis=0)
    #
    #     r2_z_o = {}
    #     for z in range(self.num_states):
    #         r2_z_o[z] = {}
    #
    #     for o in range(self.data_config["emission_dim"]):
    #         r = r2_score_custom(emissions[:, o], y_preds_per_state[..., o], z_probs)
    #         for z in range(self.num_states):
    #             r2_z_o[z][o] = r[z]
    #
    #     # print("r2_z_o soft", r2_z_o)
    #     return r2_z_o

    def score_by_z_and_o_by_fly(self, emissions, y_preds, z_seqs):
        raise NotImplementedError
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

    def pearson(self, emissions, y_preds):
        pearson_o = self.pearson_by_o(emissions, y_preds)
        return np.mean([pearson_o[o] for o in range(self.data_config["emission_dim"])])    # equivalent to multioutput="uniform_average" in r2_score
    #
    def pearson_by_fly(self, emissions, y_preds):
        pearson_o_fly = self.pearson_by_o_by_fly(emissions, y_preds)
        return np.mean(np.vstack([pearson_o_fly[o] for o in range(self.data_config["emission_dim"])]), axis=0)    # equivalent to multioutput="uniform_average" in r2_score
    #
    # def pearson_by_z(self, emissions, y_preds_per_state, z_probs):
    #     pearson_z_o = self.pearson_by_z_by_o(emissions, y_preds_per_state, z_probs)
    #     pearson_z = {}
    #     for z in pearson_z_o:
    #         # print("pearson_z_o[z]", pearson_z_o[z])
    #         pearson_z[z] = np.mean([pearson_z_o[z][o] for o in range(self.data_config["emission_dim"])])  # equivalent to multioutput="uniform_average" in r2_score
    #     # print("pearson_z", pearson_z)
    #     return pearson_z

    def pearson_by_z_by_fly(self, emissions, y_preds_per_state, z_probs):
        pearson_z_o_fly = self.pearson_by_z_and_o_by_fly(emissions, y_preds_per_state, z_probs)
        pearson_z_fly = {}
        for z in pearson_z_o_fly:
            # print("pearson_z_o_fly[z]", pearson_z_o_fly[z])
            pearson_z_fly[z] = np.mean(np.vstack([pearson_z_o_fly[z][o] for o in range(self.data_config["emission_dim"])]), axis=0)  # equivalent to multioutput="uniform_average" in r2_score
        # print("pearson_z_fly", pearson_z_fly)
        return pearson_z_fly

    def pearson_by_o(self, emissions, y_preds):
        y_preds = np.concatenate(y_preds, axis=0)
        emissions = np.concatenate(emissions, axis=0)
        pearson_o = {}

        for o in range(self.data_config["emission_dim"]):
            a = y_preds[:, o]
            b = emissions[:, o]
            pearson_o[o] = pearsonr(a, b)     # basically, zero lag correlation, ensures mean-centered and normalized
        return pearson_o

    def pearson_by_o_by_fly(self, emissions, y_preds):
        pearson_o = {}
        for o in range(self.data_config["emission_dim"]):
            pearson_o[o] = []
            for i in range(len(emissions)):
                a = y_preds[i][:, o]
                b = emissions[i][:, o]
                c = pearsonr(a, b)
                pearson_o[o].append(c[0])     # basically, zero lag correlation
            pearson_o[o] = np.array(pearson_o[o])
        return pearson_o

    # def pearson_by_z_by_o(self, emissions, y_preds_per_state, z_probs):
    #
    #     y_preds_per_state = np.concatenate(y_preds_per_state, axis=0)
    #     emissions = np.concatenate(emissions, axis=0)
    #     z_probs = np.concatenate(z_probs, axis=0)
    #
    #     pearson_z_o = {}
    #     for z in range(self.num_states):
    #         pearson_z_o[z] = {}
    #
    #     for o in range(self.data_config["emission_dim"]):
    #         pro = pearsonr_custom(emissions[:, o], y_preds_per_state[..., o], z_probs)  # pearson for emission o in each state
    #         for z in range(self.num_states):
    #             pearson_z_o[z][o] = pro[z]
    #     # print(pearson_z_o)
    #     return pearson_z_o

    def pearson_by_z_and_o_by_fly(self, emissions, y_preds_per_state, z_probs):

        pearson_z_o_by_fly = {}

        for z in range(self.num_states):
            pearson_z_o_by_fly[z] = {}
            for o in range(self.data_config["emission_dim"]):
                pearson_z_o_by_fly[z][o] = []

        for i in range(len(emissions)):
            for o in range(self.data_config["emission_dim"]):
                pro = pearsonr_custom(emissions[i][:, o], y_preds_per_state[i][..., o], z_probs[i])  # pearson for emission o in each state
                # print("flyi", i, "o", o, pro)
                for z in range(self.num_states):
                    pearson_z_o_by_fly[z][o].append(pro[z])

        for z in range(self.num_states):
            for o in range(self.data_config["emission_dim"]):
                pearson_z_o_by_fly[z][o] = np.array(pearson_z_o_by_fly[z][o])
        return pearson_z_o_by_fly

    # def correlation_max_by_o(self, emissions, y_preds):
    #     """To summarize: with the calculation done as above, a positive lag means the first series lags the second,
    #     or the second leads the first--peaks earlier in time, so at a location to the left on the time series plot.
    #     Source: https://currents.soest.hawaii.edu/ocn_data_analysis/_static/SEM_EDOF.html
    #
    #     +ve lags at the peak will mean "a" lags "b", or "b" leads "a". So, a=model lags b=truth, i.e. model prediction
    #     is delayed — it's lagging behind the behavior.
    #     """
    #     y_preds = np.concatenate(y_preds, axis=0)
    #     emissions = np.concatenate(emissions, axis=0)
    #     corrs_dict = {}
    #     lags_dict = {}
    #     for o in range(self.data_config["emission_dim"]):
    #         a = y_preds[:, o]       # a = model, predicted
    #         b = emissions[:, o]     # b = data, truth
    #         a = a - np.mean(a)  # ensure both are mean-centered
    #         b = b - np.mean(b)
    #         # print(np.mean(a), np.mean(b))
    #         c = scipy.signal.correlate(a, b, mode='full') / (np.linalg.norm(a) * np.linalg.norm(b))
    #         lags = np.arange(-len(a) + 1, len(a))
    #         corrs_dict[o] = np.max(c)
    #         lags_dict[o] = lags[np.argmax(c)]
    #     return corrs_dict, lags_dict

    def correlation_max_by_o_by_fly(self, emissions, y_preds):
        """To summarize: with the calculation done as above, a positive lag means the first series lags the second,
        or the second leads the first--peaks earlier in time, so at a location to the left on the time series plot.
        Source: https://currents.soest.hawaii.edu/ocn_data_analysis/_static/SEM_EDOF.html

        +ve lags at the peak will mean "a" lags "b", or "b" leads "a". So, a=model lags b=truth, i.e. model prediction
        is delayed — it's lagging behind the behavior.
        """
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
                # print(i, np.mean(a), np.mean(b))
                c = scipy.signal.correlate(a, b, mode='full') / (np.linalg.norm(a) * np.linalg.norm(b))
                lags = np.arange(-len(a) + 1, len(a))
                max_c, lag_for_max_c = np.max(c), lags[np.argmax(c)]
                if o == 0: print(o, max_c, lag_for_max_c)
                corrs_dict[o].append(max_c)
                lags_dict[o].append(lag_for_max_c)
            corrs_dict[o] = np.array(corrs_dict[o])
            lags_dict[o] = np.array(lags_dict[o])
        return corrs_dict, lags_dict
