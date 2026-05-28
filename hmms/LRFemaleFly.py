"""
Docstring for hmms.LRFemaleFly. Supports input masking per output dimension.
"""

import sys

import scipy.linalg
import tensorflow_probability.substrates.jax.distributions as tfd
import jax
import numpy as np
from jax import vmap
import jax.numpy as jnp

from sklearn.linear_model import LinearRegression

from hmms.BaseFemaleFly import BaseFemaleFly
from hmms.ChanceFemaleFly import ChanceFemaleFly
from utilities.io import get_chance_logprob

# print("jax.config", jax.config.values)
jax.config.update("jax_enable_x64", True)


class LRFemaleFly(BaseFemaleFly):

    prefix = 'lr'

    def __init__(self, data_config, model_config):
        """
        model_config in Linear Regression is unused.
        :param data_config:
        :param model_config:
        """
        self.data_config = data_config
        self.model_config = model_config.copy()
        self.num_states = 1
        self.model_config['num_states'] = self.num_states
        self.learned_params = None
        self.learned_lps = None
        self.emission_dim = len(self.data_config['emission_labels'])
        self.input_mask_by_emission = self.data_config['input_mask_by_emission'].astype(int)
        self.model = [LinearRegression(fit_intercept=True) for o in range(self.emission_dim)]
        self.chance_mu = None
        self.chance_cov = None
        super().__init__()

    def fit(self, emissions, inputs, output_mn_std=None):
        print(f'Begin fitting chance...')
        chance = ChanceFemaleFly(self.data_config, self.model_config)
        chance.fit(emissions, inputs)
        chance_params = chance.learned_params
        self.chance_mu = chance_params['mu']
        self.chance_cov = chance_params['cov']
        print('chance fit.')
        print(f'Begin fitting {self.__class__.__name__}...')
        X_tr = np.concatenate(inputs, axis=0)
        # print(X_tr.shape)
        Y_tr = np.concatenate(emissions, axis=0)
        # print(Y_tr.shape)
        w = []
        b = []
        for o in range(self.emission_dim):
            # print(o, self.input_mask_by_emission[o], X_tr.shape, Y_tr.shape)
            mask = self.input_mask_by_emission[o] == 1
            # if o == 0: r = np.r_[0:28]
            # if o == 1: r = np.r_[28:28+24]
            # if o == 2: r = np.r_[28+24:]
            # print(o, mask, mask.sum(), np.where(mask)[0])
            x = X_tr[:, mask]
            y = Y_tr[:, o]
            # print(o, "X_tr", X_tr[0], X_tr[0].shape)
            # print(o, "x", x[0], x.shape, y.shape)
            m = self.model[o]
            m.fit(x, y)
            w_ = m.coef_
            b_ = m.intercept_
            # print(o, "w_", w_, w_.shape, b_)
            w.append(w_)
            b.append(b_)

        w = scipy.linalg.block_diag(*w)
        # print(w, w.shape)
        self.learned_params = {
            'w': np.expand_dims(w, 0),
            'b': np.expand_dims(b, 0)
        }
        # print(self.learned_params['w'].shape, self.learned_params['b'].shape)
        self.update_status()
        print(f'End fitting {self.__class__.__name__}...')
        # sys.exit(0)
        return

    def check_nan_in_fit_params(self):
        return ~np.any(np.isnan(self.learned_params['w']))

    def predict(self, emissions, inputs):
        """ emissions is unused """
        y_preds = []
        z_seqs = []
        y_preds_per_state = []
        z_probs = []
        fwd_z_probs = []
        for _ in inputs:
            y_preds_ = []
            for o in range(self.emission_dim):
                mask = self.input_mask_by_emission[o] == 1
                y_preds_o = self.model[o].predict(_[:, mask])
                y_preds_.append(y_preds_o)
            y_preds_ = np.array(y_preds_).T
            # print(y_preds_.shape)
            y_preds.append(y_preds_)
            y_preds_per_state.append(y_preds_[:, None])
            z_seqs_ = np.zeros(_.shape[0])
            z_seqs.append(z_seqs_)
        z_probs = [np.ones(_.shape[0]).reshape(-1, 1) for _ in inputs]
        fwd_z_probs = [np.ones(_.shape[0]).reshape(-1, 1) for _ in inputs]
        return y_preds, z_seqs, y_preds_per_state, z_probs, fwd_z_probs

    def get_data_logprob_old(self, emissions, inputs):
        """
        OUTDATED: Linear regression P(Y|X, w), relative to chance.
        """
        def fit_normal_residuals(fit_y, true_y):
            residuals = fit_y - true_y
            sigma = jnp.cov(residuals.T)
            mu = jnp.zeros(residuals.shape[-1])
            p = tfd.MultivariateNormalFullCovariance(loc=mu, covariance_matrix=sigma).log_prob(residuals)
            log_Y_given_wx = jnp.sum(p)
            return log_Y_given_wx

        emissions_pred = self.predict(None, inputs)[0]
        total_emissions_size = np.sum([len(_) for _ in emissions])
        lp = fit_normal_residuals(np.concatenate(emissions_pred, axis=0), np.concatenate(emissions, axis=0)) / total_emissions_size
        # print("LR lp calc tog", lp)
        chance_lp = get_chance_logprob(np.concatenate(emissions, axis=0), self.chance_mu, self.chance_cov) / total_emissions_size
        relative_lp = lp - chance_lp
        # print("chance_lp", chance_lp)
        # print("relative_lp", relative_lp)
        return relative_lp

    def get_data_logprob(self, emissions, inputs):
        def calc(y_pred, y_true):
            """
            Compute frequentist log-likelihood of linear regression model.
            Assumes: Gaussian noise, independent across output dimensions.
            Parameters:
                y_true: (N, D) observed
                y_pred: (N, D) predicted
            Returns:
                log_likelihood: float (total over N and D)
            """
            residuals = y_true - y_pred
            N = len(residuals)
            var = np.var(residuals, axis=0, ddof=1)  # (D,) # Estimate variance per dimension (ddof=1 for unbiased)

            # Avoid log(0). ideally the variance should be close to 1 as each session is zscored
            # (separately, that's why var not 1 but close to 1)
            var = np.maximum(var, 1e-15)
            log_likelihood_c = -0.5 * np.sum(N * np.log(2 * np.pi * var) - np.sum((residuals ** 2) / var, axis=0))
            return log_likelihood_c

        emissions_pred = self.predict(None, inputs)[0]
        total_emissions_size = np.sum([len(_) for _ in emissions])
        lp = calc(np.concatenate(emissions_pred, axis=0), np.concatenate(emissions, axis=0)) / total_emissions_size
        # print("lp", lp)
        chance_lp = get_chance_logprob(np.concatenate(emissions, axis=0), self.chance_mu, self.chance_cov) / total_emissions_size
        # print("chance_lp", chance_lp)
        relative_lp = lp - chance_lp
        # print("relative_lp", relative_lp)
        return relative_lp

    def get_data_logprob_by_fly_old(self, emissions, inputs):
        """
        OUTDATED: Linear regression P(Y|X, w), by fly
        """
        def fit_normal_residuals(fit_y, true_y):
            residuals = fit_y - true_y
            sigma = jnp.cov(residuals.T)
            mu = jnp.zeros(residuals.shape[-1])
            p = tfd.MultivariateNormalFullCovariance(loc=mu, covariance_matrix=sigma).prob(residuals)
            p = jnp.maximum(p, 1e-15)
            log_Y_given_wx = jnp.sum(jnp.log(p))
            return log_Y_given_wx

        emissions_pred = self.predict(None, inputs)[0]
        lps = np.array([fit_normal_residuals(yp, yt)/len(yt) for yp, yt in zip(emissions_pred, emissions)])
        # print("LR lps by fly", lps)

        chance_lps = np.array([get_chance_logprob(yt)/len(yt) for yt in emissions]) # chance model per fly. "How much better does my model predict behavior than a naive, non-informative model — for this specific session?"
        # print("chance_lps", chance_lps)
        return lps - chance_lps

    def get_data_logprob_by_fly(self, emissions, inputs):
        """
        Linear regression P(Y|X, w), by fly
        """

        def calc(y_pred, y_true):
            residuals = y_true - y_pred
            N = len(residuals)
            var = np.var(residuals, axis=0, ddof=1)  # (D,)
            var = np.maximum(var, 1e-15)
            log_likelihood_c = -0.5 * np.sum(N * np.log(2 * np.pi * var) - np.sum((residuals ** 2) / var, axis=0))
            return log_likelihood_c

        emissions_pred = self.predict(None, inputs)[0]
        lps = np.array([calc(yp, yt)/len(yt) for yp, yt in zip(emissions_pred, emissions)])
        # print("LR lps by fly", lps)

        # chance model per fly. "How much better does my model predict behavior
        # than a naive, non-informative model — for this specific session?"
        chance_lps = np.array([get_chance_logprob(yt, self.chance_mu, self.chance_cov)/len(yt) for yt in emissions])
        # print("chance_lps", chance_lps)
        return lps - chance_lps
