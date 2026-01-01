import joblib

import tensorflow_probability.substrates.jax.distributions as tfd
import jax
import numpy as np
import jax.numpy as jnp
from sklearn.metrics import r2_score

from utilities import utils
from utilities.io import get_chance_logprob
from hmms.BaseFemaleFly import BaseFemaleFly

# print("jax.config", jax.config.values)
jax.config.update("jax_enable_x64", True)


class ChanceFemaleFly(BaseFemaleFly):

    prefix = 'chance'

    def __init__(self, data_config, model_config):
        """
        model_config in Chance Model is unused.
        :param data_config:
        :param model_config:
        """
        self.data_config = data_config
        self.model_config = model_config.copy()
        self.num_states = 0
        self.model_config['num_states'] = self.num_states
        # self.model = tfd.MultivariateNormalFullCovariance(loc=np.zeros(data_config['emission_dim']), covariance_matrix=np.identity(data_config['emission_dim']))
        self.learned_params = None
        self.learned_lps = None
        super().__init__()

    def fit(self, emissions, inputs, output_mn_std=None):
        y = np.concatenate(emissions, axis=0)
        print(y.shape)
        mu = jnp.mean(y, axis=0)
        cov = jnp.cov(y.T)
        print(mu, mu.shape)
        print(cov, cov.shape)
        self.model = tfd.MultivariateNormalFullCovariance(loc=mu, covariance_matrix=cov)
        self.learned_params = {'mu': mu, 'cov': cov}
        self.update_status()
        return

    def check_nan_in_fit_params(self):
        return (~np.any(np.isnan([self.learned_params['mu']]))) | (~np.any(np.isnan([self.learned_params['cov']])))

    def predict(self, emissions, inputs):
        """

        :param emissions: Unused
        :param inputs:
        :return:
        """
        X_tr = inputs.reshape(-1, inputs.shape[-1])
        y_preds = np.tile(self.learned_params['mu'], (X_tr.shape[0], 1))
        z_seqs = np.zeros(X_tr.shape[0])

        y_preds = y_preds.reshape(inputs.shape[0], -1, self.data_config['emission_dim'])
        z_seqs = z_seqs.reshape(inputs.shape[0], -1)
        return y_preds, z_seqs

    def get_data_logprob(self, emissions, inputs):
        total_emissions_size = np.sum([len(_) for _ in emissions])
        chance_lp = get_chance_logprob(np.concatenate(emissions, axis=0)) / total_emissions_size
        print("chance lp", chance_lp)
        return chance_lp

    def get_data_logprob_by_fly(self, emissions, inputs):
        chance_lps = np.array([get_chance_logprob(yt) / len(yt) for yt in emissions])
        print("chance lps by fly", chance_lps)
        return chance_lps

    def score(self, emissions, inputs):
        y_preds = self.predict(None, inputs)[0]
        y_preds = y_preds.reshape(-1, self.data_config['emission_dim'])
        y_tr = emissions.reshape(-1, self.data_config['emission_dim'])
        return round(r2_score(y_tr, y_preds), 4)

    def get_state_probs(self, emissions, inputs=None):
        z_probs = np.ones((emissions.shape[0], emissions.shape[1]))
        return z_probs

    def get_forward_state_probs(self, emissions, inputs=None):
        return self.get_state_probs(emissions, inputs)
