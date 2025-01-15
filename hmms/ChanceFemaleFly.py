import joblib

import tensorflow_probability.substrates.jax.distributions as tfd
import jax
import numpy as np
import jax.numpy as jnp
from sklearn.metrics import r2_score

from utils import utils
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
        self.model_config = {}
        self.num_states = 1
        self.model_config['num_states'] = self.num_states
        # self.model = tfd.MultivariateNormalFullCovariance(loc=np.zeros(data_config['emission_dim']), covariance_matrix=np.identity(data_config['emission_dim']))
        self.learned_params = None
        self.learned_lps = None
        super().__init__()

    def fit(self, emissions, inputs):

        y = emissions.reshape(-1, self.data_config['emission_dim'])
        print(emissions.shape, y.shape)
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
        """
        Multivariate gaussian model
        """
        # p = multivariate_normal.pdf(y, mean=mu, cov=cov)
        p = self.model.prob(emissions)
        p = jnp.maximum(p, 1e-15)
        log_Y_given_mvn = jnp.sum(jnp.log(p))
        lp = log_Y_given_mvn.sum() / emissions.size
        return lp

    def score(self, emissions, inputs):
        y_preds = self.predict(None, inputs)[0]
        y_preds = y_preds.reshape(-1, self.data_config['emission_dim'])
        y_tr = emissions.reshape(-1, self.data_config['emission_dim'])
        return round(r2_score(y_tr, y_preds), 4)

# 20241211_233249_designation fly_data_cos=4_ortho_o=15.pkl


