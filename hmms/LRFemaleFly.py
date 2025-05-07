import tensorflow_probability.substrates.jax.distributions as tfd
import jax
import numpy as np
from jax import vmap
import jax.numpy as jnp

from sklearn.linear_model import LinearRegression

from hmms.BaseFemaleFly import BaseFemaleFly

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
        self.model_config = {}
        self.num_states = 1
        self.model_config['num_states'] = self.num_states
        self.model = LinearRegression(fit_intercept=True)
        self.learned_params = None
        self.learned_lps = None
        super().__init__()

    def fit(self, emissions, inputs, output_mn_std=None):
        print(f'Begin fitting {self.__class__.__name__}...')
        X_tr = np.concatenate(inputs, axis=0)
        print(X_tr.shape)
        Y_tr = np.concatenate(emissions, axis=0)
        print(Y_tr.shape)
        self.model.fit(X_tr, Y_tr)
        self.learned_params = {
            'w': np.expand_dims(self.model.coef_, 0),
            'b': np.expand_dims(self.model.intercept_, 0)
        }
        self.update_status()
        print(f'End fitting {self.__class__.__name__}...')
        return

    def check_nan_in_fit_params(self):
        return ~np.any(np.isnan(self.learned_params['w']))

    def predict(self, emissions, inputs):
        """

        :param emissions: Unused
        :param inputs:
        :return:
        """
        # X_tr = inputs.reshape(-1, inputs.shape[-1])
        # y_preds = self.model.predict(X_tr)
        # z_seqs = np.zeros(y_preds.shape[0])
        #
        # y_preds = y_preds.reshape(inputs.shape[0], -1, self.data_config['emission_dim'])
        # z_seqs = z_seqs.reshape(inputs.shape[0], -1)
        #
        y_preds = []
        z_seqs = []
        for _ in inputs:
            y_preds_ = self.model.predict(_)
            z_seqs_ = np.zeros(_.shape[0])
            y_preds.append(y_preds_)
            z_seqs.append(z_seqs_)
        return y_preds, z_seqs

    def get_data_logprob(self, emissions, inputs):
        """
        Linear regression P(Y|X, w)
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
        # lp = vmap(fit_normal_residuals)(emissions_pred, emissions).sum() / emissions.size
        total_emissions_size = np.sum([len(_) for _ in emissions])
        lp = np.sum([fit_normal_residuals(yp, yt) for yp, yt in zip(emissions_pred, emissions)]) / total_emissions_size
        return lp

    def get_state_probs(self, emissions, inputs=None):
        z_probs = [np.ones(_.shape[0]) for _ in emissions]
        return z_probs

    def get_forward_state_probs(self, emissions, inputs=None):
        return self.get_state_probs(emissions, inputs)
