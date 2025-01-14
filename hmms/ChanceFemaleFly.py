import joblib

import tensorflow_probability.substrates.jax.distributions as tfd
import jax
import numpy as np
import jax.numpy as jnp
from sklearn.metrics import r2_score

from utils import utils
from BaseFemaleFly import BaseFemaleFly

# print("jax.config", jax.config.values)
jax.config.update("jax_enable_x64", True)


class ChanceFemaleFly(BaseFemaleFly):

    prefix = 'chance'

    def __init__(self, data_config, model_config):
        self.data_config = data_config
        self.model_config = model_config
        self.num_states = 1
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


if __name__ == '__main__':

    data = joblib.load(f'../data/fly_data_cos=4_ortho_o=15.pkl')
    model_config = {}

    data_config, emissions, inputs = data['data_config'], data['emissions'], data['inputs']
    num_batches = data_config['num_sessions']
    num_train_batches = int(num_batches * 0.8)
    train_emissions, train_inputs = emissions[:num_train_batches], inputs[:num_train_batches]
    test_emissions, test_inputs = emissions[num_train_batches:], inputs[num_train_batches:]

    model = ChanceFemaleFly(data_config, model_config)
    model.fit(train_emissions, None)
    test_emission_predictions, test_z_predictions = model.predict(None, test_inputs)

    dump_filepath = utils.getafilepath(model.prefix)
    print(">> Saving at:", dump_filepath)
    utils.save(model, train_emissions, train_inputs, test_emissions, test_inputs, dump_filepath)

    print(model.score(train_emissions, train_inputs))

    print(model.score(test_emissions, test_inputs))
    print(model.score_by_z_and_o(test_emissions, test_inputs))
    print(model.correlation_by_o(test_emissions, test_inputs))

# 20241211_233249_designation fly_data_cos=4_ortho_o=15.pkl


