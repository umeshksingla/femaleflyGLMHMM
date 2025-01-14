from functools import partial

import joblib

import jax
import numpy as np
from jax import vmap
import jax.random as jr
from dynamax.hidden_markov_model import GaussianHMM

from utils import utils, fitting
from BaseFemaleFly import BaseFemaleFly

# print("jax.config", jax.config.values)
jax.config.update("jax_enable_x64", True)


class GHMMFemaleFly(BaseFemaleFly):

    prefix = 'ghmm'

    def __init__(self, data_config, model_config):
        self.data_config = data_config
        self.model_config = model_config
        self.num_states = model_config['num_states']
        self.seed = model_config.get('seed', 0)
        self.model = GaussianHMM(num_states=self.model_config['num_states'],
                                    emission_dim=self.data_config['emission_dim'],
                                    transition_matrix_stickiness=self.model_config['transition_matrix_stickiness'])
        self.learned_params = None
        self.learned_lps = None
        super().__init__()

    def fit(self, emissions, inputs):
        key = jr.PRNGKey(self.seed)
        em_params, em_lps = fitting.fitEM(key, self.model, emissions, train_inputs=None)
        self.learned_params = em_params
        self.learned_lps = em_lps
        self.update_status()
        return

    def check_nan_in_fit_params(self):
        return ~np.any(np.isnan(self.learned_params.transitions.transition_matrix))

    def predict(self, emissions, inputs):

        def calc(params, z):
            return params.emissions.means[z]

        y_preds = []
        z_seqs = []
        for btch in range(len(emissions)):
            z_seq = self.model.most_likely_states(self.learned_params, emissions[btch], None)  # inferred states
            y_pred = vmap(partial(calc, self.learned_params))(z_seq)  # inferred y given z
            y_preds.append(y_pred)
            z_seqs.append(z_seq)
        y_preds = np.array(y_preds)
        z_seqs = np.array(z_seqs)
        return y_preds, z_seqs

    def get_data_logprob(self, emissions, inputs=None):
        """Evaluate the log probability of the data under the given model and model parameters"""
        lp = vmap(partial(self.model.marginal_log_prob, self.learned_params))(emissions, None).sum()
        lp += self.model.log_prior(self.learned_params)
        lp = lp / emissions.size
        return lp


if __name__ == '__main__':

    data = joblib.load(f'data/fly_data_cos=4_ortho_o=15_directional.pkl')
    model_config = {
        'num_states': 3,
        'transition_matrix_stickiness': 10,
    }

    data_config, emissions, inputs = data['data_config'], data['emissions'], data['inputs']
    num_batches = data_config['num_sessions']
    num_train_batches = int(num_batches * 0.8)
    train_emissions, train_inputs = emissions[:num_train_batches], inputs[:num_train_batches]
    test_emissions, test_inputs = emissions[num_train_batches:], inputs[num_train_batches:]

    model = GHMMFemaleFly(data_config, model_config)
    model.fit(train_emissions, None)
    # test_emission_predictions, test_z_predictions = model.predict(test_emissions, test_inputs)

    dump_filepath = utils.getafilepath(f'{model.prefix}_{model.model_config["num_states"]}')
    print(">> Saving at:", dump_filepath)
    utils.save(model, train_emissions, train_inputs, test_emissions, test_inputs, dump_filepath)

    print(model.score(train_emissions, train_inputs))

    print(model.score(test_emissions, test_inputs))
    print(model.score_by_z_and_o(test_emissions, test_inputs))
    print(model.correlation_by_o(test_emissions, test_inputs))


# 20241213_142835_bowtie with LR 3
# 20241213_151732_webmail noLR 3
# 20241217_170426_nonsense with LR 3 directional only