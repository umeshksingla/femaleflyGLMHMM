from functools import partial

import jax
import numpy as np
from jax import vmap
import jax.random as jr
from dynamax.hidden_markov_model import LinearRegressionHMM
from BaseFemaleFly import BaseFemaleFly

from utils import fitting

# print("jax.config", jax.config.values)
jax.config.update("jax_enable_x64", True)


class LRHMMFemaleFly(BaseFemaleFly):

    prefix = 'lrhmm'

    def __init__(self, data_config, model_config):
        self.data_config = data_config
        self.model_config = model_config
        self.num_states = model_config['num_states']
        print("self.model_config", self.model_config)
        self.seed = model_config.get('seed', 0)
        self.model = LinearRegressionHMM(num_states=self.model_config['num_states'],
                                    input_dim=self.data_config['input_dim'],
                                    emission_dim=self.data_config['emission_dim'],
                                    transition_matrix_stickiness=self.model_config['transition_matrix_stickiness'])
        self.learned_params = None
        self.learned_lps = None
        super().__init__()

    def fit(self, emissions, inputs):
        key = jr.PRNGKey(self.seed)
        em_params, em_lps = fitting.fitEM(key, self.model, emissions, train_inputs=inputs)
        self.learned_params = em_params
        self.learned_lps = em_lps
        self.update_status()
        return

    def check_nan_in_fit_params(self):
        print(self.learned_params.transitions.transition_matrix)
        return ~np.any(np.isnan(self.learned_params.transitions.transition_matrix))

    def predict(self, emissions, inputs):

        def calc(params, z, i):
            return params.emissions.weights[z] @ i + params.emissions.biases[z]

        y_preds = []
        z_seqs = []
        for btch in range(len(emissions)):
            z_seq = self.model.most_likely_states(self.learned_params, emissions[btch], inputs[btch])  # inferred states
            y_pred = vmap(partial(calc, self.learned_params))(z_seq, inputs[btch])  # inferred y given z
            y_preds.append(y_pred)
            z_seqs.append(z_seq)
            # print(btch, y_pred.shape)
        y_preds = np.array(y_preds)
        z_seqs = np.array(z_seqs)
        return y_preds, z_seqs

    def get_data_logprob(self, emissions, inputs=None):
        """Evaluate the log probability of the data under the given model and model parameters"""
        lp = vmap(partial(self.model.marginal_log_prob, self.learned_params))(emissions, inputs).sum()
        lp += self.model.log_prior(self.learned_params)
        lp = lp / emissions.size
        return lp


# 20241213_142846_lane with LR 3
# 20241213_151646_diction noLR 3
# 20241217_164917_toad with LR 5
# 20241217_164829_toothpaste noLR 5
# 20241217_170304_tasty with LR 3 directional only
# 20241217_175018_watch with LR 3 smoothed
# 20241217_175455_lecture with LR 3 heavily smoothed
# 20241219_190821_administration with LR cos=8


