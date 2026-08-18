import time

import jax.lax
import jax.numpy as jnp

from utilities.io import get_chance_logprob, ENABLE_FLOAT64
from utilities import utils, fitting, io
from hmms.BaseFemaleFly import BaseFemaleFly
from hmms.ChanceFemaleFly import ChanceFemaleFly

import numpy as np
import jax.random as jr
from jax import vmap
from dynamax.hidden_markov_model import LinearAutoregressiveHMM

jax.config.update("jax_enable_x64", ENABLE_FLOAT64)


class ARHMMFemaleFly(BaseFemaleFly):
    prefix = 'arhmm'

    def __init__(self, data_config, model_config):
        self.data_config = data_config
        self.model_config = model_config
        self.num_states = model_config['num_states']
        self.seed = model_config.get('seed', 0)
        self.num_lags = model_config['num_lags']
        self.model = LinearAutoregressiveHMM(self.model_config['num_states'],
                                             self.data_config['emission_dim'],
                                             num_lags=self.model_config['num_lags'],
                                             transition_matrix_stickiness=self.model_config['transition_matrix_stickiness'])
        self.learned_params = None
        self.learned_lps = None
        self.chance_mu = None
        self.chance_cov = None
        super().__init__()

    def reindex_params(self, em_params, emissions, inputs, output_mn_std):
        """Reindex states by some metric"""
        print('reindexing params...')

        # Reindex by the activity index
        _, z_seqs, _, _, _ = self.predict(emissions, inputs)
        emissions_z = utils.get_emissions_by_state(emissions, z_seqs, self.num_states, output_mn_std)
        state_activity_index = [np.mean(np.sqrt(emissions_z[z][:, 0]**2 + emissions_z[z][:, 1]**2)) for z in emissions_z]
        new_index = np.argsort(state_activity_index)[::-1]
        print("new ordering:", new_index)

        params = em_params._replace(
            initial=em_params.initial._replace(
                probs=em_params.initial.probs[new_index]
            ),
            transitions=em_params.transitions._replace(
                transition_matrix=em_params.transitions.transition_matrix[new_index, :][:, new_index]
            ),
            emissions=em_params.emissions._replace(
                weights=em_params.emissions.weights[new_index],
                biases=em_params.emissions.biases[new_index],
                covs=em_params.emissions.covs[new_index],
            )
        )
        # print("After:", params)
        print('reindexed params.')
        return params

    def fit(self, batched_emissions, batched_inputs, batched_output_mn_std):
        chance = ChanceFemaleFly(self.data_config, self.model_config)
        chance.fit(batched_emissions, batched_inputs)
        chance_params = chance.learned_params
        self.chance_mu = chance_params['mu']
        self.chance_cov = chance_params['cov']
        print('chance fit.')
        print(f'Begin fitting {self.__class__.__name__}...')
        key = jr.PRNGKey(self.seed)

        # since the sessions are variable length, chunk the sessions.
        emissions_to_fit = io.chunk_data(batched_emissions, chunk_size=5000)
        print("emissions", emissions_to_fit.shape, "self.num_lags", self.num_lags)
        lagged_inputs = jnp.array([self.model.compute_inputs(e) for e in emissions_to_fit])
        print("lagged_inputs", len(lagged_inputs), lagged_inputs[0].shape)
        em_params, em_lps = fitting.fitEM(key, self.model, emissions_to_fit, train_inputs=lagged_inputs)
        self.learned_params = em_params
        self.learned_params = self.reindex_params(em_params, batched_emissions, batched_inputs, batched_output_mn_std)
        self.learned_lps = em_lps
        self.update_status()
        print(f'End fitting {self.__class__.__name__}... (SUCCESS={self.fit_success})')
        return

    def check_nan_in_fit_params(self):
        return ~np.any(np.isnan(self.learned_params.transitions.transition_matrix))

    def predict(self, emissions, inputs):
        return self.predict_v4(emissions, inputs)

    def predict_v4(self, emissions, inputs):
        """Soft predictions
        """

        W = self.learned_params.emissions.weights  # shape: (K, D, I)
        b = self.learned_params.emissions.biases  # shape: (K, D)
        K = self.model.num_states

        lagged_inputs = [self.model.compute_inputs(e) for e in emissions]

        y_preds = []
        z_seqs = []
        preds_per_states = []
        z_probs = []
        fwd_z_probs = []
        for btch in range(len(emissions)):
            y_true = emissions[btch]
            x = lagged_inputs[btch]
            post = self.model.smoother(self.learned_params, y_true, x)
            gamma = post.predicted_probs
            preds_per_state = np.stack([(x @ W[k].T + b[k]) for k in range(K)], axis=1)
            soft_predictions = np.sum(gamma[:, :, None] * preds_per_state, axis=1)  # (T, D)
            y_pred = soft_predictions
            z_seq = np.argmax(gamma, axis=1)  # shape: (T, 1)
            y_preds.append(y_pred)
            z_seqs.append(z_seq)
            preds_per_states.append(preds_per_state)
            z_probs.append(post.smoothed_probs)
            fwd_z_probs.append(post.predicted_probs)

        return y_preds, z_seqs, preds_per_states, z_probs, fwd_z_probs

    def get_data_logprob(self, emissions, inputs=None):
        """Evaluate the log probability of the data under the given model and model parameters"""
        lagged_inputs = [self.model.compute_inputs(e) for e in emissions]
        inputs = lagged_inputs
        lps = [self.model.marginal_log_prob(self.learned_params, e, i) for e, i in zip(emissions, inputs)]
        lp = np.sum(lps)
        total_emissions_size = np.sum([len(_) for _ in emissions])
        lp = lp / total_emissions_size
        chance_lp = get_chance_logprob(np.concatenate(emissions, axis=0), self.chance_mu,
                                       self.chance_cov) / total_emissions_size
        relative_lp = lp - chance_lp
        return relative_lp

    def get_data_logprob_by_fly(self, emissions, inputs=None):
        """Evaluate the log probability of the data under the given model and model parameters, by fly."""
        lagged_inputs = [self.model.compute_inputs(e) for e in emissions]
        inputs = lagged_inputs
        lps = np.array([(self.model.marginal_log_prob(self.learned_params, e, i))/len(e) for e, i in zip(emissions, inputs)])
        chance_lps = np.array([get_chance_logprob(yt, self.chance_mu, self.chance_cov)/len(yt) for yt in emissions])
        relative_lps = lps - chance_lps
        return relative_lps
