from functools import partial
import joblib

import jax
import numpy as np
from jax import vmap
import jax.random as jr
import jax.numpy as jnp
from dynamax.hidden_markov_model import DiagonalGaussianHMM

from utilities.io import get_chance_logprob, ENABLE_FLOAT64, get_state_activity_index
from utilities import utils, fitting, io
from hmms.ChanceFemaleFly import ChanceFemaleFly
from hmms.BaseFemaleFly import BaseFemaleFly

jax.config.update("jax_enable_x64", ENABLE_FLOAT64)


class DiagGHMMFemaleFly(BaseFemaleFly):

    prefix = 'dghmm'

    def __init__(self, data_config, model_config):
        self.data_config = data_config
        self.model_config = model_config
        self.num_states = model_config['num_states']
        self.seed = model_config.get('seed', 0)
        self.init_method = model_config['init_method']
        self.model = DiagonalGaussianHMM(num_states=self.model_config['num_states'],
                                    emission_dim=self.data_config['emission_dim'],
                                    transition_matrix_stickiness=self.model_config['transition_matrix_stickiness'])
        self.learned_params = None
        self.learned_lps = None
        self.chance_mu = None
        self.chance_cov = None
        super().__init__()

    def reindex_params(self, em_params, emissions, inputs, output_mn_std):
        """Reindex states by some metric"""

        print('reindexing params...')
        new_index = get_state_activity_index(self, emissions, inputs, output_mn_std)

        params = em_params._replace(
            initial=em_params.initial._replace(
                probs=em_params.initial.probs[new_index]
            ),
            transitions=em_params.transitions._replace(
                transition_matrix=em_params.transitions.transition_matrix[new_index, :][:, new_index]
            ),
            emissions=em_params.emissions._replace(
                means=em_params.emissions.means[new_index],
                scale_diags=em_params.emissions.scale_diags[new_index],
            )
        )
        print('reindexed params.')
        return params

    def fit(self, batched_emissions, batched_inputs, batched_output_mn_std, do_reindex_params=True):
        print(f'Begin fitting {self.__class__.__name__}...')
        print(f'Begin fitting chance...')
        chance = ChanceFemaleFly(self.data_config, self.model_config)   # This will give us same Chance flies for get_data_logprob on train/heldout later
        chance.fit(batched_emissions, batched_inputs)
        chance_params = chance.learned_params
        self.chance_mu = chance_params['mu']
        self.chance_cov = chance_params['cov']
        print('chance fit.')
        key = jr.PRNGKey(self.seed)

        # since the sessions are variable length, chunk the sessions.
        emissions_to_fit = io.chunk_data(batched_emissions, chunk_size=5000)

        flat = np.asarray(emissions_to_fit).reshape(-1, np.shape(emissions_to_fit)[-1])
        if self.init_method == 'kmeans':
            params, props = self.model.initialize(key, method='kmeans', emissions=emissions_to_fit)
        else:
            # Use some frames as init points, as "prior" inits from too far and the second step of EM ends up too identical
            rng = np.random.default_rng(self.seed)
            means = flat[rng.choice(len(flat), size=self.num_states, replace=False)]
            scales = np.tile(flat.std(axis=0), (self.num_states, 1))
            params, props = self.model.initialize(key, emission_means=jnp.asarray(means), emission_scale_diags=jnp.asarray(scales))

        em_params, em_lps = fitting.fitDiagEM(self.model, params, props, emissions_to_fit, train_inputs=None)
        self.learned_params = em_params
        if do_reindex_params:
            self.learned_params = self.reindex_params(em_params, batched_emissions, batched_inputs, batched_output_mn_std)
        self.learned_lps = em_lps
        print(f'DGHMM training lps: start {float(em_lps[0]):.4f}, end {float(em_lps[-1]):.4f}')
        self.update_status()
        print(f'End fitting {self.__class__.__name__}...')
        return

    def check_nan_in_fit_params(self):
        return ~np.any(np.isnan(self.learned_params.transitions.transition_matrix))

    def predict(self, emissions, inputs, full=False):
        return self.predict_v4(emissions, inputs, full=full)

    def predict_v4(self, emissions, inputs, full=False):
        """Soft predictions. if full is True, use smoothed probs (to depict model)."""

        mu = self.learned_params.emissions.means   # shape: (K, D)
        mu = mu[:, :, None]     # shape: (K, D, I)
        # print("mu", mu.shape)
        K = self.num_states

        y_preds = []
        z_seqs = []
        preds_per_states = []
        z_probs = []
        fwd_z_probs = []
        for btch in range(len(emissions)):
            y_true = emissions[btch]    # shape: (T, D)
            x = np.ones((inputs[btch].shape[0], 1))

            post = self.model.smoother(self.learned_params, y_true, None)
            gamma = post.predicted_probs     # shape: (T, K)
            # print("x", x.shape, "mu[k]", mu[0].shape)
            # print("gamma", gamma.shape)
            if full:
                gamma = post.smoothed_probs
            preds_per_state = np.stack([x @ mu[k].T for k in range(K)], axis=1)   # (T, K, D)
            # print(preds_per_state)
            # print("preds_per_state", preds_per_state.shape)
            soft_predictions = np.sum(gamma[:, :, None] * preds_per_state, axis=1)      # (T, D)
            # print("soft_predictions", soft_predictions.shape)

            y_pred = soft_predictions
            z_seq = np.argmax(gamma, axis=1)    # shape: (T, 1)

            y_preds.append(y_pred)
            z_seqs.append(z_seq)
            preds_per_states.append(preds_per_state)
            z_probs.append(post.smoothed_probs)
            fwd_z_probs.append(post.predicted_probs)
            # print("nan?", btch, np.sum(np.isnan(y_pred)), np.sum(np.isnan(preds_per_state)))
        return y_preds, z_seqs, preds_per_states, z_probs, fwd_z_probs

    def get_data_logprob(self, emissions, inputs=None):
        """Evaluate the log probability of the data under the given model and model parameters"""

        lps = [self.model.marginal_log_prob(self.learned_params, e, None) for e, _ in zip(emissions, inputs)]
        # print("lps", lps)
        lp = np.sum(lps)
        total_emissions_size = np.sum([len(_) for _ in emissions])
        lp = lp / total_emissions_size
        # print("lp", lp, "emissions_size", total_emissions_size)
        chance_lp = get_chance_logprob(np.concatenate(emissions, axis=0), self.chance_mu, self.chance_cov)/total_emissions_size
        relative_lp = lp - chance_lp
        # print("chance_lp", chance_lp)
        return relative_lp

    def get_data_logprob_by_fly(self, emissions, inputs=None):
        """Evaluate the log probability of the data under the given model and model parameters, by fly."""
        lps = np.array([(self.model.marginal_log_prob(self.learned_params, e, None))/len(e) for e, _ in zip(emissions, inputs)])
        # print("lps", lps)
        chance_lps = np.array([get_chance_logprob(yt, self.chance_mu, self.chance_cov)/len(yt) for yt in emissions])
        # print("chance_lps", chance_lps)
        relative_lps = lps - chance_lps
        return relative_lps
