from functools import partial
import joblib

import jax
import numpy as np
from jax import vmap
import jax.random as jr
from dynamax.hidden_markov_model import GaussianHMM

from utilities.io import get_chance_logprob
from utilities import utils, fitting, io
from hmms.BaseFemaleFly import BaseFemaleFly
from hmms.ChanceFemaleFly import ChanceFemaleFly

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
        self.chance_mu = None
        self.chance_cov = None
        super().__init__()

    def reindex_params(self, em_params, emissions, inputs, output_mn_std):
        """Reindex states by some metric"""

        # print("Before:", em_params)

        # OR Reindex by the activity index
        _, z_seqs, _, _, _ = self.predict(emissions, inputs)
        emissions_z = utils.get_emissions_by_state(emissions, z_seqs, self.num_states, output_mn_std)
        state_activity_index = [np.mean(np.sqrt(emissions_z[z][:, 0]**2 + emissions_z[z][:, 1]**2)) for z in emissions_z]
        new_index = np.argsort(state_activity_index)[::-1]
        # print("state_tot_activity_mean", state_activity_index)
        print("new ordering:", new_index)

        params = em_params._replace(
            initial=em_params.initial._replace(
                probs=em_params.initial.probs[new_index]
            ),
            transitions=em_params.transitions._replace(
                transition_matrix=em_params.transitions.transition_matrix[new_index, :][:, new_index]
            ),
            emissions=em_params.emissions._replace(
                means=em_params.emissions.means[new_index],
                covs=em_params.emissions.covs[new_index],
            )
        )
        # print("After:", params)
        return params

    def fit(self, batched_emissions, batched_inputs, batched_output_mn_std):
        print(f'Begin fitting chance...')
        chance_params = ChanceFemaleFly(self.data_config, self.model_config).fit(batched_emissions, batched_inputs)
        self.chance_mu = chance_params['mu']
        self.chance_cov = chance_params['cov']
        print('chance fit.')
        print(f'Begin fitting {self.__class__.__name__}...')
        key = jr.PRNGKey(self.seed)

        # since the sessions are variable length, chunk the sessions.
        emissions_to_fit = io.chunk_data(batched_emissions, chunk_size=5000)

        em_params, em_lps = fitting.fitEM(key, self.model, emissions_to_fit, train_inputs=None)
        self.learned_params = em_params
        self.learned_params = self.reindex_params(em_params, batched_emissions, batched_inputs, batched_output_mn_std)
        self.learned_lps = em_lps
        self.update_status()
        print(f'End fitting {self.__class__.__name__}...')
        return

    def check_nan_in_fit_params(self):
        return ~np.any(np.isnan(self.learned_params.transitions.transition_matrix))

    def predict(self, emissions, inputs):
        return self.predict_v4(emissions, inputs)

    def predict_v4(self, emissions, inputs):
        """Soft predictions"""

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
        # print("lp", lp)
        # lp_prior = self.model.log_prior(self.learned_params)
        # print("lp_prior", lp_prior)
        # lp += lp_prior
        total_emissions_size = np.sum([len(_) for _ in emissions])
        lp = lp / total_emissions_size
        # print("lp", lp, "emissions_size", total_emissions_size)
        chance_lp = get_chance_logprob(np.concatenate(emissions, axis=0), self.chance_mu, self.chance_cov)/total_emissions_size
        relative_lp = lp - chance_lp
        # print("chance_lp", chance_lp)
        return relative_lp

    def get_data_logprob_by_fly(self, emissions, inputs=None):
        """Evaluate the log probability of the data under the given model and model parameters, by fly."""
        lp_prior = self.model.log_prior(self.learned_params)
        # print("lp_prior", lp_prior)
        lps = np.array([(self.model.marginal_log_prob(self.learned_params, e, None) + lp_prior)/len(e) for e, _ in zip(emissions, inputs)])
        # print("lps", lps)
        chance_lps = np.array([get_chance_logprob(yt, self.chance_mu, self.chance_cov)/len(yt) for yt in emissions])
        # print("chance_lps", chance_lps)
        relative_lps = lps - chance_lps
        return relative_lps

