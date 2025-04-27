from functools import partial

import jax
import numpy as np
from jax import vmap
import jax.random as jr
from dynamax.hidden_markov_model import LinearRegressionHMM
from hmms.BaseFemaleFly import BaseFemaleFly

from utilities import fitting, utils

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

    def reindex_params(self, em_params, emissions, inputs, output_mn_std):
        """Reindex states by some metric"""

        # print("Before:", em_params)
        #
        # # Reindex by steady state frequency
        # state_occ = utils.calculate_steady_state_p(em_params.transitions.transition_matrix)  # Reindex by steady state frequency
        # new_index = np.argsort(state_occ)[::-1]
        # print("state_occ", state_occ)
        # print("new ordering:", new_index)

        # # OR Reindex by the forward velocity mean
        # y, z = self.predict(emissions, inputs)
        # emissions_z = utils.get_emissions_by_state(y, z, self.num_states)
        # state_vel_means = [np.mean(emissions_z[z][:, 0]) for z in emissions_z]
        # new_index = np.argsort(state_vel_means)[::-1]
        # print("state_vel_means", state_vel_means)
        # print("new ordering:", new_index)

        # OR Reindex by the activity index
        _, zseq = self.predict(emissions, inputs)
        emissions_z = utils.get_emissions_by_state(emissions, zseq, output_mn_std, self.num_states)
        state_activity_index = [np.mean(np.sqrt(emissions_z[z][:, 0]**2 + emissions_z[z][:, 1]**2)) for z in emissions_z]
        new_index = np.argsort(state_activity_index)[::-1]
        print("state_tot_activity_mean", state_activity_index)
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
        return params

    def fit(self, emissions, inputs, output_mn_std=None):
        key = jr.PRNGKey(self.seed)
        em_params, em_lps = fitting.fitEM(key, self.model, emissions, train_inputs=inputs)
        self.learned_params = em_params
        self.learned_params = self.reindex_params(em_params, emissions, inputs, output_mn_std)
        self.learned_lps = em_lps
        self.update_status()
        return

    def check_nan_in_fit_params(self):
        return ~np.any(np.isnan(self.learned_params.transitions.transition_matrix))

    def predict(self, emissions, inputs):
        return self.predict_v3(emissions, inputs)

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

    def predict_v2(self, emissions, inputs):

        def calc(params, z, i):
            return params.emissions.weights[z] @ i + params.emissions.biases[z]

        y_preds = []
        z_seqs = []
        for btch in range(len(emissions)):
            post = self.model.filter(self.learned_params, emissions[btch], inputs[btch])
            z_seq = np.argmax(post.filtered_probs, axis=1)  # inferred states
            # print("btch", btch, post.filtered_probs.shape, "z_seq", z_seq.shape)
            y_pred = vmap(partial(calc, self.learned_params))(z_seq, inputs[btch])  # computed y given z
            y_preds.append(y_pred)
            z_seqs.append(z_seq)
            # print(btch, y_pred.shape)
        y_preds = np.array(y_preds)
        z_seqs = np.array(z_seqs)
        return y_preds, z_seqs

    def predict_v3(self, emissions, inputs):

        def calc(params, pz, i):
            return np.sum(pz[z] * (params.emissions.weights[z] @ i + params.emissions.biases[z]) for z in np.arange(len(pz)))

        y_preds = []
        z_seqs = []
        for btch in range(len(emissions)):
            post = self.model.filter(self.learned_params, emissions[btch], inputs[btch])
            y_pred = vmap(partial(calc, self.learned_params))(post.predicted_probs, inputs[btch])  # computed y given z
            y_preds.append(y_pred)

            post = self.model.smoother(self.learned_params, emissions[btch], inputs[btch])
            z_seq = np.argmax(post.smoothed_probs, axis=1)
            z_seqs.append(z_seq)

        y_preds = np.array(y_preds)
        z_seqs = np.array(z_seqs)
        return y_preds, z_seqs

    def get_data_logprob(self, emissions, inputs=None):
        """Evaluate the log probability of the data under the given model and model parameters"""
        lp = vmap(partial(self.model.marginal_log_prob, self.learned_params))(emissions, inputs).sum()
        lp += self.model.log_prior(self.learned_params)
        lp = lp / emissions.size
        return lp

    def get_state_probs(self, emissions, inputs=None):
        z_probs = []
        for btch in range(len(emissions)):
            z_prob = self.model.smoother(self.learned_params, emissions[btch], inputs[btch])
            # print(z_prob.smoothed_probs.shape)
            z_probs.append(z_prob.smoothed_probs)
        z_probs = np.array(z_probs)
        return z_probs

    def get_forward_state_probs(self, emissions, inputs=None):
        z_probs = []
        for btch in range(len(emissions)):
            z_prob = self.model.filter(self.learned_params, emissions[btch], inputs[btch])
            # print(z_prob.filtered_probs.shape)
            z_probs.append(z_prob.filtered_probs)
        z_probs = np.array(z_probs)
        return z_probs
