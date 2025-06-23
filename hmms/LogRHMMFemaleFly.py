from functools import partial

import jax
import numpy as np
from jax import vmap
import jax.random as jr
from jax.nn import sigmoid
from dynamax.hidden_markov_model import LogisticRegressionHMM
from hmms.BaseFemaleFly import BaseFemaleFly

from sklearn.metrics import accuracy_score
from utilities import fitting, utils

# print("jax.config", jax.config.values)
jax.config.update("jax_enable_x64", True)


class LogRHMMFemaleFly(BaseFemaleFly):

    prefix = 'logrhmm'

    def __init__(self, data_config, model_config):
        self.data_config = data_config
        self.model_config = model_config
        self.num_states = model_config['num_states']
        # print("self.model_config", self.model_config)
        self.seed = model_config.get('seed', 0)
        self.model = LogisticRegressionHMM(num_states=self.model_config['num_states'],
                                    input_dim=self.data_config['input_dim'],
                                    transition_matrix_stickiness=self.model_config['transition_matrix_stickiness'])
        self.learned_params = None
        self.learned_lps = None
        super().__init__()

    def fit(self, emissions, inputs, output_mn_std):
        print(f'Begin fitting {self.__class__.__name__}...')
        key = jr.PRNGKey(self.seed)
        em_params, em_lps = fitting.fitEM(key, self.model, emissions, train_inputs=inputs)
        self.learned_params = em_params
        # self.learned_params = self.reindex_params(em_params, emissions, inputs, output_mn_std)
        self.learned_lps = em_lps
        self.update_status()
        print(f'End fitting {self.__class__.__name__}...')
        return

    def check_nan_in_fit_params(self):
        return ~np.any(np.isnan(self.learned_params.transitions.transition_matrix))

    def predict(self, emissions, inputs):
        return self.predict_v4(emissions, inputs)

    def predict_v3(self, emissions, inputs):
        """Hard predictions"""

        def calc(params, i, z):
            return sigmoid(params.emissions.weights[z] @ i + params.emissions.biases[z])

        y_preds = []
        z_seqs = []
        for btch in range(len(emissions)):
            post = self.model.smoother(self.learned_params, emissions[btch].squeeze(-1), inputs[btch])
            z_seq = np.argmax(post.smoothed_probs, axis=1)

            y_pred = vmap(partial(calc, self.learned_params))(inputs[btch], z_seq)  # computed y given z
            y_preds.append(y_pred)
            z_seqs.append(z_seq)

        return y_preds, z_seqs

    def predict_v4(self, emissions, inputs):
        """Soft predictions"""

        W = self.learned_params.emissions.weights   # shape: (K, D, I)
        b = self.learned_params.emissions.biases    # shape: (K, D)
        K = self.num_states

        y_preds = []
        z_seqs = []
        preds_per_states = []
        for btch in range(len(emissions)):
            y_true = emissions[btch].squeeze(-1)    # shape: (T, D)
            x = inputs[btch]            # shape: (T, I)

            post = self.model.smoother(self.learned_params, y_true, x)
            gamma = post.smoothed_probs     # shape: (T, K)

            logits_per_state = np.stack([x @ W[k].T + b[k] for k in range(K)], axis=1)   # (T, K, D)
            preds_per_state = sigmoid(logits_per_state)  # (T, K, D)
            # print("gamma", gamma.shape, "preds_per_state", preds_per_state.shape)
            soft_predictions = np.sum(gamma[:, :] * preds_per_state, axis=1)[:, None]      # (T, D)
            # print("soft_predictions", soft_predictions)

            y_pred = (soft_predictions > 0.5).astype(int)
            z_seq = np.argmax(gamma, axis=1)    # shape: (T, 1)

            y_preds.append(y_pred)
            z_seqs.append(z_seq)
            preds_per_states.append(preds_per_state)
        return y_preds, z_seqs, preds_per_states

    def get_data_logprob(self, emissions, inputs=None):
        """Evaluate the log probability of the data under the given model and model parameters"""

        lps = [self.model.marginal_log_prob(self.learned_params, e.squeeze(-1), i) for e, i in zip(emissions, inputs)]
        lp = np.sum(lps)
        # print("lp", lp)
        lp_prior = self.model.log_prior(self.learned_params)
        # print("lp_prior", lp_prior)
        lp += lp_prior
        emissions_size = np.sum(e.size for e in emissions)
        lp = lp / emissions_size
        # print("lp", lp, "emissions_size", emissions_size)
        return lp

    def get_data_logprob_by_fly(self, emissions, inputs=None):
        """Evaluate the log probability of the data under the given model and model parameters, by fly."""
        lp_prior = self.model.log_prior(self.learned_params)
        # print("lp_prior", lp_prior)
        lps = np.array([(self.model.marginal_log_prob(self.learned_params, e.squeeze(-1), i) + lp_prior)/e.size for e, i in zip(emissions, inputs)])
        # print("lps by fly", lps)
        return lps

    def get_state_probs(self, emissions, inputs=None):
        z_probs = []
        for btch in range(len(emissions)):
            z_prob = self.model.smoother(self.learned_params, emissions[btch].squeeze(-1), inputs[btch])
            # print(z_prob.smoothed_probs.shape)
            z_probs.append(z_prob.smoothed_probs)
        # z_probs = np.array(z_probs)
        return z_probs

    def get_forward_state_probs(self, emissions, inputs=None):
        z_probs = []
        for btch in range(len(emissions)):
            z_prob = self.model.filter(self.learned_params, emissions[btch].squeeze(-1), inputs[btch])
            # print(z_prob.filtered_probs.shape)
            z_probs.append(z_prob.filtered_probs)
        # z_probs = np.array(z_probs)
        return z_probs

    def score(self, emissions, y_preds):
        y_preds = np.concatenate(y_preds, axis=0)
        emissions = np.concatenate(emissions, axis=0)
        a = accuracy_score(emissions, y_preds)
        print("accuracy_score", a)
        return a

    def scores_by_fly(self, emissions, y_preds):
        accuracy_scores = np.array([accuracy_score(emissions[i], y_preds[i]) for i in range(len(emissions))])
        print("accuracy_scores by fly", accuracy_scores)
        return accuracy_scores
