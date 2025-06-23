import tensorflow_probability.substrates.jax.distributions as tfd
import jax
import numpy as np
from jax import vmap
import jax.numpy as jnp

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from hmms.BaseFemaleFly import BaseFemaleFly

# print("jax.config", jax.config.values)
jax.config.update("jax_enable_x64", True)


class LogRFemaleFly(BaseFemaleFly):

    prefix = 'logr'

    def __init__(self, data_config, model_config):
        """
        model_config in Logistic Regression is unused.
        :param data_config:
        :param model_config:
        """
        self.data_config = data_config
        self.model_config = {}
        self.num_states = 1
        self.model_config['num_states'] = self.num_states
        self.model = LogisticRegression(fit_intercept=True, class_weight='balanced', max_iter=500)
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
        y_preds, _, z_seqs, y_preds_per_state = self.predict_prob(emissions, inputs)
        return y_preds, z_seqs, y_preds_per_state

    def predict_prob(self, emissions, inputs):
        """ emissions is unused """
        y_preds = []
        y_preds_probs = []
        z_seqs = []
        y_preds_per_state = []
        for i, _ in enumerate(inputs):
            y_preds_probs_ = self.model.predict_proba(_)
            y_preds_ = self.model.predict(_)[:, None]
            z_seqs_ = np.zeros(_.shape[0])
            y_preds_probs.append(y_preds_probs_)
            y_preds.append(y_preds_)
            y_preds_per_state.append(y_preds_)
            z_seqs.append(z_seqs_)
            print(i, emissions[i], emissions[i].shape)
            print(i, y_preds_, y_preds_.shape)
        a = self.score(emissions, y_preds)
        print("accuracy", a)
        return y_preds, y_preds_probs, z_seqs, y_preds_per_state

    def predict_v3(self, emissions, inputs):
        return self.predict(emissions, inputs)[:2]

    def get_state_probs(self, emissions, inputs=None):
        z_probs = [np.ones(_.shape[0]).reshape(-1, 1) for _ in emissions]
        return z_probs

    def get_forward_state_probs(self, emissions, inputs=None):
        return self.get_state_probs(emissions, inputs)

    def score(self, emissions, y_preds):
        y_preds = np.concatenate(y_preds, axis=0)
        emissions = np.concatenate(emissions, axis=0)
        print("y_preds emissions", y_preds.shape, emissions.shape)
        a = accuracy_score(emissions, y_preds)
        print("accuracy_score", a)
        return a

    def scores_by_fly(self, emissions, y_preds):
        accuracy_scores = np.array([accuracy_score(emissions[i], y_preds[i]) for i in range(len(emissions))])
        print("accuracy_scores by fly", accuracy_scores)
        return accuracy_scores

    def get_data_logprob(self, emissions, inputs):
        """
        Logistic regression P(Y|X, w)
        """
        # def fit_normal_residuals(fit_y, true_y):
        #     residuals = fit_y - true_y
        #     sigma = jnp.cov(residuals.T)
        #     mu = jnp.zeros(residuals.shape[-1])
        #     p = tfd.MultivariateNormalFullCovariance(loc=mu, covariance_matrix=sigma).prob(residuals)
        #     p = jnp.maximum(p, 1e-15)
        #     log_Y_given_wx = jnp.sum(jnp.log(p))
        #     return log_Y_given_wx

        def fit_bernoulli_loglik(fit_y, true_y):
            # eps = 1e-7  # for numerical stability
            # fit_y = jnp.clip(fit_y, eps, 1 - eps)
            log_probs = true_y * np.log(fit_y) + (1 - true_y) * np.log(1 - fit_y)
            return np.sum(log_probs)

        emissions_pred = self.predict_prob(emissions, inputs)[1]
        total_emissions_size = np.sum([len(_) for _ in emissions])
        lp = np.sum([fit_bernoulli_loglik(yp, yt) for yp, yt in zip(emissions_pred, emissions)]) / total_emissions_size
        print("lp", lp)
        return lp

    def get_data_logprob_by_fly(self, emissions, inputs):
        """
        Logistic regression P(Y|X, w), by fly
        """
        def fit_bernoulli_loglik(fit_y, true_y):
            # eps = 1e-7  # for numerical stability
            # fit_y = jnp.clip(fit_y, eps, 1 - eps)
            log_probs = true_y * np.log(fit_y) + (1 - true_y) * np.log(1 - fit_y)
            return np.sum(log_probs)

        emissions_pred = self.predict_prob(emissions, inputs)[1]
        lps = np.array([fit_bernoulli_loglik(yp, yt)/yt.size for yp, yt in zip(emissions_pred, emissions)])
        print("lps by fly", lps)
        return lps
