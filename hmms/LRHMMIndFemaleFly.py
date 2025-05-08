import jax
import numpy as np
import jax.random as jr
from hmms.LRHMMCustomInitFemaleFly import LRHMMCustomInitFemaleFly

from utilities import fitting

jax.config.update("jax_enable_x64", True)


class LRHMMIndFemaleFly(LRHMMCustomInitFemaleFly):
    prefix = 'lrhmmci2'

    def fit(self, global_params, emissions, inputs, output_mn_std=None):
        key = jr.PRNGKey(self.seed)
        # np.random.seed(seed)  # TODO: set seeds?
        W = global_params.emissions.weights
        b = global_params.emissions.biases
        W = W + np.random.random(W.shape) * 0.01
        b = b + np.random.random(b.shape) * 0.01
        em_params, em_lps = fitting.fitEMCustomInit(key, self.model, emissions, train_inputs=inputs,
                                                    emission_weights=W, emission_biases=b)
        self.learned_params = em_params
        self.learned_lps = em_lps
        return
