import jax
import numpy as np
import jax.random as jr
from hmms.LRHMMFemaleFly import LRHMMFemaleFly
from hmms.LRFemaleFly import LRFemaleFly

from utilities import fitting

jax.config.update("jax_enable_x64", True)


class LRHMMCustomInitFemaleFly(LRHMMFemaleFly):

    prefix = 'lrhmmci'

    def fit(self, emissions, inputs):
        key = jr.PRNGKey(self.seed)
        
        lr = LRFemaleFly(self.data_config, self.model_config)
        lr.fit(emissions, inputs)
        W = np.repeat(lr.learned_params['w'], repeats=self.num_states, axis=0)
        b = np.repeat(lr.learned_params['b'], repeats=self.num_states, axis=0)
        W = W + np.random.random(W.shape) * 0.01
        b = b + np.random.random(b.shape) * 0.01
        print("Initial W and b computed.")
        em_params, em_lps = fitting.fitEMCustomInit(key, self.model, emissions, train_inputs=inputs,
                                                    emission_weights=W, emission_biases=b)
        self.learned_params = em_params
        self.learned_params = self.reindex_params(em_params, emissions, inputs)
        self.learned_lps = em_lps
        self.update_status()
        return

# python run_single.py --mc '{"names": "lrhmmci", "seeds": 6205, "num_states": 2, "transition_matrix_stickiness": 10}'
