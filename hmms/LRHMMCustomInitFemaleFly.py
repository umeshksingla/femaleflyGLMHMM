import jax
import numpy as np
import jax.random as jr
from hmms.LRHMMFemaleFly import LRHMMFemaleFly
from hmms.LRFemaleFly import LRFemaleFly

from utilities import fitting

jax.config.update("jax_enable_x64", True)


class LRHMMCustomInitFemaleFly(LRHMMFemaleFly):

    prefix = 'glm-hmm'

    def fit(self, emissions, inputs, output_mn_std):
        print(f'Begin fitting {self.__class__.__name__}...')
        key = jr.PRNGKey(self.seed)

        lr = LRFemaleFly(self.data_config, self.model_config)
        lr.fit(emissions, inputs, output_mn_std)
        W = np.repeat(lr.learned_params['w'], repeats=self.num_states, axis=0)
        b = np.repeat(lr.learned_params['b'], repeats=self.num_states, axis=0)
        W = W + np.random.random(W.shape) * 0.01
        b = b + np.random.random(b.shape) * 0.01
        print("LR global W and b computed.", W.shape, b.shape)
        em_params, em_lps = fitting.fitEMCustomInit(key, self.model, emissions, train_inputs=inputs,
                                                    emission_weights=W, emission_biases=b)
        self.learned_params = em_params
        self.learned_params = self.reindex_params(em_params, emissions, inputs, output_mn_std)
        self.learned_lps = em_lps
        self.update_status()
        print(f'End fitting {self.__class__.__name__}...')
        return
