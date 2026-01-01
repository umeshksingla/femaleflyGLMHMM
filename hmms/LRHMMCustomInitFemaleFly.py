import jax
import numpy as np
import jax.random as jr
from hmms.LRHMMFemaleFly import LRHMMFemaleFly
from hmms.LRFemaleFly import LRFemaleFly

from utilities import fitting, io

jax.config.update("jax_enable_x64", True)


class LRHMMCustomInitFemaleFly(LRHMMFemaleFly):

    prefix = 'glm-hmm'

    def fit(self, batched_emissions, batched_inputs, batched_output_mn_std):
        print(f'Begin fitting {self.__class__.__name__}...')
        key = jr.PRNGKey(self.seed)

        # since the sessions are variable length, chunk the sessions.
        emissions_to_fit = io.chunk_data(batched_emissions, chunk_size=5000)
        inputs_to_fit = io.chunk_data(batched_inputs, chunk_size=5000)

        lr = LRFemaleFly(self.data_config, self.model_config)
        lr.fit(emissions_to_fit, inputs_to_fit)
        W = np.repeat(lr.learned_params['w'], repeats=self.num_states, axis=0)
        b = np.repeat(lr.learned_params['b'], repeats=self.num_states, axis=0)
        W = W + np.random.random(W.shape) * 1e-4
        b = b + np.random.random(b.shape) * 1e-4
        # print("W", W)
        print("LR global W and b computed.", W.shape, b.shape)
        em_params, em_lps = fitting.fitEMCustomInit(key, self.model, emissions_to_fit, train_inputs=inputs_to_fit,
                                                    emission_weights=W, emission_biases=b)
        self.learned_params = em_params
        self.learned_params = self.reindex_params(em_params, batched_emissions, batched_inputs, batched_output_mn_std)
        self.learned_lps = em_lps
        self.update_status()
        print(f'End fitting {self.__class__.__name__}...')
        return
