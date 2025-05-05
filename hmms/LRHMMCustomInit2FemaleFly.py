import jax
import numpy as np
import jax.random as jr
from hmms.LRHMMCustomInitFemaleFly import LRHMMCustomInitFemaleFly

from utilities import fitting

jax.config.update("jax_enable_x64", True)


class LRHMMCustomInit2FemaleFly(LRHMMCustomInitFemaleFly):
    prefix = 'lrhmmci2'

    def fit(self, emissions, inputs, output_mn_std):
        """
        TODO: need a batch id, to only fit that specific batch.
        So need to regenerate a few of the figures: r2, logll, filters, state probs,
        """
        print(f'Begin fitting {self.__class__.__name__}...')
        key = jr.PRNGKey(self.seed)

        lrhmmci = LRHMMCustomInitFemaleFly(self.data_config, self.model_config)
        lrhmmci.fit(emissions, inputs, output_mn_std)
        print(lrhmmci.learned_params)
        W = lrhmmci.learned_params.emissions.weights
        b = lrhmmci.learned_params.emissions.biases
        W = W + np.random.random(W.shape) * 0.01
        b = b + np.random.random(b.shape) * 0.01
        print("Global Initial State W and b computed.")
        em_params, em_lps = fitting.fitEMCustomInit(key, self.model, emissions, train_inputs=inputs,
                                                    emission_weights=W, emission_biases=b)
        self.learned_params = em_params
        self.learned_params = self.reindex_params(em_params, emissions, inputs, output_mn_std)
        self.learned_lps = em_lps
        self.update_status()
        print(f'End fitting {self.__class__.__name__}...')
        return
