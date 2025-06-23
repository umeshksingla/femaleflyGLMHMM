import jax
import numpy as np
import jax.random as jr
from hmms.LogRHMMFemaleFly import LogRHMMFemaleFly
from hmms.LogRFemaleFly import LogRFemaleFly

from utilities import fitting

jax.config.update("jax_enable_x64", True)


class LogRHMMCustomInitFemaleFly(LogRHMMFemaleFly):

    prefix = 'logrhmmci'

    def fit(self, emissions, inputs, output_mn_std):
        print(f'Begin fitting {self.__class__.__name__}...')
        key = jr.PRNGKey(self.seed)

        lr = LogRFemaleFly(self.data_config, self.model_config)
        lr.fit(emissions, inputs, output_mn_std)
        print("LR global W and b computed.", lr.learned_params['w'].shape, lr.learned_params['b'].shape)
        W = np.repeat(lr.learned_params['w'].squeeze(axis=0), repeats=self.num_states, axis=0)  # NOTE .squeeze() coz LogRegHMM only supports one emission
        b = np.repeat(lr.learned_params['b'].squeeze(axis=0), repeats=self.num_states, axis=0)
        W = W + np.random.random(W.shape) * 0.01
        b = b + np.random.random(b.shape) * 0.01
        print("LR global W and b computed.", W.shape, b.shape)
        em_params, em_lps = fitting.fitEMLogRHMMCustomInit(key, self.model, emissions, train_inputs=inputs,
                                                    emission_weights=W, emission_biases=b)
        self.learned_params = em_params
        # self.learned_params = self.reindex_params(em_params, emissions, inputs, output_mn_std)
        self.learned_lps = em_lps
        self.update_status()
        print(f'End fitting {self.__class__.__name__}...')
        return
