import jax
import numpy as np
import jax.numpy as jnp
import jax.random as jr
from hmms.InputDrivenLRHMMFemaleFly import InputDrivenLRHMMFemaleFly
from hmms.LRFemaleFly import LRFemaleFly
from hmms.LRHMMCustomInitFemaleFly import LRHMMCustomInitFemaleFly
from hmms.ChanceFemaleFly import ChanceFemaleFly
from utilities import fitting, io

jax.config.update("jax_enable_x64", True)


class InputDrivenLRHMMCustomInitFemaleFly(InputDrivenLRHMMFemaleFly):

    prefix = 'id-glm-hmm'

    def fit(self, batched_emissions, batched_inputs, batched_output_mn_std):
        """
            batched_emissions: session by session
            batched_inputs:
            batched_output_mn_std:
        """
        print(f'Begin fitting chance...')
        chance = ChanceFemaleFly(self.data_config, self.model_config)
        chance.fit(batched_emissions, batched_inputs)
        chance_params = chance.learned_params
        self.chance_mu = chance_params['mu']
        self.chance_cov = chance_params['cov']
        print('chance fit.')
        print(f'Begin fitting {self.__class__.__name__}...')
        key = jr.PRNGKey(self.seed)

        # since the sessions are variable length, chunk the sessions.
        emissions_to_fit = io.chunk_data(batched_emissions, chunk_size=5000)
        inputs_to_fit = io.chunk_data(batched_inputs, chunk_size=5000)

        lrhmmci = LRHMMCustomInitFemaleFly(self.data_config, self.model_config)
        lrhmmci.fit(batched_emissions, batched_inputs, batched_output_mn_std)

        W = lrhmmci.learned_params.emissions.weights
        b = lrhmmci.learned_params.emissions.biases
        W = W + np.random.random(W.shape) * 1e-4
        b = b + np.random.random(b.shape) * 1e-4

        init_P = lrhmmci.learned_params.initial.probs
        tr_b = np.log(lrhmmci.learned_params.transitions.transition_matrix)

        print("LRHMMCI global W and b computed.", W.shape, b.shape)
        print("ID-GLMHMMCI init_P and tr_b init.", init_P.shape, tr_b.shape)
        em_params, em_lps = fitting.fitEMInputDrivenCustomInit(key,
                                                               self.model,
                                                               emissions_to_fit,
                                                               train_inputs=inputs_to_fit,
                                                               initial_probs=init_P,
                                                               transition_weights=None, # weights can be initialized from normal in the class itself
                                                               transition_biases=tr_b,
                                                               emission_weights=W,
                                                               emission_biases=b)
        self.learned_params = em_params
        self.learned_params = self.reindex_params(em_params, batched_emissions, batched_inputs, batched_output_mn_std)
        self.learned_lps = em_lps
        self.update_status()
        print(f'End fitting {self.__class__.__name__}...')
        return
