import jax
import numpy as np
import jax.numpy as jnp
import jax.random as jr
from hmms.InputDrivenLRHMMFemaleFly import InputDrivenLRHMMFemaleFly
from hmms.LRFemaleFly import LRFemaleFly
from hmms.LRHMMCustomInitFemaleFly import LRHMMCustomInitFemaleFly
from hmms.ChanceFemaleFly import ChanceFemaleFly
from utilities import fitting, io
from utilities.io import ENABLE_FLOAT64

jax.config.update("jax_enable_x64", ENABLE_FLOAT64)


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
        lrhmmci.fit(batched_emissions, batched_inputs, batched_output_mn_std, do_reindex_params=False)
        p = lrhmmci.learned_params
        print("LRHMMCI W, b and s computed.")
        D = inputs_to_fit.shape[-1]
        em_params, em_lps = fitting.fitEMInputDrivenCustomInit(key,
                                                               self.model,
                                                               emissions_to_fit,
                                                               train_inputs=inputs_to_fit,

                                                               initial_weights=jnp.zeros((self.num_states, D)),
                                                               initial_biases=jnp.log(jnp.clip(p.initial.probs, 1e-8)),

                                                               transition_weights=np.zeros((self.num_states, self.num_states, D)), # init weights to be 0
                                                               transition_biases=np.log(p.transitions.transition_matrix),

                                                               emission_weights=p.emissions.weights,
                                                               emission_biases=p.emissions.biases,
                                                               emission_covariances=p.emissions.covs)
        self.learned_params = em_params
        self.learned_params = self.reindex_params(em_params, batched_emissions, batched_inputs, batched_output_mn_std)
        self.learned_lps = em_lps
        print(f'ID-GLMHMMCI training loss: start {float(em_lps[0]):.4f}, end {float(em_lps[-1]):.4f}')
        self.update_status()
        print(f'End fitting {self.__class__.__name__}...')
        return
