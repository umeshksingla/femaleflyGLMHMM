"""
Linear regression hidden Markov model (HMM) with state-dependent weights and input-driven state transitions.

Based on code from Dynamax (MIT License). Modified by Umesh Singla.
"""
from typing import Any, Dict, NamedTuple, Optional, Tuple, Union
import jax.random as jr
from jaxtyping import Array, Float, Int, PyTree
import optax

from dynamax.hidden_markov_model.models.abstractions import HMM, HMMParameterSet, HMMPropertySet
from dynamax.hidden_markov_model.models.linreg_hmm import ParamsLinearRegressionHMMEmissions
from dynamax.types import Scalar

from library.linreghmm import LinearRegressionHMMEmissionsCustom
from library.inputdriven_initstate import InputDrivenHMMInitialState, ParamsInputDrivenHMMInitialState
from library.inputdriven_transitions import InputDrivenHMMTransitions, ParamsInputDrivenHMMTransitions


class ParamsInputDrivenLinearRegressionHMM(NamedTuple):
    """Parameters for an input-driven linear regression HMM."""
    initial: ParamsInputDrivenHMMInitialState
    transitions: ParamsInputDrivenHMMTransitions
    emissions: ParamsLinearRegressionHMMEmissions


class InputDrivenLinearRegressionHMM(HMM):

    def __init__(self,
                 num_states: int,
                 input_dim: int,
                 emission_dim: int,
                 input_mask_by_emission: Float[Array, "emission_dim input_dim"] = None,
                 input_mask_first: Float[Array, "input_dim"] = None,
                 m_step_optimizer: optax.GradientTransformation = optax.adam(1e-2),
                 m_step_num_iters: int = 50,
                 l2_penalty: float = 1.0):
        self.emission_dim = emission_dim
        self.input_dim = input_dim
        self.l2_penalty = l2_penalty
        print("!!! Model initialized with", self.l2_penalty)
        initial_component = InputDrivenHMMInitialState(num_states, input_dim, input_mask_first)
        transition_component = InputDrivenHMMTransitions(num_states, input_dim, input_mask_first, m_step_optimizer=m_step_optimizer, m_step_num_iters=m_step_num_iters, l2_penalty=l2_penalty)
        emission_component = LinearRegressionHMMEmissionsCustom(num_states, input_dim, emission_dim, input_mask_by_emission)
        super().__init__(num_states, initial_component, transition_component, emission_component)

    @property
    def inputs_shape(self):
        """Return the shape of the input."""
        return (self.input_dim,)

    def initialize(self,
                   key: Array=jr.PRNGKey(0),
                   method: str="prior",
                   initial_probs: Optional[Float[Array, " num_states"]]=None,
                   transition_weights: Optional[Float[Array, "num_states num_states input_dim"]]=None,
                   transition_biases: Optional[Float[Array, "num_states num_states"]]=None,
                   emission_weights: Optional[Float[Array, "num_states emission_dim input_dim"]]=None,
                   emission_biases: Optional[Float[Array, "num_states emission_dim"]]=None,
                   emission_covariances:  Optional[Float[Array, "num_states emission_dim emission_dim"]]=None,
                   emissions:  Optional[Float[Array, "num_timesteps emission_dim"]]=None
        ) -> Tuple[HMMParameterSet, HMMPropertySet]:
        key1, key2, key3 = jr.split(key , 3)
        params, props = dict(), dict()
        params["initial"], props["initial"] = self.initial_component.initialize(key1, method=method, initial_probs=initial_probs)
        params["transitions"], props["transitions"] = self.transition_component.initialize(key2, method=method, transition_weights=transition_weights, transition_biases=transition_biases)
        params["emissions"], props["emissions"] = self.emission_component.initialize(key3, method=method, emission_weights=emission_weights, emission_biases=emission_biases, emission_covariances=emission_covariances, emissions=emissions)
        return ParamsInputDrivenLinearRegressionHMM(**params), ParamsInputDrivenLinearRegressionHMM(**props)
