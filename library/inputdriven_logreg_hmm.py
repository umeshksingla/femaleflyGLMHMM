"""
Logistic regression hidden Markov model (HMM) with state-dependent weights and input-driven state transitions.

Untested. 1/1/26

"""
from typing import Any, Dict, NamedTuple, Optional, Tuple, Union
import jax.random as jr
from jaxtyping import Array, Float, Int, PyTree
import optax

from dynamax.hidden_markov_model.models.abstractions import HMM, HMMParameterSet, HMMPropertySet
from dynamax.hidden_markov_model.models.logreg_hmm import ParamsLogisticRegressionHMMEmissions, LogisticRegressionHMMEmissions
from dynamax.types import Scalar

from library.inputdriven_initstate import InputDrivenHMMInitialState, ParamsInputDrivenHMMInitialState
from library.inputdriven_transitions import InputDrivenHMMTransitions, ParamsInputDrivenHMMTransitions


class ParamsInputDrivenLogisticRegressionHMM(NamedTuple):
    """Parameters for an input-driven logistic regression HMM."""
    initial: ParamsInputDrivenHMMInitialState
    transitions: ParamsInputDrivenHMMTransitions
    emissions: ParamsLogisticRegressionHMMEmissions


class InputDrivenLogisticRegressionHMM(HMM):

    def __init__(self,
                 num_states: int,
                 input_dim: int,
                 emission_dim: int,
                 emission_matrices_scale: Scalar=1e8,
                 m_step_optimizer: optax.GradientTransformation = optax.adam(1e-2),
                 m_step_num_iters: int = 50):
        self.emission_dim = emission_dim
        self.input_dim = input_dim
        initial_component = InputDrivenHMMInitialState(num_states, input_dim)
        transition_component = InputDrivenHMMTransitions(num_states, input_dim, m_step_optimizer=m_step_optimizer, m_step_num_iters=m_step_num_iters)
        emission_component = LogisticRegressionHMMEmissions(num_states, input_dim, emission_matrices_scale=emission_matrices_scale, m_step_optimizer=m_step_optimizer, m_step_num_iters=m_step_num_iters)
        super().__init__(num_states, initial_component, transition_component, emission_component)

    @property
    def inputs_shape(self):
        """Return the shape of the input."""
        return (self.input_dim,)

    def initialize(self,
                   key: Array=jr.PRNGKey(0),
                   method: str="prior",
                   initial_probs: Optional[Float[Array, " num_states"]]=None,
                   emission_weights: Optional[Float[Array, "num_states emission_dim input_dim"]]=None,
                   emission_biases: Optional[Float[Array, "num_states emission_dim"]]=None,
                   emissions:  Optional[Float[Array, "num_timesteps emission_dim"]]=None,
                   inputs: Optional[Float[Array, "num_timesteps input_dim"]] = None,
        ) -> Tuple[HMMParameterSet, HMMPropertySet]:
        key1, key2, key3 = jr.split(key , 3)
        params, props = dict(), dict()
        params["initial"], props["initial"] = self.initial_component.initialize(key1, method=method, initial_probs=initial_probs)
        params["transitions"], props["transitions"] = self.transition_component.initialize(key2, method=method)
        params["emissions"], props["emissions"] = self.emission_component.initialize(key3, method=method, emission_weights=emission_weights, emission_biases=emission_biases, emissions=emissions, inputs=inputs)
        return ParamsInputDrivenLogisticRegressionHMM(**params), ParamsInputDrivenLogisticRegressionHMM(**props)
