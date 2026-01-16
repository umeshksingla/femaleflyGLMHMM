"""
Categorial Regression hidden Markov model (HMM) with state-dependent weights and input-driven state transitions.

Based on code from Dynamax (MIT License). Modified by Umesh Singla.
"""

from typing import Any, Dict, NamedTuple, Optional, Tuple, Union
import jax.random as jr
from jaxtyping import Array, Float, Int, PyTree
import optax

from dynamax.hidden_markov_model.models.abstractions import HMM, HMMParameterSet, HMMPropertySet
from dynamax.hidden_markov_model.models.categorical_glm_hmm import ParamsCategoricalRegressionHMMEmissions, CategoricalRegressionHMMEmissions

from library.inputdriven_initstate import InputDrivenHMMInitialState, ParamsInputDrivenHMMInitialState
from library.inputdriven_transitions import InputDrivenHMMTransitions, ParamsInputDrivenHMMTransitions


class ParamsInputDrivenCategoricalRegressionHMM(NamedTuple):
    """Parameters for an input-driven categorical regression HMM."""
    initial: ParamsInputDrivenHMMInitialState
    transitions: ParamsInputDrivenHMMTransitions
    emissions: ParamsCategoricalRegressionHMMEmissions


class InputDrivenCategoricalRegressionHMM(HMM):

    def __init__(self,
                 num_states: int,
                 num_classes: int,
                 input_dim: int,
                 m_step_optimizer: optax.GradientTransformation = optax.adam(1e-2),
                 m_step_num_iters: int = 50):
        self.num_classes = num_classes
        self.input_dim = input_dim
        initial_component = InputDrivenHMMInitialState(num_states, input_dim)
        transition_component = InputDrivenHMMTransitions(num_states, input_dim, m_step_optimizer=m_step_optimizer, m_step_num_iters=m_step_num_iters)
        emission_component = CategoricalRegressionHMMEmissions(num_states, num_classes, input_dim, m_step_optimizer=m_step_optimizer, m_step_num_iters=m_step_num_iters)
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
        ) -> Tuple[HMMParameterSet, HMMPropertySet]:
        key1, key2, key3 = jr.split(key , 3)
        params, props = dict(), dict()
        params["initial"], props["initial"] = self.initial_component.initialize(key1, method=method, initial_probs=initial_probs)
        params["transitions"], props["transitions"] = self.transition_component.initialize(key2, method=method)
        params["emissions"], props["emissions"] = self.emission_component.initialize(key=key3, method=method, emission_weights=emission_weights, emission_biases=emission_biases)
        return ParamsInputDrivenCategoricalRegressionHMM(**params), ParamsInputDrivenCategoricalRegressionHMM(**props)
