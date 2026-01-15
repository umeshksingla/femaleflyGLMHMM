"""
Input-driven Initial State.

Based on code from Dynamax (MIT License). Modified by Umesh Singla.
"""
from typing import Any, Dict, NamedTuple, Optional, Tuple, Union
import jax.random as jr
from jaxtyping import Array, Float, Int, PyTree
import tensorflow_probability.substrates.jax.distributions as tfd
import optax

from dynamax.hidden_markov_model.inference import HMMPosterior
from dynamax.hidden_markov_model.models.abstractions import HMMInitialState
from dynamax.parameters import ParameterProperties, ParameterSet
from dynamax.types import Scalar


class ParamsInputDrivenHMMInitialState(NamedTuple):
    weights: Union[Float[Array, "num_states input_dim"], ParameterProperties]
    biases: Union[Float[Array, "num_states"], ParameterProperties]


class InputDrivenHMMInitialState(HMMInitialState):
    def __init__(self,
                 num_states: int,
                 input_dim: int,
                 m_step_optimizer: optax.GradientTransformation = optax.adam(1e-2),
                 m_step_num_iters: int = 50):
        """
        Args:
            num_states: Number of discrete states
            input_dim: Dimensionality of input vectors
        """
        super().__init__(m_step_optimizer=m_step_optimizer, m_step_num_iters=m_step_num_iters)
        self.num_states = num_states
        self.input_dim = input_dim

    def distribution(self, params: ParamsInputDrivenHMMInitialState, inputs=Float[Array, " input_dim"]) -> tfd.Distribution:
        """Return the distribution object of the initial distribution."""
        # print(params.weights.shape, inputs.shape)
        logits = params.weights @ inputs[0] + params.biases
        return tfd.Categorical(logits=logits)

    def initialize(
            self,
            key: Optional[Array] = None,
            method: str = "prior",
            **kwargs
    ) -> Tuple[ParamsInputDrivenHMMInitialState, ParamsInputDrivenHMMInitialState]:
        if method == "prior":
            # Initialize with small random weights (near zero) so transitions start near uniform
            key_w, key_b = jr.split(key)
            weights = jr.normal(key_w, (self.num_states, self.input_dim)) * 0.01
            biases = jr.normal(key_b, (self.num_states,)) * 0.01
        else:
            raise ValueError(f"Unknown initialization method: {method}")
        # Package the results into dictionaries
        params = ParamsInputDrivenHMMInitialState(weights=weights, biases=biases)
        props = ParamsInputDrivenHMMInitialState(weights=ParameterProperties(), biases=ParameterProperties())
        return params, props

    def log_prior(self, params: ParamsInputDrivenHMMInitialState) -> Scalar:
        """Compute the log prior of the parameters."""
        return 0.0

    def collect_suff_stats(self,
                           params: ParameterSet,
                           posterior: HMMPosterior,
                           inputs: Optional[Float[Array, "num_timesteps input_dim"]]=None
                           ) -> Tuple[Float[Array, " num_states"], Optional[Float[Array, " input_dim"]]]:
        return posterior.smoothed_probs[0], inputs

    def __getstate__(self):
        # Get parent's state first
        state = super().__getstate__() if hasattr(super(), '__getstate__') else self.__dict__.copy()
        # Remove the optimizer (it's stored in parent but we remove it here)
        state.pop('m_step_optimizer', None)
        return state

    def __setstate__(self, state):
        # Restore state
        if hasattr(super(), '__setstate__'):
            super().__setstate__(state)
        else:
            self.__dict__.update(state)
        # Recreate optimizer
        self.m_step_optimizer = optax.adam(1e-2)
