"""
Input-driven State transitions.

Based on code from Dynamax (MIT License). Modified by Umesh Singla.

"""
from typing import Any, Dict, NamedTuple, Optional, Tuple, Union
import jax.random as jr
from jaxtyping import Array, Float, Int, PyTree
import tensorflow_probability.substrates.jax.distributions as tfd
import optax

from dynamax.hidden_markov_model.inference import HMMPosterior
from dynamax.hidden_markov_model.models.abstractions import HMMTransitions
from dynamax.parameters import ParameterProperties, ParameterSet
from dynamax.types import Scalar


class ParamsInputDrivenHMMTransitions(NamedTuple):
    """Parameters for the transitions of an input-driven HMM."""
    weights: Union[Float[Array, "num_states num_states input_dim"], ParameterProperties]    # CHECK??
    biases: Union[Float[Array, "num_states num_states"], ParameterProperties]


class InputDrivenHMMTransitions(HMMTransitions):
    """
    HMM transitions for an input-driven HMM.
    The transition probabilities depend on external inputs/covariates:
        P(z_t | z_{t-1}, u_t) where u_t are inputs at time t

    For each previous state j, we use multinomial logistic regression:
        P(z_t = k | z_{t-1} = j, u_t) = softmax(W_j @ u_t + b_j)[k]
    """

    def __init__(
            self,
            num_states: int,
            input_dim: int,
            m_step_optimizer: optax.GradientTransformation = optax.adam(1e-2),
            m_step_num_iters: int = 50
    ):
        """
        Args:
            num_states: Number of discrete states
            input_dim: Dimensionality of input vectors
        """
        super().__init__(m_step_optimizer=m_step_optimizer, m_step_num_iters=m_step_num_iters)
        self.num_states = num_states
        self.input_dim = input_dim

    def distribution(
            self,
            params: ParamsInputDrivenHMMTransitions,
            state: Union[int, Int[Array, ""]],
            inputs: Float[Array, " input_dim"]) -> tfd.Distribution:
        """
        Return the distribution over the next state given the current state and input.

        Compute logits: W[state] @ inputs + b[state]
        weights[state] has shape (input_dim, num_states)
        inputs has shape (input_dim,)
        Result has shape (num_states,)
        """
        if inputs is None:
            raise ValueError("Inputs must be provided for input-driven transitions")

        logits = params.weights[state] @ inputs + params.biases[state]
        return tfd.Categorical(logits=logits)

    def initialize(
            self,
            key: Optional[Array] = None,
            method: str = "prior", **kwargs
    ) -> Tuple[ParamsInputDrivenHMMTransitions, ParamsInputDrivenHMMTransitions]:
        if method == "prior":
            # Initialize with small random weights (near zero) so transitions start near uniform
            key_w, key_b = jr.split(key)
            weights = jr.normal(key_w, (self.num_states, self.num_states, self.input_dim)) * 0.01
            biases = jr.normal(key_b, (self.num_states, self.num_states)) * 0.01
        else:                                                                               # CHECK??
            raise ValueError(f"Unknown initialization method: {method}")
        # Package the results into dictionaries
        params = ParamsInputDrivenHMMTransitions(weights=weights, biases=biases)
        props = ParamsInputDrivenHMMTransitions(weights=ParameterProperties(), biases=ParameterProperties())
        return params, props

    def log_prior(self, params: ParamsInputDrivenHMMTransitions) -> Scalar:
        """Return the log-prior probability of the emission parameters.

        Currently, there is no prior so this function returns 0.
        """
        return 0.0

    def collect_suff_stats(self,
                           params: ParameterSet,
                           posterior: HMMPosterior,
                           inputs: Optional[Float[Array, "num_timesteps input_dim"]]=None
    ) -> PyTree:
        """Collect sufficient statistics for updating the transition distribution parameters.

        Args:
            params: transition distribution parameters
            posterior: posterior distribution over latent states
            inputs: optional inputs

        Returns:
            PyTree of sufficient statistics for updating the transition distribution

        """
        return posterior.trans_probs, inputs

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
