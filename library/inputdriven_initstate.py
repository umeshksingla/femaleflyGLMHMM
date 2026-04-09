"""
Input-driven Initial State.

Based on code from Dynamax (MIT License). Modified by Umesh Singla.
"""
from typing import Any, Dict, NamedTuple, Optional, Tuple, Union
import jax.random as jr
import jax.numpy as jnp
from jax import vmap
from jaxtyping import Array, Float, Int, PyTree
import tensorflow_probability.substrates.jax.distributions as tfd
import optax

from dynamax.utils.optimize import run_gradient_descent
from dynamax.parameters import to_unconstrained, from_unconstrained
from dynamax.hidden_markov_model.inference import HMMPosterior
from dynamax.hidden_markov_model.models.abstractions import HMMInitialState
from dynamax.parameters import ParameterProperties, ParameterSet, PropertySet
from dynamax.types import Scalar


class ParamsInputDrivenHMMInitialState(NamedTuple):
    weights: Union[Float[Array, "num_states input_dim"], ParameterProperties]
    biases: Union[Float[Array, "num_states"], ParameterProperties]


class InputDrivenHMMInitialState(HMMInitialState):
    def __init__(self,
                 num_states: int,
                 input_dim: int,
                 input_mask_first=None,
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
        if input_mask_first is None:
            input_mask_first = jnp.ones((self.input_dim,))
        self.input_mask_first_full = jnp.broadcast_to(input_mask_first,(num_states, input_dim))
        print("IDHMMInitState input_mask_by_emission", input_mask_first.shape, "input_mask_by_emission_full", self.input_mask_first_full.shape)

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

    def m_step(self,
               params: ParameterSet,
               props: PropertySet,
               batch_stats: PyTree,
               m_step_state: Any,
               scale: float=1.0
    ) -> Tuple[ParameterSet, Any]:
        """Perform an M-step on the initial distribution parameters.

        Args:
            params: current initial distribution parameters
            props: parameter properties
            batch_stats: PyTree of sufficient statistics from each sequence, as output by :meth:`collect_suff_stats`.
            m_step_state: any state required for the M-step
            scale: how to scale the objective

        Returns:
            Parameters that maximize the expected log joint probability.

        """

        # Extract the remaining unconstrained params, which should only be for the emissions.
        unc_params = to_unconstrained(params, props)

        # Minimize the negative expected log joint probability
        def neg_expected_log_joint(unc_params):
            """Compute the negative expected log joint probability."""
            params = from_unconstrained(unc_params, props)
            params = params._replace(weights=params.weights * self.input_mask_first_full)  # Zero out the irrelevant weights (Step 1)
            def _single_expected_log_like(stats):
                """Compute the expected log likelihood for a single sequence."""
                expected_initial_state, inpt = stats
                log_initial_prob = jnp.log(self._compute_initial_probs(params, inpt))
                lp = jnp.sum(expected_initial_state * log_initial_prob)
                return lp

            log_prior = self.log_prior(params)
            batch_ells = vmap(_single_expected_log_like)(batch_stats)
            expected_log_joint = log_prior + batch_ells.sum()
            return -expected_log_joint / scale

        # Run gradient descent
        unc_params, m_step_state, losses = \
            run_gradient_descent(neg_expected_log_joint,
                                 unc_params,
                                 self.m_step_optimizer,
                                 optimizer_state=m_step_state,
                                 num_mstep_iters=self.m_step_num_iters)

        # Return the updated parameters and optimizer state
        params = from_unconstrained(unc_params, props)
        params = params._replace(weights=params.weights * self.input_mask_first_full)   # Zero out the irrelevant weights (Step 2)
        return params, m_step_state

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
