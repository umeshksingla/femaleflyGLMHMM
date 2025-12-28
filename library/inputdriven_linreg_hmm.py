"""
Linear regression hidden Markov model (HMM) with state-dependent weights and input-driven state transitions.
"""
from typing import Any, Dict, NamedTuple, Optional, Tuple, Union
import jax.numpy as jnp
import jax.random as jr
from jax import vmap
from jaxtyping import Array, Float, Int, PyTree
from tensorflow_probability.substrates import jax as tfp
import tensorflow_probability.substrates.jax.distributions as tfd
import optax

from dynamax.parameters import to_unconstrained, from_unconstrained
from dynamax.hidden_markov_model.inference import HMMPosterior
from dynamax.hidden_markov_model.models.abstractions import HMM, HMMInitialState, HMMParameterSet, HMMPropertySet, HMMTransitions
# from dynamax.hidden_markov_model.models.initial import StandardHMMInitialState, ParamsStandardHMMInitialState
# from dynamax.hidden_markov_model.models.transitions import StandardHMMTransitions
from dynamax.hidden_markov_model.models.linreg_hmm import ParamsLinearRegressionHMMEmissions, LinearRegressionHMMEmissions
from dynamax.parameters import ParameterProperties, ParameterSet, PropertySet
from dynamax.types import Scalar
from dynamax.utils.optimize import run_gradient_descent
from dynamax.utils.utils import pytree_sum
from dynamax.utils.utils import pytree_slice
from dynamax.utils.bijectors import RealToPSDBijector

from library.linreghmm import LinearRegressionHMMEmissionsCustom


class ParamsInputDrivenHMMInitialState(NamedTuple):
    weights: Union[Float[Array, "num_states input_dim"], ParameterProperties]
    biases: Union[Float[Array, "num_states"], ParameterProperties]


class ParamsInputDrivenHMMTransitions(NamedTuple):
    """Parameters for the transitions of an input-driven HMM."""
    weights: Union[Float[Array, "num_states num_states input_dim"], ParameterProperties]    # CHECK??
    biases: Union[Float[Array, "num_states num_states"], ParameterProperties]


class ParamsInputDrivenLinearRegressionHMM(NamedTuple):
    """Parameters for a linear regression HMM."""
    initial: ParamsInputDrivenHMMInitialState
    transitions: ParamsInputDrivenHMMTransitions
    emissions: ParamsLinearRegressionHMMEmissions


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


class InputDrivenLinearRegressionHMM(HMM):

    def __init__(self,
                 num_states: int,
                 input_dim: int,
                 emission_dim: int,
                 input_mask_by_emission: Float[Array, "emission_dim input_dim"] = None,
                 initial_probs_concentration: Union[Scalar, Float[Array, " num_states"]]=1.1,
                 m_step_optimizer: optax.GradientTransformation = optax.adam(1e-2),
                 m_step_num_iters: int = 50):
        self.emission_dim = emission_dim
        self.input_dim = input_dim
        initial_component = InputDrivenHMMInitialState(num_states, input_dim)
        transition_component = InputDrivenHMMTransitions(num_states, input_dim, m_step_optimizer=m_step_optimizer, m_step_num_iters=m_step_num_iters)
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
                   emission_weights: Optional[Float[Array, "num_states emission_dim input_dim"]]=None,
                   emission_biases: Optional[Float[Array, "num_states emission_dim"]]=None,
                   emission_covariances:  Optional[Float[Array, "num_states emission_dim emission_dim"]]=None,
                   emissions:  Optional[Float[Array, "num_timesteps emission_dim"]]=None
        ) -> Tuple[HMMParameterSet, HMMPropertySet]:
        key1, key2, key3 = jr.split(key , 3)
        params, props = dict(), dict()
        params["initial"], props["initial"] = self.initial_component.initialize(key1, method=method, initial_probs=initial_probs)
        params["transitions"], props["transitions"] = self.transition_component.initialize(key2, method=method)
        params["emissions"], props["emissions"] = self.emission_component.initialize(key3, method=method, emission_weights=emission_weights, emission_biases=emission_biases, emission_covariances=emission_covariances, emissions=emissions)
        return ParamsInputDrivenLinearRegressionHMM(**params), ParamsInputDrivenLinearRegressionHMM(**props)
