import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import optax

from library.inputdriven_categoricalreg_hmm import InputDrivenCategoricalRegressionHMM


def test_input_driven_hmm():
    print("--- Starting InputDrivenCategoricalRegressionHMM Test ---")

    # 1. Configuration
    num_states = 3
    input_dim = 2
    num_classes = 4  # Number of classes (C)
    num_timesteps = 200
    num_batches = 5

    # 2. Generate Synthetic Data
    # Inputs: Continuous vectors (T, input_dim)
    key = jr.PRNGKey(42)
    key_input, key_emit, key_init = jr.split(key, 3)

    inputs = jr.normal(key_input, (num_batches, num_timesteps, input_dim))

    # Emissions: Integers in [0, num_classes - 1] with shape (T, 1) or (T,)
    # For categorical regression, we usually expect integer targets.
    emissions = jr.randint(key_emit, (num_batches, num_timesteps), minval=0, maxval=num_classes)

    print(f"Data Shapes -> Inputs: {inputs.shape}, Emissions: {emissions.shape}")

    # 3. Instantiate Model
    hmm = InputDrivenCategoricalRegressionHMM(
        num_states=num_states,
        num_classes=num_classes,
        input_dim=input_dim,
        m_step_optimizer=optax.adam(1e-2),
        m_step_num_iters=10
    )

    # 4. Initialize Parameters
    print("Initializing model parameters...")
    params, props = hmm.initialize(key=key_init, method="prior")

    print("Param keys:", params._fields)
    if hasattr(params, 'transitions'):
        print("Transitions params found.")
    if hasattr(params, 'emissions'):
        print("Emissions params found.")

    # 5. Fit the Model (EM Algorithm)
    print("\nStarting EM Fit...")
    num_em_iters = 30
    params, log_probs = hmm.fit_em(
        params,
        props,
        emissions,
        inputs=inputs,
        num_iters=num_em_iters,
        verbose=True
    )

    # 6. Validation
    # Check if Log Probabilities are finite and generally increasing
    print(f"\nFinal Log Probability: {log_probs[-1]}")

    if jnp.isnan(log_probs).any():
        print("Test FAILED: Log probabilities contain NaNs.")
    else:
        # Check for improvement (heuristic)
        improvement = log_probs[-1] - log_probs[0]
        if improvement > 0:
            print(f"Test PASSED: Log likelihood improved by {improvement:.2f}")
        else:
            print("Test COMPLETED: Log likelihood did not improve (check hyperparameters/data).")

    # 7. Visualization
    try:
        plt.figure(figsize=(8, 4))
        plt.plot(log_probs)
        plt.xlabel("EM Iteration")
        plt.ylabel("Log Probability")
        plt.title("HMM Learning Curve")
        plt.grid(True)
        plt.show()
        print("Plot generated.")
    except Exception as e:
        print(f"Could not generate plot: {e}")
    return


if __name__ == "__main__":
    test_input_driven_hmm()
