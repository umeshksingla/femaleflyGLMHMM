This code was developed to analyze female _Drosophila_ behavior during courtship, as described in "**Singla et al. (2026). Latent state dynamics underlying female Drosophila response variability during courtship**"

It implements an input-driven GLM-HMM, extending the base Dynamax library to support state transitions explicitly modulated by external sensory inputs. The current project uses Linear-Gaussian emissions (for continuous observations) but the model class for categorical emissions is also provided (e.g., to predict song outputs as in Calhoun et al., 2019.). It can easily be extended to support other emission types.

### Directory Structure

```text
.
├── hmms/                               # Model classes built on top of the base architecture
│   ├── LRFemaleFly.py                  # Standard Linear Regression class
│   ├── LRHMMFemaleFly.py               # Standard HMM with linear regression emissions.
│   └── InputDrivenLRHMMFemaleFly.py    # Core GLM-HMM with input-driven transitions
├── library/                            # Custom extensions to Dynamax (features not yet in core)
│   └── inputdriven_initstate.py            # Input-driven Initial State class
│   └── inputdriven_transitions.py          # Input-driven Transitions class
│   └── inputdriven_linreg_hmm.py           # Input-driven Linear Regression HMM class
│   └── inputdriven_categoricalreg_hmm.py   # Input-driven Categorical Regression HMM class
|   └── linreghmm.py                        # (Modified linreg_hmm classes, see below)
├── preprocess/                         # Data pipelines
│   ├── extract_data_from_h5.py         # Extract raw tracking/sensory data
│   └── get_designmatrix.py             # Z-scoring, basis transformations
├── utilities/                          # Helper functions for saving trained models and data, for figures
├── plotting/                           # Visualization tools (ethograms, filters, latent states)
└── adhoc_scripts/                      # Scripts for comparing and plotting cross-validation results
```

### Installation

The project was developed on `Python 3.13.5` on `macOS Sequoia 15.6` and also tested on various [Princeton Della](https://researchcomputing.princeton.edu/systems/della) nodes. This codebase relies on specific versions of `JAX` and `Dynamax` due to breaking changes in the JAX ecosystem. 

Install the following specific versions to avoid errors such as `module 'jax.interpreters.xla' has no attribute 'pytype_aval_mappings'`.

```bash
pip install dynamax==1.0.1
pip install jax==0.8.2 jaxlib==0.8.2
pip install jaxtyping==0.3.4
pip install numpy==2.2.6
pip install optax==0.2.6
pip install scipy==1.16.3 h5py==3.15.1 joblib==1.5.3 opencv-python==4.12.0.88
```

You must use the nightly build of TensorFlow Probability to be compatible with JAX 0.8.2. Uninstall any existing version first:
```bash
pip uninstall tensorflow-probability
pip install tfp-nightly==0.26.0.dev20251231
```

### Key Modifications
The `library/` folder contains critical modifications to the relevant HMM classes that are not in core Dynamax yet (v1.0.1):
* **Input-Driven Transitions**: Implements logic for transition probabilities $P(z_t | z_{t-1}, s_t)$ dependent on inputs $s_t$.
* **Numerical Stability**: Enforces diagonal covariance with a 1e-6 jitter to ensure PSD matrices during updates.
* **L2 Normalization** & **Input Masking**: Added support for passing arguments to mask specific inputs during training.
* **Serialization**: Custom __getstate__ and __setstate__ methods to ensure JAX model objects pickle correctly. 
* **Bug Fix**: Corrects the `t−1` index issue in the `hmm_backward_filter` (See this [commit](https://github.com/probml/dynamax/commit/19330338a25510e88388686eef9a41648c05c895)). Note that this was manually modified in the local installation of dynamax directly.

#### Usage Pipeline
##### 0. Configuration
Before running, update the paths to your raw data (e.g., cluster paths like /cup or /tigress) in:
* `preprocess/leaprig.py`
* `preprocess/new16mic.py`

##### 1. Data Extraction (Extract features from SLEAP tracked files)
Extract full timeseries of features (velocities, sensory cues like tap, visual, audio) from raw H5 tracking files. 

Change DATA class to ```WT_DATA``` or ```FREDCLEANED_DATA```. This steps needs access to h5 files on cup or tigress. It will generate a folder `wt` or `wt_fred` with a pkl file of name similar to `sessions_features_81.pkl` in the `data` folder.
```bash
python preprocess/extract_data_from_h5.py
```

##### 2. Construct design matrix (inputs and outputs (or emissions) from extracted features)
Format the data for the GLM-HMM. This includes z-scoring, defining input-output windows, and performing basis function expansions.

Change source to ```wt``` or ```wt_fred```. It will generate a pkl file of name similar to `wt_fly_data_cos=4_ortho_o=15_aux_data.pkl` in the `data` folder. 
```bash
python preprocess/get_designmatrix.py
```

##### 3. Model Training
Use `run_global.py` or `run_separate.py` to train the model and generate the figures. Trained models and figures are saved to `models/<path>/<model_name>_<numstates>/` folder.

#### Slurm Support (running on HPC clusters)
Scripts are provided to run training jobs on Slurm-based clusters (we tested on Princeton Della).
* **Configure**: Edit `submit_slurm.py` to set job parameters.
* **Submit**: Run `python submit_slurm.py`. This script automatically generates configurations and calls `run_slurm.sh` to submit jobs.

> **Note:** This implementation builds on [Dynamax](https://probml.github.io/dynamax/index.html) and JAX and can also be used to reproduce results from similar input-driven HMM studies, such as *Calhoun et al., 2019*.
