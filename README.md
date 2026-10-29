# Latent state dynamics underlying female behavior during *Drosophila* courtship

This repository contains the code for

> Singla et al. (2026). **Latent state dynamics underlying female behavior during *Drosophila* courtship interactions.** [Link]

We model the female's locomotion during courtship (her forward, lateral and angular velocity, in 33-ms bins) with an input-driven GLM-HMM. Its inputs are cues from the male: his movement and position relative to her, his song and his taps. Each latent state has its own GLM from the male cues to her velocity, and the cues also change the probability of switching between states. Wing flicks, a discrete rejection behavior, are predicted afterwards within each state.

The model extends [Dynamax](https://github.com/probml/dynamax) with state transitions that depend on external inputs. The paper uses linear-Gaussian emissions for continuous outputs. It can easily be extended to support other emission types.


## Repository layout

```text
.
├── preprocess/                          # stages 1-2: tracking files -> features -> design matrix
│   ├── extract_data_from_h5.py          # features from the SLEAP tracking files
│   ├── get_designmatrix.py              # z-scoring, input-output windows, basis functions
│   └── ...
├── hmms/                                # model classes (see "Names used in the code")
│   ├── BaseFemaleFly.py                 # code shared by the models
│   ├── ChanceFemaleFly.py, LRFemaleFly.py
│   ├── DiagGHMMFemaleFly.py, GHMMFemaleFly.py
│   ├── LRHMMFemaleFly.py, LRHMMCustomInitFemaleFly.py
│   ├── InputDrivenLRHMMFemaleFly.py, InputDrivenLRHMMCustomInitFemaleFly.py
│   └── ARHMMFemaleFly.py, IDARHMMFemaleFly.py   # autoregressive variants (not used in the paper)
├── library/                             # extensions to Dynamax (see "Changes to Dynamax")
│   ├── inputdriven_initstate.py         # input-driven initial state
│   ├── inputdriven_transitions.py       # input-driven transitions
│   ├── inputdriven_linreg_hmm.py        # input-driven linear-regression HMM
│   ├── inputdriven_arhmm.py             # input-driven autoregressive HMM
│   └── linreghmm.py                     # modified linear-regression HMM classes
├── utilities/
│   ├── utils.py                         # saving checkpoints
│   ├── io.py, fitting.py, video_utils.py
│   └── flick_model.py                   # post-hoc wing-flick model
├── plotting/                            # ethograms, filters, latent states (plots.py); figure style
├── crossvalidation/                     # stage 4: cross-validated figures
├── misc/                                # behavior supplements
├── run_global.py                        # entry point for fitting
├── run_ladder.py                        # one fold: penalty tuning and nested fits
├── regen_figures.py                     # remakes figures from a saved model
├── submit_slurm.py, run_slurm.sh        # job submission on Slurm clusters
└── requirements.txt
```

## Installation

The project was developed on `Python 3.13.5` on `macOS Sequoia 15.6` and also tested on various [Princeton Della](https://researchcomputing.princeton.edu/systems/della) nodes. This codebase relies on specific versions of `JAX` and `Dynamax` due to breaking changes in the JAX ecosystem. Install the following specific versions to avoid errors such as `module 'jax.interpreters.xla' has no attribute 'pytype_aval_mappings'`.

```bash
git clone https://github.com/umeshksingla/femaleflyGLMHMM.git
cd femaleflyGLMHMM
python3.13 -m venv .venv && source .venv/bin/activate

pip install dynamax==1.0.1
pip install jax==0.8.2 jaxlib==0.8.2
pip install jaxtyping==0.3.4
pip install numpy==2.2.6
pip install optax==0.2.6
pip install scipy==1.16.3 h5py==3.15.1 joblib==1.5.3 opencv-python==4.12.0.88
```

Dynamax installs the stable release of TensorFlow Probability, which does not work with JAX 0.8.2. Replace it with the nightly build:
```bash
pip uninstall tensorflow-probability
pip install tfp-nightly==0.26.0.dev20251231
```

**Patch Dynamax 1.0.1.** The `library/` folder contains critical modifications to the relevant HMM classes that are not in core Dynamax yet (v1.0.1):
* **Input-Driven Transitions**: Implements logic for transition probabilities $P(z_t | z_{t-1}, s_t)$ dependent on inputs $s_t$.
* **Numerical Stability**: Enforces diagonal covariance with a 1e-3 jitter to ensure PSD matrices during updates.
* **L2 Penalties** & **Input Masking**: ridge (L2) penalties on the emission and transition weights, and input masks so that each output uses only the inputs assigned to it.
* **Serialization**: Custom __getstate__ and __setstate__ methods so that dynamax model objects pickle correctly. 
* **Bug Fix**: Corrects the `t−1` index issue in the `hmm_backward_filter` (See this [commit](https://github.com/probml/dynamax/commit/19330338a25510e88388686eef9a41648c05c895)). This fix is applied to the locally installed Dynamax instead.


## Data

| Name in the code | In the paper | Pairs | Used for |
|---|---|---|---|
| `wt` | Dataset 1 (LEAP rig, 150 fps) | 81 | all analyses |
| `wt_fred` | Dataset 2 (16-microphone rig, 60 fps), from [Roemschied et al. (2023)](https://doi.org/10.1038/s41586-023-06632-1) | 11 | testing whether the model results generalize |
| `ac_both` | Deaf females (both aristae removed) | 31 | auto- and cross-correlations only |
| `both_eye_blind` | Blind females | 10 | auto- and cross-correlations only |

For the model, both datasets are binned at 30 Hz (33 ms).

**Getting the data:** [Link to the deposited data: to be added]. With the raw SLEAP tracking files you can run every stage below. If you have the design matrices from stage 2, start at stage 3; if you have the fitted models, start at stage 4.

Set the paths to the raw data in `preprocess/leaprig.py` (Dataset 1) and `preprocess/new16mic.py` (Dataset 2). We used cluster storage paths such as `/cup` or `/tigress`.


## Reproducing the figures

The pipeline has four stages. Each one saves its output, so you can start from whichever stage you have files for.

| Stage | Script | Output |
|---|---|---|
| 1. Tracking files → features | `preprocess/extract_data_from_h5.py` | `data/wt/sessions_features_81.pkl` |
| 2. Features → model inputs and outputs | `preprocess/get_designmatrix.py` | `data/wt_fly_data_cos=4_ortho_o=15_aux_data.pkl` |
| 3. Fit the models (cross-validated, and once to all flies) | `run_global.py`, submitted with `submit_slurm.py` | one checkpoint folder per fold, number of states and model, under `models/` |
| 4. Collect the results and plot them | `crossvalidation/`, `regen_figures.py`, `misc/` | CSV tables and figures |


### Step by step

**1. Extract features.** Set `DATA` to `WT_DATA` (Dataset 1) or `FREDCLEANED_DATA` (Dataset 2), then run

```bash
python preprocess/extract_data_from_h5.py
```

This reads the SLEAP `.h5` tracking files and writes the full time series of features (velocities, and the visual, song and tap cues) to `data/wt/` or `data/wt_fred/`, in a file like `sessions_features_81.pkl`.

**2. Build the design matrix.** Set `source` to `wt` or `wt_fred`, then run

```bash
python preprocess/get_designmatrix.py
```

This z-scores the features, defines the input and output windows, expands the male-cue histories in basis functions, and writes a file like `wt_fly_data_cos=4_ortho_o=5.pkl`.

**3. Fit the models.** Each job fits one fold for one number of states, K (`run_ladder.py`, called from `run_global.py`):

- The fold's training flies are split 80/20, and the ridge penalty on the emission weights and the L2 penalty on the transition weights are chosen by log-likelihood on the 20%.
- On all training flies, the job then fits Chance, the single-state GLM, and three nested models, each starting from the fitted parameters of the previous one: HMM without inputs (best of several random starts) → GLM-HMM with fixed transitions → GLM-HMM with cue-dependent transitions.
- It saves a checkpoint for each model and a `*_tuning.json` with the chosen penalties and validation scores. Held-out flies are scored by function `enhance` in `utilities/utils.py`.

On a Slurm cluster, set the job parameters in `submit_slurm.py` and run

```bash
python submit_slurm.py   # writes the job configurations and submits them through run_slurm.sh
```

For Dataset 1, the paper uses 10 folds over flies for K = 2–8 and 20, plus one fit to all flies with the penalties given in the Methods.

**4. Collect the results and make the figures.** Run from the repository root, with the root on `PYTHONPATH` so that the saved models can be unpickled:

```bash
export PYTHONPATH="$PWD"
RUNS=models/octladder_cv_wt_female    # folder with the cross-validation runs

python crossvalidation/collect_folds.py $RUNS 5                  # per-fly held-out scores, K = 5 -> per_fly_scores.csv
python crossvalidation/collect_k_sweep.py $RUNS                  # every K -> per_fly_by_K.csv
python crossvalidation/collect_pearsoncorrelations.py $RUNS 5    # correlations -> correlations_by_fly.csv
python crossvalidation/make_figures.py                           # Prediction figures (LLs, pearson, F1, etc) -> figures/
```

- **Collect scripts:** each prints the statistics reported in the paper (medians, interquartile ranges, Wilcoxon signed-rank tests) and writes a CSV to the current folder. The first run reads every checkpoint once and caches a small summary next to it (`scores.json`, `correlations.json`), so later runs are fast. `collect_folds_auxem.py` collects the wing-flick results (the post-hoc model in `utilities/flick_model.py`).
- **`make_figures.py`:** takes its input paths from `CONFIG` at the top of the file. For Dataset 2, run the collect scripts on its folder and add the tables to `CONFIG`.
- **Figs. 3–5:** run `regen_figures.py` on the fit to all flies. It remakes the figures from the saved model without refitting.
- **Behavior supplements:** these come from `misc/autocorr.py`, `misc/crosscorr.py` and `misc/plot_leap_vs_16mic_dists.py`, which read the stage-1 features.


## Names used in the code

| In the paper | In the code (checkpoint prefix, class) |
|---|---|
| Chance | `chance`, `hmms/ChanceFemaleFly.py` |
| GLM (one state) | `lr`, `hmms/LRFemaleFly.py` |
| HMM without inputs | `dghmm`, `hmms/DiagGHMMFemaleFly.py` |
| GLM-HMM with fixed transitions | `lrhmmci_`, `hmms/LRHMMCustomInitFemaleFly.py` |
| GLM-HMM with cue-dependent transitions (the full model) | `id-glm-hmm`, `hmms/InputDrivenLRHMMCustomInitFemaleFly.py` |
| Wing flicks | auxiliary outputs (`aux`, `auxem`); `utilities/flick_model.py` |
| Dataset 1, Dataset 2 | `wt`, `wt_fred` |

The `CustomInit` classes start from the fitted parameters of the previous model in the sequence.

## Contact

- **Lead Contact:** Mala Murthy (mmurthy@princeton.edu)
- **Lab Website:** [murthylab.princeton.edu](https://murthylab.princeton.edu)
- **Code Issues:** Please open an issue on GitHub.

## Citation

If you use this code, please cite the paper above. The model builds on [Dynamax](https://github.com/probml/dynamax) and on the GLM-HMM of Calhoun et al. (2019), *Unsupervised identification of the internal states that shape natural behavior*, Nature Neuroscience. Dataset 2 is from Roemschied et al. (2023), *Flexible circuit mechanisms for context-dependent song sequencing*, Nature.

## License

See [LICENSE.md](LICENSE.md).
