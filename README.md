
### implements
input driven
female fly paper
technically also reproduce calhoun

#### Steps

### Directory structure


### Installation
1. Install Dynamax==1.0.1
2.     raise AttributeError(f"module {module!r} has no attribute {name!r}")
AttributeError: module 'jax.interpreters.xla' has no attribute 'pytype_aval_mappings'
2.  pip uninstall tensorflow-probability
3. pip install tfp-nightly tfp-nightly-0.26.0.dev2025123
2. Change Numpy, tfp-nightly
3. Jax versions

dynamax                 1.0.1
jax                     0.8.2
jaxlib                  0.8.2
jaxtyping               0.3.4
numpy                   2.2.6
optax                   0.2.6
tfp-nightly             0.26.0.dev20251231

scipy                   1.16.3
h5py                    3.15.1
joblib                  1.5.3
opencv-python           4.12.0.88

### Preprocessing Steps

Tap, visual, audio
Basis transform

##### 1. Extract features from sleap tracked files
Change DATA class to ```WT_DATA``` or ```FREDCLEANED_DATA```. This steps needs access to h5 files on cup or tigress.
It will generate a folder `wt` or `wt_fred` with a pkl file of name similar to `sessions_features_11.pkl` in the `data` folder.
`python preprocess/extract_data_from_h5.py`

##### 2. Construct design matrix i.e. inputs and outputs (or emissions) from extracted features
Change source to ```wt``` or ```wt_fred```.
It will generate a pkl file of name similar to `wt_fly_data_cos=4_ortho_o=15_aux_data.pkl` or `wt_fred_fly_data_cos=4_ortho_o=6_aux_data.pkl` in the `data` folder.
`python preprocess/get_design_matrix.py`

#### 3. Fit the model
This will run the model and generate the figures. Example:
`python run_single.py  --mc '{"names": "lrhmmci", "seeds": 25783, "num_states": 4, "transitions_matrix_stickiness": 100}' --path "general" --data_path "data/wt_fly_data_cos=4_ortho_o=15_aux_data.pkl"`
OR
`python run_single.py  --mc '{"names": "lrhmmci", "seeds": 25783, "num_states": 4, "transition_matrix_stickiness": 100}' --path "general_fred" --data_path "data/wt_fred_fly_data_cos=4_ortho_o=6_aux_data.pkl"`



Figures will be in the `models/general/lrhmmci_4/` folder.
#### Done.


### running on cluster, slurm

### library folder. modifications made to lrhmm. psd, masking, input-driven, __getstate__ and __setstate__, 
