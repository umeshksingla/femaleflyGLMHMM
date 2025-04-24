
#### Steps
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