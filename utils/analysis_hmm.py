import os
import sys

import numpy as np

from plotting import plots
import utils

# Display plots
display = True
savefig = True

# Load model
m1 = 'lrhmm_15/20250114_155847_version'
model_ckp, data_config, model_config = utils.load_specific_path(f'models/{m1}')

# Create directory to output figures
fig_dir = f'models/{m1}/figures'
os.makedirs(fig_dir, exist_ok=True)

# Load data, labels and config used while training the model
print(model_ckp.keys())
train_emissions = model_ckp['train_data']['train_emissions']
train_inputs = model_ckp['train_data']['train_inputs']
train_stateseq = model_ckp['train_data']['train_stateseq']
train_emission_predictions = model_ckp['train_data']['train_predictions']
test_emissions = model_ckp['test_data']['test_emissions']
test_inputs = model_ckp['test_data']['test_inputs']
test_stateseq = model_ckp['test_data']['test_stateseq']
test_emission_predictions = model_ckp['test_data']['test_predictions']
learned_params = model_ckp['learned_params']
learned_lps = model_ckp['learned_lps']
output_indices = model_ckp['output_indices']
basis = data_config['basis']
emission_labels = data_config['emission_labels']
input_raw_each_dim = data_config['input_raw_each_dim']

# Plots
plots.plot_loss(learned_lps, savefig=savefig, fig_dir=fig_dir, display=display)
plots.plot_prob_states(train_stateseq, model_config, title='train', savefig=savefig, fig_dir=fig_dir, display=display)
plots.plot_prob_states(test_stateseq, model_config, title='held-out', savefig=savefig, fig_dir=fig_dir, display=display)
plots.plot_transition_matrix(learned_params.transitions.transition_matrix, savefig=savefig, fig_dir=fig_dir, display=display)
plots.plot_steady_state(utils.calculate_steady_state_p(learned_params.transitions.transition_matrix), savefig=savefig, fig_dir=fig_dir, display=display)
plots.plot_filters(learned_params.emissions.weights, data_config, savefig=savefig, fig_dir=fig_dir, display=display)
plots.plot_var_explained_by_z(model_ckp['train_data']['train_score_by_z'], title='Train Data', savefig=savefig, fig_dir=fig_dir, display=display)
plots.plot_var_explained_by_z_o(model_ckp['train_data']['train_score_by_z_and_o'], emission_labels, title='Train Data', savefig=savefig, fig_dir=fig_dir, display=display)
plots.plot_correlation_by_o(model_ckp['train_data']['train_correlation_by_o'], emission_labels, title='Train Data', savefig=savefig, fig_dir=fig_dir, display=display)
plots.plot_var_explained_by_z(model_ckp['test_data']['test_score_by_z'], title='Held-out Data', savefig=savefig, fig_dir=fig_dir, display=display)
plots.plot_var_explained_by_z_o(model_ckp['test_data']['test_score_by_z_and_o'], emission_labels, title='Held-out Data', savefig=savefig, fig_dir=fig_dir, display=display)
plots.plot_correlation_by_o(model_ckp['test_data']['test_correlation_by_o'], emission_labels, title='Held-out Data', savefig=savefig, fig_dir=fig_dir, display=display)

os.makedirs(f'{fig_dir}/trajs', exist_ok=True)
for xlim in [None, (0, 1000), (1500, 2000), (10000, 15000), (0, 5000), (16000, 17000),][:2]:
    for batch in np.random.choice(range(len(train_stateseq)), size=5, replace=False):
        plots.plot_trajectories(model_ckp, data_config, batch, prefix_data='train', xlim=xlim, savefig=True, fig_path=f'{fig_dir}/trajs/train_xlim={xlim}.pdf', display=display)

for xlim in [None, (0, 1000), (1500, 2000), (10000, 15000), (0, 5000), (16000, 17000),][:2]:
    for batch in np.random.choice(range(len(test_stateseq)), size=5, replace=False):
        plots.plot_trajectories(model_ckp, data_config, batch, prefix_data='test', xlim=xlim, savefig=True, fig_path=f'{fig_dir}/trajs/test_xlim={xlim}.pdf', display=display)
print("Done with trajectories.")

sys.exit()

# Generate clips
intervals_dict = utils.get_stateseq_indices(train_stateseq, train_emissions, min_length=100)
intervals_video_frmidxs = utils.map_to_video_frame_indices(intervals_dict, output_indices)

for b in intervals_dict:
    print(f"On b={b}")
    session_key = data_config['session_keys'][b]
    mp4filepath = os.path.join('/Volumes/murthy/usingla/gold_dataset/wt/mp4', session_key.replace(".h5", ".mp4"))
    for z in intervals_video_frmidxs[b]:
        total_clips = len(intervals_video_frmidxs[b][z])
        for clip_idx in np.random.choice(total_clips, min(5, total_clips), replace=False):
            s, e = intervals_dict[b][z][clip_idx]
            s_video, e_video = intervals_video_frmidxs[b][z][clip_idx]
            print(f"s,e= ({s},{e}) s,e_video= ({s_video},{e_video})")
            output_clip_dir = f'{fig_dir}/videos/session{b}/state{z}'
            output_traj_path = f'{fig_dir}/videos/session{b}/state{z}/{s_video}.pdf'
            os.makedirs(output_clip_dir, exist_ok=True)
            video_utils.extract_clip_events(mp4filepath, s_video, e_video, f'{output_clip_dir}/{s_video}.mp4')
            plots.plot_trajectories(model_ckp, data_config, xlim=(s, e), batch=b,
                                    prefix_data='train', savefig=savefig, fig_path=output_traj_path, display=False)

# Analyze mean feedback in each state
# inputs_z, outputs_z = utils.analyze_state_mean(train_stateseq, model_config, train_emissions, train_inputs)

# fig = plot_state_mean_outputs_by_o_dists(model_config, outputs_z, data_config)
# if savefig: fig.savefig(os.path.join(fig_dir, 'state_output_means_by_o_dists.pdf'), bbox_inches='tight', dpi=300)
# if display: plt.show()

# fig = plot_state_mean_outputs_by_o(model_config, outputs_z, data_config)
# if savefig: fig.savefig(os.path.join(fig_dir, 'state_output_means_by_o.pdf'), bbox_inches='tight', dpi=300)
# if display: plt.show()

# fig = plot_state_mean_outputs_by_z(model_config, outputs_z, data_config)
# if savefig: fig.savefig(os.path.join(fig_dir, 'state_output_means_by_z.pdf'), bbox_inches='tight', dpi=300)
# if display: plt.show()

# fig = plot_state_mean_inputs(model_config, inputs_z, data_config)
# if savefig: fig.savefig(os.path.join(fig_dir, 'state_input_means.pdf'), bbox_inches='tight', dpi=300)
# if display: plt.show()
