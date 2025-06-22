import glob
import sys

import joblib
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import importlib
import matplotlib
import time

from scipy.stats import zscore

from trig_avg_utils import *
# connected_components1d, merge_intervals, skip_intervals_too_close, get_nanstd_angle, get_nanmean_angle
from leaprig import WT_DATA
# from trig_avg_utils import split_name
# from turning.feature_attrs import *

# matplotlib.rcParams['pdf.fonttype'] = 42
# matplotlib.rcParams['ps.fonttype'] = 42
# matplotlib.rcParams["font.size"] = 18


def skip_overlapping_events(event_name, a_event_lims, all_session_features):
    a_event_name, event_var = split_name(event_name)

    if event_var == '':
        return a_event_lims
    elif event_var == 'noTap':
        assert a_event_name in audio_feature_list
        b = all_session_features['tap2']
    elif event_var == 'noSong':
        assert a_event_name in mech_feature_list
        b = all_session_features['song']
    else:
        raise Exception(f'Variation {event_var} not supported.')

    b = np.abs(b.astype(int))
    event_lims_ = []
    for ai, aj in a_event_lims:
        if (ai - DATA.event_pre < 0) or (aj+DATA.event_post >= len(b)):
            continue
        if np.sum(b[ai-DATA.event_pre:aj+DATA.event_post]) > 0:
            continue
        event_lims_.append([ai, aj])
    event_lims_ = np.array(event_lims_)
    print("event_lims_", len(a_event_lims), len(event_lims_))
    return event_lims_


def get_event_intervals(event_seq):
    print(event_seq, event_seq.shape,  np.sum(event_seq))
    event_lims = connected_components1d(event_seq, return_limits=True)
    print("event_lims cc", event_lims.shape)
    if not event_lims.any():
        return np.array([])

    event_lims = merge_intervals(event_lims, DATA.MERGE_BOUTS_CLOSER_THAN)  # merge together event bouts separated by less than MERGE_BOUTS_CLOSER_THAN
    print("event_lims merged", event_lims.shape)
    event_lims = skip_intervals_too_close(event_lims, DATA.MIN_GAP_BW_BOUTS)  # remove event bouts closer to each other than MIN_GAP_BW_BOUTS
    print("event_lims skipped", event_lims.shape)
    event_bout_lengths = (event_lims[:, 1] - event_lims[:, 0])
    event_lims = event_lims[event_bout_lengths <= MAX_EVENT_BOUT_LEN]
    return event_lims


def get_event_windows(event_lims, event_window, before_event_window, session_len):
    event_bout_lengths = event_lims[:, 0]
    event_bout_lengths = event_bout_lengths[(event_bout_lengths > -before_event_window.min()) & (event_bout_lengths < (session_len - event_window.max()))]
    event_windows = event_window + event_bout_lengths.reshape(-1, 1)
    before_event_windows = before_event_window + event_bout_lengths.reshape(-1, 1)
    return event_windows, before_event_windows


def get_speed_side_mask(all_session_features, feat, event_windows):

    event_bout_onsets = event_windows[:, event_window == 0].squeeze()

    # filter events by male location at the time of onset
    side_filter = True
    if 'sLeft' in feat:     # male on left at song onset. Could be singing wing but thorax is fine for now.
        side_filter = (all_session_features['fmAng'][event_bout_onsets] < 0)
    elif 'sRight' in feat:     # male on right at song onset
        side_filter = (all_session_features['fmAng'][event_bout_onsets] > 0)
    elif 'tLeft' in feat:     # male on left at tap onset
        side_filter = (all_session_features['tap2'][event_bout_onsets] < 0)
    elif 'tRight' in feat:     # male on right at tap onset
        side_filter = (all_session_features['tap2'][event_bout_onsets] > 0)

    # filter events by female walking "speed" just before onset
    speed_justbefore_onset = np.mean(
        np.abs(all_session_features['fV'])[event_windows[:, (event_window >= -DATA.event_pre//2) & (event_window <= 0)]] * DATA.fps, axis=1)
    speed_filter = (speed_justbefore_onset >= THRESHOLD_SPEED)

    # remove events with tracking errors (big delta fThetas)
    traces = all_session_features['fTheta'][event_windows]
    unwrapped_traces = np.unwrap(traces, period=360, axis=1)
    diffunwrapped_traces = np.abs(np.diff(unwrapped_traces, axis=1))
    error_events_mask = np.sum(diffunwrapped_traces > 150, axis=1) < 1  # even if there's a single change in orientation > 150, remove that event

    # remove events where female is too close to the walls at event onsets
    wall_mask = all_session_features['fDistWall'][event_bout_onsets] >= 3   # mm

    mask = side_filter & speed_filter & error_events_mask & wall_mask
    print("mask", mask.shape, np.sum(mask))
    return mask


def get_event_seq(event_name, all_session_features):
    base_event_name, event_var = split_name(event_name)
    if base_event_name == 'silence':
        event_seq = ~all_session_features['song']
    else:
        event_seq = all_session_features[base_event_name]

    if base_event_name in trans_velocity_feature_list:
        print("all_session_features[event_name]:", all_session_features[base_event_name].shape)
        diff = np.diff(event_seq, prepend=0)
        if event_var == '':
            diff = np.abs(diff)
        elif event_var == 'pos':
            diff_p99 = np.percentile(diff[diff > 0], q=99)
            diff = diff > diff_p99  # positive diff events
        elif event_var == 'neg':
            diff_p99 = np.percentile(np.abs(diff[diff < 0]), q=99)
            diff = diff < -diff_p99  # negative diff events
        else:
            raise Exception(f'Variation {event_var} not supported for event {base_event_name}.')
        event_seq = diff
        print("in event seq:", event_seq, event_seq.shape, np.sum(event_seq))
    # else:
    #     raise Exception(f'{event_name}-triggered average not supported.')

    event_lims = get_event_intervals(event_seq)

    if (base_event_name in mech_feature_list) or (base_event_name in audio_feature_list):
        if event_var in ['noTap', 'noSong']:
            event_lims = skip_overlapping_events(event_name, event_lims, all_session_features)
    return event_lims


def get_desired_feature_traces(all_session_features, feat, event_windows, before_event_windows):

    base_feat_name = split_name(feat)[0]
    if base_feat_name == 'silence':
        feat_series = ~all_session_features['song']
    else:
        feat_series = all_session_features[base_feat_name]
    traces = feat_series[event_windows]

    mask = get_speed_side_mask(all_session_features, feat, event_windows)
    filtered_traces = traces[mask].squeeze()

    if filtered_traces.ndim <= 1:  # if only one event
        filtered_traces = filtered_traces.reshape(1, -1)

    if (base_feat_name in trans_velocity_feature_list) or (base_feat_name in trans_acc_feature_list):
        traces = filtered_traces * DATA.fps
        if MEAN_SUBTRACTED:
            before_traces = feat_series[before_event_windows][mask].squeeze()
            if before_traces.ndim <= 1:
                before_traces = before_traces.reshape(1, -1)
            mean_pre_value = np.mean(before_traces, axis=1).reshape(-1, 1)
            traces = traces - mean_pre_value
        if ZSCORED:
            traces = zscore(traces, axis=1)
    elif base_feat_name in rot_velocity_feature_list:
        traces = filtered_traces * DATA.fps
    elif base_feat_name in dist_feature_list:
        traces = filtered_traces
        if ZSCORED:
            traces = zscore(traces, axis=1)
    elif base_feat_name in orientation_feature_list:
        unwrapped_traces = np.unwrap(filtered_traces, period=360, axis=1)
        # subtract_value = unwrapped_traces[:, 0].reshape(-1, 1)
        # subtract_value = unwrapped_traces[:, event_window == -DATA.event_pre//4]
        subtract_value = unwrapped_traces[:, event_window == 0]
        traces = unwrapped_traces - subtract_value
    elif base_feat_name in audio_feature_list:
        traces = filtered_traces
    elif base_feat_name in mech_feature_list:
        traces = filtered_traces
    else:
        raise Exception(f'feat {feat} not supported.')
    return traces


def get_ETA(event, sessions_features, desired_feats):
    """
    event-triggered average of "desired_feat" traces
    """

    all_traces = {feat: [] for feat in desired_feats}

    # get traces
    for s in sessions_features:
        all_session_features = sessions_features[s]
        print("all_session_features keys", all_session_features.keys())

        event_lims = get_event_seq(event, all_session_features)
        if not len(event_lims):
            print("No valid event bouts found. Skipping this session.")
            continue

        session_len = all_session_features['fFV'].shape[0]
        event_windows, before_event_windows = get_event_windows(event_lims, event_window, before_event_window, session_len)
        for feat in desired_feats:
            traces = get_desired_feature_traces(all_session_features, feat, event_windows, before_event_windows)
            print(f'traces:', traces.shape)
            all_traces[feat].append(traces)
    all_traces = {feat: np.vstack(all_traces[feat]) for feat in desired_feats}
    print([all_traces[feat].shape for feat in desired_feats])

    # calculate ETA from traces
    ETAs = {}
    ETA_stds = {}
    ETA_sems = {}
    for feat in desired_feats:
        base_feat_name = feat.split('_')[0]  # base feature name
        if base_feat_name in velocity_feature_list:
            ETA = np.nanmean(all_traces[feat], axis=0)
            ETA_std = np.nanstd(all_traces[feat], axis=0)
        elif base_feat_name in orientation_feature_list:
            ETA = np.degrees(get_nanmean_angle(np.radians(all_traces[feat]), axis=0))
            ETA_std = np.degrees(get_nanstd_angle(np.radians(all_traces[feat]), axis=0))
        elif (base_feat_name in audio_feature_list) or (base_feat_name in mech_feature_list):
            sum_feat = np.sum(np.abs(all_traces[feat]), axis=0)     # abs for feats from left or right side.
            ETA = sum_feat / np.sum(sum_feat)   # probability
            ETA_std = np.zeros(ETA.shape)
        else:
            raise Exception(f'{base_feat_name} not supported.')
        ETA_sem = 1.96 * ETA_std / np.sqrt(all_traces[feat].shape[0])   # 95% CI
        ETAs[feat] = ETA
        ETA_stds[feat] = ETA_std
        ETA_sems[feat] = ETA_sem
    return ETAs, ETA_stds, ETA_sems, all_traces


def plot_ETAs(event, ETAs, ETA_stds, ETA_sems, all_traces, plot_ETA_TRACES=False, plot_ETA_STD=False, plot_ETA_SEM=False):
    """
    Plot Event-Triggered Averages (ETAs)
    """
    for feat in ETAs:
        base_feat_name = feat.split('_')[0]  # base feature name
        ETA = ETAs[feat]
        ETA_std = ETA_stds[feat]
        ETA_sem = ETA_sems[feat]

        plot_traces = all_traces[feat]

        alpha = 0.03
        if 'fTheta' in base_feat_name:  # plot only a sample in case of orientation traces
            plot_traces = plot_traces[np.random.choice(len(plot_traces), min(len(plot_traces), 100))]
            alpha = 0.1

        # plot all traces, ETA and ETA_std
        plt.figure(figsize=(7, 5))
        if plot_ETA_TRACES:
            plt.plot(event_window/DATA.fps, plot_traces.T, alpha=alpha)
        plt.plot(event_window/DATA.fps, ETA, c='b', label='ETA')
        if plot_ETA_STD:
            plt.fill_between(event_window/DATA.fps, ETA-ETA_std, ETA+ETA_std, color='r', interpolate=True, alpha=0.3)
        elif plot_ETA_TRACES or plot_ETA_SEM:
            plt.fill_between(event_window / DATA.fps, ETA - ETA_sem, ETA + ETA_sem, color='b', interpolate=True, alpha=0.3)

        units = units_dict[base_feat_name]
        if base_feat_name in velocity_feature_list:
            # plt.ylim(-10, 10)
            if MEAN_SUBTRACTED:
                plt.ylabel(f'mean-subtracted {feat}\n({units})')
            elif ZSCORED:
                units = 'a.u.'
                plt.ylabel(f'zscored {feat}\n({units})')
            else:
                plt.ylabel(f'{feat}\n({units})')
        elif base_feat_name in orientation_feature_list:
            plt.ylim(-20, 20)
            plt.ylabel(f'{feat}\n({units})')
        elif base_feat_name in audio_feature_list:
            plt.ylabel(f'Pdf({feat})\n')
        elif base_feat_name in mech_feature_list:
            plt.ylabel(f'Pdf({feat})\n')

        plt.axvline(0, ls=':', c='r')
        plt.axhline(0, ls=':', c='b')
        plt.title(f'[{DATA.dataset}] {event}-triggered average (n={all_traces[feat].shape[0]})')
        plt.gcf().text(0.1, 1,  s=f'female speed before onset>={THRESHOLD_SPEED}mm/s', size='x-small')
        plt.xlabel(f'time since {event} onset (secs)')
        plt.legend(loc='lower right')

        plt.tight_layout()
        folder = f'{BASE_FOLDER}/ETAs/threshold_speed={THRESHOLD_SPEED}/event={event}_test2'
        os.makedirs(folder, exist_ok=True)
        # plt.savefig(f'{folder}/'
        #             f'{feat}_eta_maxboutlen={MAX_EVENT_BOUT_LEN}_meansubt={MEAN_SUBTRACTED}_zscored={ZSCORED}'
        #             f'_plot_ETA_TRACES={plot_ETA_TRACES}_plot_ETA_STD={plot_ETA_STD}_plot_ETA_SEM={plot_ETA_SEM}.pdf',
        #             dpi=120, bbox_inches='tight')
        plt.show()
        plt.close()
    return


DATA = WT_DATA

if __name__ == '__main__':

    event_window = DATA.event_window
    before_event_window = DATA.before_event_window

    BASE_FOLDER = f'../data/{DATA.dataset}'

    # Load features for all sessions from pre-computed file
    sessions_features = joblib.load(f'{BASE_FOLDER}/sessions_features_75_may30.pkl')

    MEAN_SUBTRACTED = False
    ZSCORED = True
    THRESHOLD_SPEED = 0     # mm/s

    events = [
        'tap2',
        'tap2_noSong',

        'song',
        'song_noTap',

        # 'mix',
        # 'pulse',
        # 'sine',
        # 'silence',
        #
        # 'mix_noTap',
        # 'sine_noTap',
        # 'pulse_noTap',
        # 'silence_noTap',
        #
        # 'mFV_pos', 'fFV_pos',
        # 'mFV_neg', 'fFV_neg',
        # 'mLS_pos', 'fLS_pos',
        # 'mLS_neg', 'fLS_neg',
    ]

    desired_feats = [
        # 'tap2',
        # 'song',
        'mFV',
        'fFV',
        'fLS',
        # 'mLS',
        # 'fFA',
        # 'mFA',
        'mfDist',
        # 'pulse', 'sine', 'mix', 'silence',
        # 'fLV', 'mLV',
        # 'fLV_tLeft', 'fLV_tRight',
        'fTheta_tLeft', 'fTheta_tRight',
        # 'fLV_sLeft', 'fLV_sRight',
        'fTheta_sLeft', 'fTheta_sRight',
    ]
    for e in events:
        print(f"> Getting {e}-TAs now....")
        s_ = time.time()
        MAX_EVENT_BOUT_LEN = DATA.get_max_event_bout_len(e)

        # compute ETAs
        ETAs, ETA_stds, ETA_sems, all_traces = get_ETA(e, sessions_features, desired_feats)

        # dump ETAs
        folder = f'{BASE_FOLDER}/ETAs/'
        os.makedirs(folder, exist_ok=True)
        joblib.dump([ETAs, ETA_stds, ETA_sems, all_traces], f'{folder}/{e}-eta_traces_unzscored.pkl')

        # plot ETAs
        # plot_ETAs(e, ETAs, ETA_stds, ETA_sems, all_traces)
        # plot_ETAs(e, ETAs, ETA_stds, ETA_sems, all_traces, plot_ETA_TRACES=True)
        # plot_ETAs(e, ETAs, ETA_stds, ETA_sems, all_traces, plot_ETA_STD=True, plot_ETA_TRACES=True)
        # plot_ETAs(e, ETAs, ETA_stds, ETA_sems, all_traces, plot_ETA_SEM=True)
        print(f"{e}-TAs: completed in: {round(time.time() - s_, 2)} secs.")
        print('-----------------------------------------------')
