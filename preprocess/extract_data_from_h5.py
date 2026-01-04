"""

"""

import sys

import joblib
import numpy as np
import os
import time

from preprocess.leaprig import WT_DATA, AC_BOTH, AC_LEFT, AC_RIGHT, BLIND_BOTH
from preprocess.new16mic import FREDCLEANED_DATA
from preprocess.preproc_utils import fill_missing_tracks_SR, halfgaussian_filter
from preprocess import visual_features, female_features


def get_raw_track_data(DATA, expt_path, cop_start_frame):
    # Get raw tracks in mm space up to copulation
    fTrx_, mTrx_ = DATA.get_tracks(expt_path, cop_start_frame)
    fly_nodes = DATA.get_fly_nodes()
    data = {
        'fTrx': fTrx_,
        'mTrx': mTrx_,
        'fly_nodes': fly_nodes,
    }
    return data


def get_smoothed_raw_track_data(raw_track_data):
    data = dict()
    for p in raw_track_data:
        print(p)
        # Fill missing values
        fTrx_ = fill_missing_tracks_SR(raw_track_data[p]['fTrx'], kind="cubic")
        mTrx_ = fill_missing_tracks_SR(raw_track_data[p]['mTrx'], kind="cubic")

        # Smooth those raw tracks
        fTrx = halfgaussian_filter(fTrx_, sigma=2)
        mTrx = halfgaussian_filter(mTrx_, sigma=2)

        data[p] = {
            'fTrx': fTrx,
            'mTrx': mTrx,
            'fly_nodes': raw_track_data[p]['fly_nodes'],
        }

    return data


def get_features(DATA, expt_path, cop_start_frame):

    # Get raw tracks in mm space up to copulation
    fTrx_, mTrx_ = DATA.get_tracks(expt_path, cop_start_frame)
    fly_nodes = DATA.get_fly_nodes()

    # Fill missing values
    fTrx_ = fill_missing_tracks_SR(fTrx_, kind="cubic")
    mTrx_ = fill_missing_tracks_SR(mTrx_, kind="cubic")

    # Smooth those raw tracks
    fTrx = halfgaussian_filter(fTrx_, sigma=2)
    mTrx = halfgaussian_filter(mTrx_, sigma=2)

    all_session_features = dict()

    # Compute visual features using various body points
    visual_ftr_dict = visual_features.compute_visual_features(fTrx, mTrx, fly_nodes)
    all_session_features.update(visual_ftr_dict)

    # Compute tactile features using various body points
    tap_ftr_dict = DATA.get_tap_feature(expt_path, cop_start_frame, mTrx, fTrx)
    all_session_features.update(tap_ftr_dict)

    # Compute wing midpoint angles using various body points
    male_wingmidpoint_ftr_dict = female_features.compute_male_wing_midpoint_features(fTrx, mTrx, fly_nodes)
    all_session_features.update(male_wingmidpoint_ftr_dict)

    # Compute wing flick features using various body points
    female_wing_flick_ftr_dict = female_features.compute_wing_flick_features(fTrx, fly_nodes)
    all_session_features.update(female_wing_flick_ftr_dict)

    # Compute fending (or leg extension) features using various body points, NOT available for wt_fred
    # female_fending_ftr_dict = female_features.compute_fending_features(fTrx, fly_nodes)
    # all_session_features.update(female_fending_ftr_dict)

    # Get auditory bout features
    song = DATA.get_all_song(expt_path)[:cop_start_frame]
    all_session_features['pulse'] = song[:, 0]
    all_session_features['mix'] = song[:, 1]
    all_session_features['sine'] = song[:, 2]
    all_session_features['song'] = ~song[:, 3]
    all_session_features['silence'] = song[:, 3]

    # Get auditory individual pulse and sine features
    isong = DATA.get_all_individual_song(expt_path)[:cop_start_frame]
    all_session_features['pulse_i'] = isong[:, 0]   # no individual mix or pslow
    all_session_features['sine_i'] = isong[:, 2]
    all_session_features['song_i'] = ~isong[:, 3]
    all_session_features['silence_i'] = isong[:, 3]

    # Compute environmental features, i.e. how far the wall is from female's head
    centerW, _ = DATA.get_circle_estimator_helper(trxM=mTrx, trxF=fTrx)
    fHd = fTrx[..., fly_nodes.index('head'), :]
    all_session_features['fDistWall'] = DATA.RADIUS - np.linalg.norm(fHd - np.array(centerW), axis=1)

    all_session_features['fps'] = DATA.fps

    return all_session_features


if __name__ == '__main__':

    # DATA = WT_DATA
    # DATA = AC_BOTH
    DATA = FREDCLEANED_DATA

    BASE_FOLDER = f'../../data/{DATA.dataset}/'
    os.makedirs(BASE_FOLDER, exist_ok=True)

    raw_track_data = joblib.load(os.path.join(BASE_FOLDER, 'raw_track_data_11_jan1.pkl'))
    smoothed_track_data = get_smoothed_raw_track_data(raw_track_data)
    # for s in raw_track_data:
    #     nan_count = np.count_nonzero(np.isnan(raw_track_data[s]['fTrx']))
    #     siz = raw_track_data[s]['fTrx'].size
    #     print(s, nan_count, raw_track_data[s]['fTrx'].size, nan_count/siz * 100)
    joblib.dump(smoothed_track_data, os.path.join(BASE_FOLDER, f'smoothed_track_data_{len(raw_track_data)}_jan1.pkl'))
    sys.exit(0)

    st1 = time.time()
    # Calculate features from all sessions in a dict and dump
    sessions_features = dict()
    raw_track_data = dict()
    session_paths = DATA.get_session_paths()
    for _, session_path in list(enumerate(session_paths)):
        print(f'Loading expt {_}:', session_path)

        session_name = DATA.get_session_name(session_path)

        # if session_name in ['190723_102650_wt_18159211_rig1.1.h5']:
        #     print("Skipped. usually file open error.")
        #     continue

        if session_name in ['20190927_151317_left']:
            print("Corrupted tap file.")
            continue

        try:
            cop_frame = DATA.get_copulation_frame(session_path)
        except FileNotFoundError:
            print("No track_occupancy. Skipping this session.")
            continue

        try:
            sessions_features[session_name] = get_features(DATA, session_path, cop_frame)
            raw_track_data[session_name] = get_raw_track_data(DATA, session_path, cop_frame)    # does not have song or tap data
        except RuntimeError as e:
            print(e)
            continue
        except np.linalg.LinAlgError as e:
            print(e)
            continue
        except OSError as e:
            print(e)
            continue

        if len(raw_track_data) % 10 == 0:
            joblib.dump(sessions_features, os.path.join(BASE_FOLDER, f'sessions_features_{len(sessions_features)}_jan1.pkl'))
            joblib.dump(raw_track_data, os.path.join(BASE_FOLDER, f'raw_track_data_{len(raw_track_data)}_jan1.pkl'))
            print(f'Temp dump.')

    joblib.dump(sessions_features, os.path.join(BASE_FOLDER, f'sessions_features_{len(sessions_features)}_jan1.pkl'))
    joblib.dump(raw_track_data, os.path.join(BASE_FOLDER, f'raw_track_data_{len(raw_track_data)}_jan1.pkl'))
    print(f"Finished computing all features in: {round(time.time() - st1, 2)}secs. #sessions: {len(raw_track_data)}")
