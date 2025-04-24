import joblib
import numpy as np
import os
import time

from preprocess.leaprig import WT_DATA
from preprocess.new16mic import FREDCLEANED_DATA
from preprocess.preproc_utils import smooth, fill_missing_tracks_SR
from preprocess import visual_features


def get_features(DATA, expt_path, cop_start_frame):

    # Get raw tracks in mm space up to copulation
    fTrx_, mTrx_ = DATA.get_tracks(expt_path, cop_start_frame)
    fly_nodes = DATA.get_fly_nodes()

    # Smooth those raw tracks
    fTrx_ = smooth(fTrx_, DATA.smooth_window)
    mTrx_ = smooth(mTrx_, DATA.smooth_window)

    # Fill missing values
    fTrx = fill_missing_tracks_SR(fTrx_, kind="cubic")
    mTrx = fill_missing_tracks_SR(mTrx_, kind="cubic")   # TODO: PROBABLY DO IT BEFORE SMOOTHING?

    # Compute visual features using various body points
    visual_ftr_dict = visual_features.compute_visual_features(fTrx, mTrx, DATA.get_fly_nodes())

    # Compute tactile features using various body points
    tap_ftr_dict = DATA.get_tap_feature(expt_path, cop_start_frame, mTrx, fTrx)

    all_session_features = dict()
    all_session_features.update(visual_ftr_dict)
    all_session_features.update(tap_ftr_dict)

    # Get auditory bout features
    song = DATA.get_all_song(expt_path)[:cop_start_frame]
    all_session_features['pulse'] = song[:, 0]
    all_session_features['sine'] = song[:, 1]
    all_session_features['mix'] = song[:, 2]
    all_session_features['song'] = ~song[:, 3]
    all_session_features['silence'] = song[:, 3]

    # Get auditory individual pulse and sine features
    isong = DATA.get_all_individual_song(expt_path)[:cop_start_frame]
    all_session_features['pfast_i'] = isong[:, 0]
    all_session_features['pslow_i'] = isong[:, 1]
    all_session_features['sine_i'] = isong[:, 2]
    all_session_features['song_i'] = ~isong[:, 3]
    all_session_features['silence_i'] = isong[:, 3]

    # Compute environmental features, i.e. how far the wall is from female's head
    centerW, _ = DATA.get_circle_estimator_helper(trxM=mTrx_, trxF=fTrx_)
    fHd = fTrx[..., fly_nodes.index('head'), :]
    all_session_features['fDistWall'] = DATA.RADIUS - np.linalg.norm(fHd - np.array(centerW), axis=1)

    all_session_features['fps'] = DATA.fps

    return all_session_features


if __name__ == '__main__':

    DATA = WT_DATA
    # DATA = FREDCLEANED_DATA

    BASE_FOLDER = f'../data/{DATA.dataset}/'
    os.makedirs(BASE_FOLDER, exist_ok=True)

    st1 = time.time()
    # Calculate features from all sessions in a dict and dump
    sessions_features = dict()
    session_paths = DATA.get_session_paths()
    for _, session_path in list(enumerate(session_paths)):
        print(f'Loading expt {_}:', session_path)

        session_name = DATA.get_session_name(session_path)

        if session_name in ['190723_102650_wt_18159211_rig1.1.h5']:
            print("Skipped. usually file open error.")
            continue

        if session_name in ['20190927_151317_left']:
            print("Corrupted tap file.")
            continue

        try:
            cop_frame = DATA.get_copulation_frame(session_path)
        except FileNotFoundError:
            print("No track_occupancy. Skipping this session.")
            continue

        # sessions_features[session_name] = get_features(DATA, session_path, cop_frame)
        try:
            sessions_features[session_name] = get_features(DATA, session_path, cop_frame)
        except RuntimeError as e:
            print(e)
            continue
        except np.linalg.LinAlgError as e:
            print(e)
            continue

        # if _ % 10 == 0:
        #     joblib.dump(sessions_features,
        #                 os.path.join(BASE_FOLDER, f'sessions_features_{len(sessions_features)}.pkl'))

        break

    joblib.dump(sessions_features,
                os.path.join(BASE_FOLDER, f'sessions_features_{len(sessions_features)}.pkl'))
    print(f"Finished computing all features in: {round(time.time() - st1, 2)} "
          f"secs. #sessions: {len(sessions_features)}")
