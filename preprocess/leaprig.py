import glob
import numpy as np
import h5py

from preprocess import visual_features, tactile_features
from preprocess.preproc_utils import h5read
from preprocess import preproc_utils


class WT_DATA:

    # data and rig constants
    fps = 150
    smooth_window = 5
    PIXEL_TO_MM = 30.3
    RADIUS = 15
    dataset = 'wt'

    @staticmethod
    def get_session_name(session_path):
        return session_path.split("/")[-1]

    @staticmethod
    def get_copulation_frame(session_path):
        tracking_h5_path = session_path.replace('/h5/', '/trackingh5/').replace('.h5', '.tracking.h5')
        track_occupancy = h5read(tracking_h5_path, 'track_occupancy')
        cop_start_frame = np.where(np.sum(track_occupancy, axis=1) == 2)[0][-1] + 1
        return cop_start_frame

    @staticmethod
    def get_circle_estimator_helper(trxM=None, trxF=None):
        trxM = trxM[..., ::-1]
        trxF = trxF[..., ::-1]
        return preproc_utils.circle_estimator_helper(trxM, trxF)

    @classmethod
    def get_tracks(cls, expt_path, cop_start_frame):
        fTrx = h5read(expt_path, 'trxF') / cls.PIXEL_TO_MM
        mTrx = h5read(expt_path, 'trxM') / cls.PIXEL_TO_MM
        print("Session length:", len(fTrx), "Copulation start frame", cop_start_frame)
        return fTrx[:cop_start_frame], mTrx[:cop_start_frame]

    @staticmethod
    def get_session_paths():
        return list(glob.glob('/Volumes/murthy/usingla/gold_dataset/wt/h5/*.h5'))

    @staticmethod
    def get_all_song(expt_path):
        """
        Return sine, mix and pulse bouts for whole session
        """
        with h5py.File(expt_path, "r") as f:
            n_frames = f["trxF"].shape[0]

            # Load.
            frame_at_sample = f["frame_at_sample"][:]
            s1 = f["pulse_bouts"][:]
            s2 = f["sine_bouts"][:]
            s3 = f["mix_bouts"][:]

        # Reconstruct masks.
        s1 = preproc_utils.lims_to_mask(frame_at_sample[s1], n_frames)
        s2 = preproc_utils.lims_to_mask(frame_at_sample[s2], n_frames)
        s3 = preproc_utils.lims_to_mask(frame_at_sample[s3], n_frames)
        silence = ~(s1 | s2 | s3)
        all_song = np.stack([s1, s2, s3, silence], axis=1)
        return all_song

    @staticmethod
    def get_all_individual_song(expt_path):
        """
        Return individual sines and pulses for whole session
        """
        with h5py.File(expt_path, "r") as f:
            n_frames = f["trxF"].shape[0]

            # Load.
            frame_at_sample = f["frame_at_sample"][:]
            s1 = f["pfast_lims"][:].astype(int)
            s2 = f["pslow_lims"][:].astype(int)
            s3 = f["sine_lims"][:].astype(int)

        # Reconstruct masks.
        s1 = preproc_utils.lims_to_mask(frame_at_sample[s1], n_frames)
        s2 = preproc_utils.lims_to_mask(frame_at_sample[s2], n_frames)
        s3 = preproc_utils.lims_to_mask(frame_at_sample[s3], n_frames)
        silence = ~(s1 | s2 | s3)
        all_individual_song = np.stack([s1, s2, s3, silence], axis=1)
        return all_individual_song

    @staticmethod
    def get_tap_feature(expt_path=None, cop_start_frame=None, mTrx=None, fTrx=None):
        d = dict()
        d['tap'] = tactile_features.compute_mechanical_features_v1(mTrx, fTrx, WT_DATA.get_fly_nodes())['touch']
        tap_feat = tactile_features.compute_mechanical_features_v2(mTrx, fTrx, WT_DATA.get_fly_nodes())
        d['tap2'] = tap_feat['tapClosestDist'] <= 0.0  # unsigned tap
        # d['tap2_directed'] = (tap_feat['tapClosestDist'] <= 0.0) * np.sign(tap_feat['tapClosestAng'])  # signed tap
        return d

    @staticmethod
    def get_fly_nodes():
        fly_nodes = [
            "head",
            "thorax",
            "abdomen",
            "wingL",
            "wingR",
            "forelegL4",
            "forelegR4",
            "midlegL4",
            "midlegR4",
            "hindlegL4",
            "hindlegR4",
            "eyeL",
            "eyeR",
        ]
        return fly_nodes


class AC_BOTH(WT_DATA):
    dataset = 'ac_both'

    @staticmethod
    def get_session_paths():
        return list(glob.glob('/Volumes/murthy/usingla/gold_dataset/ac_both/h5/*.h5'))


class AC_LEFT(WT_DATA):
    dataset = 'ac_left'

    @staticmethod
    def get_session_paths():
        return list(glob.glob('/Volumes/murthy/usingla/gold_dataset/ac_left/h5/*.h5'))


class AC_RIGHT(WT_DATA):
    dataset = 'ac_right'

    @staticmethod
    def get_session_paths():
        return list(glob.glob('/Volumes/murthy/usingla/gold_dataset/ac_right/h5/*.h5'))


class BLIND_BOTH(WT_DATA):
    dataset = 'blind_both'

    @staticmethod
    def get_session_paths():
        return list(glob.glob('/Volumes/murthy/usingla/gold_dataset/both_eye_blind/h5/*.h5'))
