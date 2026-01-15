import glob
import os
import numpy as np
import scipy.io as sio

from preprocess import preproc_utils
from preprocess.preproc_utils import h5read


class FREDCLEANED_DATA:

    # data and rig constants
    fps = 60
    smooth_window = 2
    PIXEL_TO_MM = 18.56
    RADIUS = 21
    dataset = 'wt_fredcleaned'

    copulation_frame_idxs = {
        '20190927_161548_right': 12750,
        '20191001_114431_left': 25540,
        '20191002_105826_left': 90155,
        '20191002_120717_right': 103050,
        '20191015_135433_right': 46744,
    }

    @classmethod
    def get_tracks(cls, expt_path, cop_start_frame):
        fTrx = h5read(expt_path, '/tracks')[1, ..., :cop_start_frame].T / cls.PIXEL_TO_MM
        mTrx = h5read(expt_path, '/tracks')[0, ..., :cop_start_frame].T / cls.PIXEL_TO_MM
        return fTrx, mTrx

    @staticmethod
    def get_session_name(session_path):
        return session_path.split("/")[-3]

    @staticmethod
    def get_copulation_frame(session_path):
        session = FREDCLEANED_DATA.get_session_name(session_path)
        cop_frame = FREDCLEANED_DATA.copulation_frame_idxs.get(session, None)
        print("Getting copulation frame:", session, "cop_frame=", cop_frame)
        return cop_frame

    @staticmethod
    def get_copulation_bool_from_session(session, session_len):
        return session in FREDCLEANED_DATA.copulation_frame_idxs

    @staticmethod
    def get_circle_estimator_helper(trxM=None, trxF=None):
        return preproc_utils.circle_estimator_helper(trxM, trxF)

    @staticmethod
    def get_session_paths():
        # return sorted(list(glob.glob("/Volumes/murthy/usingla/fred_wt_data/**/**/000000.mp4.inference.000_000000.analysis.h5")))
        return sorted(list(glob.glob('/Volumes/fileset-mmurthy/usingla/fred_data/**/**/000000.mp4.inference.000_000000.analysis.h5')))

    @staticmethod
    def get_all_song(expt_path):
        """
        Credits: Talmo Pereira
        """
        def __load_song__(fname):
            seg = sio.loadmat(fname)

            # Bout sample limits.
            bout_lims = seg["bInf"]["stEn"][0][0]
            pulse_bouts = bout_lims[np.where(seg["bInf"]["Type"][0][0] == "Pul")[0]]
            mix_bouts = bout_lims[np.where(seg["bInf"]["Type"][0][0] == "Mix")[0]]
            sine_bouts = bout_lims[np.where(seg["bInf"]["Type"][0][0] == "Sin")[0]]

            return pulse_bouts, mix_bouts, sine_bouts

        n_frames = h5read(expt_path, "/track_occupancy").squeeze().shape[0]
        s1, s2, s3 = __load_song__(os.path.join(os.path.dirname(expt_path), 'sInf_params_new16mic.mat'))

        vfaas = h5read(os.path.join(os.path.dirname(expt_path), 'data.mat'), "/vfaas").squeeze()
        frame_at_sample = vfaas

        # Reconstruct masks.
        s1 = preproc_utils.lims_to_mask(frame_at_sample[s1], n_frames)
        s2 = preproc_utils.lims_to_mask(frame_at_sample[s2], n_frames)
        s3 = preproc_utils.lims_to_mask(frame_at_sample[s3], n_frames)
        silence = ~(s1 | s2 | s3)
        all_song = np.stack([s1, s2, s3, silence], axis=1)  # pulse, mix, sine, then silence
        return all_song

    @staticmethod
    def get_all_individual_song(expt_path):
        """TODO: Locate individual song elements for new16mic data"""
        return FREDCLEANED_DATA.get_all_song(expt_path)

    @staticmethod
    def get_tap_feature(expt_path, cop_start_frame, mTrx=None, fTrx=None, fTheta=None):
        d = dict()
        d['tap2'] = (np.load(os.path.join(os.path.dirname(expt_path), 'cropped_predictions.npy')) > 0.9)[:cop_start_frame]
        # d['tap2'] = d['tap']
        return d

    @staticmethod
    def get_fly_nodes():
        fly_nodes = [
            "head",
            "thorax",
            "wingL",
            "wingR",
        ]
        return fly_nodes
