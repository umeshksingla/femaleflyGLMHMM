import joblib


def process_and_dump_sessions():
    from preprocess.leaprig import WT_DATA
    from preprocess.new16mic import FREDCLEANED_DATA
    from preprocess.extract_data_from_h5 import fill_missing_tracks_SR, smooth

    # DATA = WT_DATA
    DATA = FREDCLEANED_DATA

    # Get raw tracks in mm space up to copulation
    session_paths = DATA.get_session_paths()
    ftracks = {}
    mtracks = {}

    for s, s_path in enumerate(session_paths):
        print(f"On session {s}: {s_path}")
        session_name = DATA.get_session_name(s_path)

        if session_name in ['20190927_151317_left']:
            print("Corrupted tap file.")
            continue

        try:
            cop_frame = DATA.get_copulation_frame(s_path)
        except FileNotFoundError:
            print("No track_occupancy. Skipping this session.")
            continue

        try:
            fTrx_, mTrx_ = DATA.get_tracks(s_path, cop_start_frame=cop_frame)
        except RuntimeError:
            print(f"Error opening file {s}: {s_path}")
            continue

        # Smooth those raw tracks
        fTrx_ = smooth(fTrx_, DATA.smooth_window)
        mTrx_ = smooth(mTrx_, DATA.smooth_window)

        # Fill missing values
        fTrx = fill_missing_tracks_SR(fTrx_, kind="cubic")
        mTrx = fill_missing_tracks_SR(mTrx_, kind="cubic")  # TODO: PROBABLY DO IT BEFORE SMOOTHING?

        ftracks[s_path] = fTrx
        mtracks[s_path] = mTrx
        print("==")
        if s%5 == 0:
            joblib.dump(ftracks, f'{DATA.dataset}_processed_ftracks.pkl')
            joblib.dump(mtracks, f'{DATA.dataset}_processed_mtracks.pkl')

    joblib.dump(ftracks, f'{DATA.dataset}_processed_ftracks.pkl')
    joblib.dump(mtracks, f'{DATA.dataset}_processed_mtracks.pkl')
    return


process_and_dump_sessions()
