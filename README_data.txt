This femaleflystates_readme.txt file was generated on 2026-11-11 by Umesh Singla


GENERAL INFORMATION

1. Title of Dataset: Body-part tracking and song annotations of courting Drosophila melanogaster
   pairs with wild-type, deaf or blind females

2. Author Information
	A. Principal Investigator Contact Information
		Name: Mala Murthy
		Institution: Princeton Neuroscience Institute, Princeton University
		Address: Princeton, NJ 08544, USA
		Email: mmurthy@princeton.edu

	B. Alternate Contact Information
		Name: Albert Lin
		Institution: Princeton Neuroscience Institute, Princeton University
		Address: Princeton, NJ 08544, USA
		Email: albertl@princeton.edu


SHARING/ACCESS INFORMATION

1. Licenses/restrictions placed on the data: [FILL: license chosen in the PDC submission form, e.g. CC BY 4.0].
    The analysis code is licensed separately (see the code repository).

2. Links to publications that cite or use the data:
    Singla U, et al. (2026). Latent state dynamics underlying female behavior during Drosophila courtship interactions.
    [FILL: DOI or preprint link]

3. Links to other publicly accessible locations of the data: None

4. Links/relationships to ancillary data sets: Pose estimation with SLEAP (Pereira et al. 2022, Nature Methods)

5. Was data derived from another source? No


DATASET 1 OVERVIEW (folder "gold_dataset")

1. File List:
There is one folder per female condition; males are wild type in all conditions. Each
   condition folder has two subfolders, with one file per recording (one female-male pair).

   wt/               Wild-type females (Dataset 1 in the paper) recorded on LEAP rig. 82 recordings; 81 pairs
                     in the paper.
   ac_both/          Deaf females. 32 recordings; 31 pairs in the paper.
   both_eye_blind/   Blind females. 10 recordings; 10 pairs in the paper.

     h5/<session>.h5                    Body-part tracks of the female and the male, and song
                                        annotations.
     trackingh5/<session>.tracking.h5   SLEAP tracking output for the same recording [CHECK].
                                        The analysis code uses it only to find copulation onset.
     mp4/<session>.mp4                  session raw video
     slp/<session>.slp                  SLEAP tracked files for the same recording. (unused)

   <session> starts with the recording date and time (YYMMDD_HHMMSS), for example 190723_102650_wt_18159211_rig1.1


2. Relationship between files, if important:

- <session>.h5 and <session>.tracking.h5 describe the same recording, and their frames are
     aligned
- In <session>.h5, the tracks are indexed by video frame (150 fps) and the song annotations by
     audio sample. frame_at_sample gives the video frame for each audio sample.
- Copulation onset is the frame after the last frame in which both flies are tracked (rows of
     track_occupancy that sum to 2). Recordings last up to 30 min (270,000 frames); a recording
     whose pre-copulation part is shorter than that ended in copulation. Files may continue
     past copulation onset; the analyses use only the frames before it.


File-specific information for Dataset 1: <condition>/h5/<session>.h5
All of the data are stored in self-describing HDF5 files.
Every file in wt/h5/, ac_both/h5/ and both_eye_blind/h5/ has this format.

1. Only the following variables were used from <condition>/h5/<session>.h5 files and features computed.
   - /trxF, /trxM: one row per video frame (up to 270,000 = 30 min at 150 fps)
   - frame_at_sample: one value per audio sample
   - *_bouts, *_lims: one row per song bout or song element

2. Variable List:
   trxF             float, (n_frames, 13, 2), pixels
                    Female body-part positions. Axis 0 = video frame. Axis 1 = body part, in this
                    order: head, thorax, abdomen, wingL, wingR, forelegL4, forelegR4, midlegL4,
                    midlegR4, hindlegL4, hindlegR4, eyeL, eyeR (L/R = left/right; the leg nodes
                    are leg tips). Axis 2 = image coordinates (x, y).
   trxM             float, (n_frames, 13, 2), pixels
                    Male body-part positions, same layout as trxF.
   frame_at_sample  integer, (n_samples,)
                    Video frame index at each audio sample. Converts the song
                    limits below to video frames: frame_at_sample[pulse_bouts] is (n_bouts, 2).
   pulse_bouts      integer, (n_bouts, 2), audio samples: start and end of each pulse-only bout
   sine_bouts       integer, (n_bouts, 2), audio samples: start and end of each sine-only bout
   mix_bouts        integer, (n_bouts, 2), audio samples: start and end of each bout with both
                    pulse and sine song
   pfast_lims       (n_pulses, 2), audio samples: start and end of each fast pulse (Pfast)
   pslow_lims       (n_pulses, 2), audio samples: start and end of each slow pulse (Pslow)
   sine_lims        (n_segments, 2), audio samples: start and end of each sine song segment



DATASET 2 OVERVIEW (folder "new16mic_dataset")

1. File List:
   Wild-type females (Dataset 2 in the paper), recorded on the 16-microphone rig
    of Roemschied et al. (2023). 11 recordings; 11 pairs in the
    paper (the analysis code skips 20190927_151317_left, whose tap file is
    corrupted). One folder per recording, new16mic_dataset/<session>/<chamber>/, with:

     000000.mp4.inference.000_000000.analysis.h5   SLEAP tracks of the male and the female
     sInf_params_new16mic.mat                      song segmentation (song bouts)
     data.mat                                      video frame at each audio sample
     cropped_predictions.npy                       per-frame tap probabilities
     000000.mp4                                    raw video

   Here <session> is the recording date and time (YYYYMMDD_HHMMSS) and the chamber (left or
   right), for example 20190927_161548_right; <subfolder> is the <chamber>.


2. Relationship between files, if important:

- The files in one recording's folder go together. The tracks and the tap
     probabilities are indexed by video frame and the song bouts by audio sample
     (10 kHz); /vfaas in data.mat gives the video frame for each audio sample.
- Copulation was scored manually from the videos. Five recordings ended in
     copulation, at these video frames (only the frames before them are analyzed).
     The other recordings did not end in copulation.


File-specific information for Dataset 2: new16mic_dataset/<session>/<subfolder>/
Every recording folder in new16mic_dataset/ has these files.

1. Only the following variables were used from these files and features computed.
   - /tracks, /track_occupancy (000000.mp4.inference.000_000000.analysis.h5): one entry per
     video frame
   - bInf.stEn, bInf.Type (sInf_params_new16mic.mat): one row per song bout
   - /vfaas (data.mat): one value per audio sample
   - cropped_predictions.npy: one value per video frame

2. Variable List:
   000000.mp4.inference.000_000000.analysis.h5 (HDF5; SLEAP analysis export)
   /tracks          float, (2, 2, 4, n_frames), pixels
                    Body-part positions. Axis 0 = fly (0 = male, 1 = female). Axis 1 = image
                    coordinates (x, y). Axis 2 = body part, in this order: head, thorax, wingL,
                    wingR (wing tips). Axis 3 = video frame. NaN where a body part was not
                    detected. tracks[1].T is the female's (n_frames, 4, 2) array, laid out like
                    trxF.

   sInf_params_new16mic.mat
   bInf.stEn        (n_bouts, 2), audio samples: start and end of each song bout
   bInf.Type        (n_bouts,): bout type, Pul (pulse only), Sin (sine only) or Mix (both
                    pulse and sine song)

   data.mat
   /vfaas           one value per audio sample (a MATLAB vector; squeeze it): the video frame
                    at that sample. Converts the bout limits to video frames, as
                    frame_at_sample does for the Dataset 1 files.

   cropped_predictions.npy
                    one value per video frame: the probability, from a convolutional network,
                    that the male is tapping the female. The analysis code counts frames above
                    0.9 as taps.


METHODOLOGICAL INFORMATION (collection/generation/processing of data): Methods of Singla et al. (2026).

See the github repository for more info on how the code uses these data files:
https://github.com/umeshksingla/femaleflystates
