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


DATA & FILE OVERVIEW

1. File List:
There is one folder per female condition; males are wild type in all conditions. Each
   condition folder has two subfolders, with one file per recording (one female-male pair).

   wt/               Wild-type females (Dataset 1 in the paper). 82 recordings; 81 pairs
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


METHODOLOGICAL INFORMATION (collection/generation/processing of data): Methods of Singla et al. (2026).

FILE-SPECIFIC INFORMATION FOR: <condition>/h5/<session>.h5
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


See the github repository for more info on how the code uses these data files:
https://github.com/umeshksingla/femaleflystates
