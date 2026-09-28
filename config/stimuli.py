"""Stimulus condition grids and geometry used across the RSA pipeline.

## Gratings

static_gratings/drifting_gratings are not stored as images in Allen's ophys
NWB files. RSM/allen_data.py's get_activations_sg/
get_activations_dg build the *mouse* response matrix by iterating this exact
grid, in this exact nested order (sf outer / orientation middle / phase inner
for static; tf outer / direction inner for drifting). Values here are picked up by
both the generator and the mouse dataset loader.

Stimulus parameters (Allen Institute, "Allen Brain Observatory -
Visual Coding: Overview" technical white paper, Oct 2018 v.4, and
observatory.brain-map.org/visualcoding/stimulus/{drifting,static}_gratings):
  - drifting gratings: fixed spatial frequency 0.04 cycles/degree, 80%
    contrast, 8 directions (0-315 deg in 45 deg steps), 5 temporal
    frequencies (1, 2, 4, 8, 15 Hz). Each of the 40 conditions is repeated
    15 times, in random order (DRIFTING_NUM_REPEATS).
  - static gratings: 6 orientations (0-150 deg in 30 deg steps), 5 spatial
    frequencies (0.02-0.32 cycles/degree), 4 phases. Contrast is not stated
    on the pages above; we assume 80% to match the drifting-gratings value,
    since both are part of the same stimulus battery. Override if you find
    the real number. Each of the 120 conditions is presented 50 times, in
    random order (STATIC_NUM_REPEATS).
  - these fixed repeat counts are what let RSM/allen_data.py's
    get_activations_dg/get_activations_sg build a (trial, condition, neuron)
    response array, instead of collapsing repeats into a per-
    condition median. get_RSM's multi-session pooling (np.concatenate along
    the neuron axis) and its noise-ceiling bootstrap (which randomly splits
    the trial axis in half) both require that trial axis to exist and be the
    same size across every session being pooled; since Allen's stimulus
    protocol is hardware-triggered and identical across sessions, a fixed
    constant is simpler and safer than recomputing a possibly-inconsistent
    count per session.
  - display geometry: the white paper states the stimulus monitor was a 24"
    screen positioned 10 cm from the mouse's eye (Fig. 3A/B), which computes
    to a horizontal field of view of roughly 120-140 degrees (2*atan(half
    width / distance) for a 24" 16:9 panel). DEFAULT_DEGREES_VISUAL_ANGLE =
    120 is a documented approximation used to convert cycles/degree to
    cycles/image; it doesn't reproduce Allen's spherical screen-warp
    correction (Marshel et al., 2011).

## RDK

Local motion random-dot kinematograms (RDKs) are not part of Allen's stimuli,
but are used by the paper. The RDK generator is datagen/stimuli/rdk.py to produce a (trial, condition, neuron)
response array for a given model. The generator is deterministic and reproducible through seeding.

Dot parameters below are the paper's, from its section 3.3: "random-dot
kinetograms (dot density = 2.5%, dot size = 2 pixels, speed = 5 pixels/frame). In every sample of
the stimulus, a portion of dots move coherently in one of four directions (up, down, right, left),
and the rest move completely randomly." Density is dot pixels as a fraction of the circular
aperture's area, and dot size is read as a diameter, so a dot is a 2x2 block of pixels.

RDK_COHERENCE_LEVELS / RDK_DURATIONS_FRAMES are the evaluation grid, and are the
defaults for the Fig. 5c/5d sweeps in RSM/downstream/runners.py.

What the paper does not specify, and this generator therefore chooses: whether a dot's coherent/
random assignment is fixed for a whole sample or redrawn each frame (here: redrawn each frame),
whether noise dots keep a fixed random direction or take a fresh one each frame (here: fresh each
frame, i.e. the "random walk" noise of Scase, Braddick & Raymond 1996, at the signal speed so that
speed cannot cue coherence), and dot lifetime (here: RDK_DOT_LIFETIME_FRAMES).
"""

from config.training import TrainingConfig


ALLEN_MOVIE_FRAME_RATE_HZ = 30.0  # native frame rate of Allen's ophys movies
TRAINING_FRAME_RATE_HZ = 25.0  # native frame rate of the training movies (UCF101)

# Source frames per model input frame, so an eval stimulus advances at the rate the model was
# trained on. dataset_3d.py samples every TrainingConfig().ds'th UCF101 frame, i.e., one frame per
# ds/25 s; stepping the 30 Hz Allen movie by that same interval takes ds * 30/25 = 3.6 frames.
# Fractional, so movie_frame_indices rounds to the nearest source frame. Shared by every training
# set: moving_animals clips have no physical frame rate, so its checkpoints see the Allen movies at
# this same timing, which keeps the stimulus windows and brain-side blocks identical across datasets.
MOVIE_FRAME_STRIDE = TrainingConfig().ds * ALLEN_MOVIE_FRAME_RATE_HZ / TRAINING_FRAME_RATE_HZ

# Rate the model actually sees frames at, 25/3 = 8.33 Hz. Drifting gratings are rendered against
# it, so a grating's temporal frequency in Hz means the same real
# time as it does in the movie stimuli, and both match training. Note this puts Nyquist at
# 4.17 Hz, so DRIFTING_TEMPORAL_FREQUENCIES_HZ's 8 and 15 Hz conditions alias.
DEFAULT_FRAME_RATE_HZ = TRAINING_FRAME_RATE_HZ / TrainingConfig().ds

STATIC_ORIENTATIONS_DEG = [0, 30, 60, 90, 120, 150]
STATIC_SPATIAL_FREQUENCIES_CPD = [0.02, 0.04, 0.08, 0.16, 0.32]
STATIC_PHASES = [0, 0.25, 0.5, 0.75]
STATIC_DEFAULT_CONTRAST = 0.8  # assumed
STATIC_NUM_REPEATS = 50

DRIFTING_DIRECTIONS_DEG = [0, 45, 90, 135, 180, 225, 270, 315]
DRIFTING_TEMPORAL_FREQUENCIES_HZ = [1, 2, 4, 8, 15]
DRIFTING_SPATIAL_FREQUENCY_CPD = 0.04  # fixed for the whole drifting-gratings battery
DRIFTING_DEFAULT_CONTRAST = 0.8
DRIFTING_NUM_REPEATS = 15

DEFAULT_DEGREES_VISUAL_ANGLE = 120.0

RDK_DIRECTIONS_DEG = (0, 90, 180, 270)
RDK_DOT_DENSITY = 0.025  # dot pixels / aperture area
RDK_DOT_SIZE_PX = 2  # dot diameter, so a dot covers RDK_DOT_SIZE_PX**2 pixels
RDK_SPEED_PX_PER_FRAME = 5.0
RDK_DOT_LIFETIME_FRAMES = 5  # not specified by the paper
RDK_COHERENCE_LEVELS = (0.1, 0.3, 0.5, 0.7, 1.0)
RDK_DURATIONS_FRAMES = (5, 10, 20, 30, 40)
