"""Synthetic drifting/static grating stimuli, matching the Allen Brain
Observatory's own drifting_gratings/static_gratings stimulus tables.

Unlike natural_scenes/natural_movie_*, Allen's ophys NWB files do not contain
pre-rendered image stacks for gratings. Gratings are procedurally rendered at
display time in the original experiment, and only their parameters (orientation,
spatial_frequency, phase for static; orientation, temporal_frequency for drifting)
are stored, in a stimulus table. Images generated here at runtime. Gratings are
deterministic given their parameters.
"""

import numpy as np
from scipy.ndimage import gaussian_filter

from config.stimuli import (
    DEFAULT_DEGREES_VISUAL_ANGLE,
    DEFAULT_FRAME_RATE_HZ,
    DRIFTING_DEFAULT_CONTRAST,
    DRIFTING_DIRECTIONS_DEG,
    DRIFTING_SPATIAL_FREQUENCY_CPD,
    DRIFTING_TEMPORAL_FREQUENCIES_HZ,
    STATIC_DEFAULT_CONTRAST,
    STATIC_ORIENTATIONS_DEG,
    STATIC_PHASES,
    STATIC_SPATIAL_FREQUENCIES_CPD,
)


# Target amplitude (fraction of original) that the anti-aliasing filter
# leaves at the final grid's Nyquist frequency (0.5 cycles/final-pixel).
# ~0.05 (~ -26 dB) is aggressive enough to prevent a coherent alias while
# still preserving most of the amplitude of legitimate content well below
# Nyquist. See the derivation in _antialiasing_sigma_hires.
NYQUIST_ATTENUATION_TARGET = 0.05


def _antialiasing_sigma_hires(supersample, nyquist_attenuation=NYQUIST_ATTENUATION_TARGET):
    """Gaussian blur sigma (in supersampled-pixel units) to apply before
    decimating a `supersample`x-oversampled render back down to the final
    grid, so that content at/above the final grid's Nyquist frequency is
    suppressed rather than aliased into a spurious lower frequency.

    A Gaussian spatial kernel with standard deviation sigma has frequency
    response exp(-2 pi^2 sigma^2 f^2) (f in cycles/pixel, same pixel units as
    sigma). Solving for the sigma (in final-pixel units) that attenuates the
    final Nyquist frequency (f=0.5) down to `nyquist_attenuation`, then
    converting to supersampled-pixel units (sigma scales with resolution):
        sigma_final = sqrt(-ln(nyquist_attenuation) / (2 pi^2 * 0.5^2))
        sigma_hires = sigma_final * supersample
    """
    sigma_final = np.sqrt(-np.log(nyquist_attenuation) / (2 * np.pi**2 * 0.5**2))
    return sigma_final * supersample


def _grating_frame(orientation_deg, cycles_per_pixel, phase_cycles, image_size, contrast, supersample=4):
    """Render at `supersample`x resolution, low-pass filter, then decimate
    back down to image_size. At image_size resolution, high spatial frequencies (the
    top of STATIC_SPATIAL_FREQUENCIES_CPD, at DEFAULT_DEGREES_VISUAL_ANGLE/64px,
    work out to ~0.6 cycles/pixel, i.e., above the 0.5 cycles/pixel Nyquist limit for that grid)
    would point-sample into a lower-frequency alias rather than the intended grating. Rendering
    at supersample=4x makes the same content ~0.15 cycles/pixel at the higher-res grid, and a
    Gaussian blur tuned to the final grid's Nyquist frequency (see _antialiasing_sigma_hires) removes
    the content that can't be represented at image_size before the final decimation.
    """
    theta = np.deg2rad(orientation_deg)
    hi_res = image_size * supersample
    xs = np.arange(hi_res) - hi_res / 2
    x, y = np.meshgrid(xs, xs)
    projected = x * np.cos(theta) + y * np.sin(theta)
    # cycles_per_pixel is defined relative to the final image_size grid; each
    # supersampled pixel is 1/supersample as wide, so the frequency in
    # supersampled-pixel units is scaled down to match.
    wave = np.cos(2 * np.pi * (cycles_per_pixel / supersample) * projected - 2 * np.pi * phase_cycles)
    hi_res_img = 127.5 + 127.5 * contrast * wave
    hi_res_img = gaussian_filter(hi_res_img, sigma=_antialiasing_sigma_hires(supersample), mode="wrap")
    return hi_res_img[::supersample, ::supersample]


def generate_static_grating(
    orientation_deg,
    spatial_frequency_cpd,
    phase,
    image_size=64,
    contrast=STATIC_DEFAULT_CONTRAST,
    degrees_visual_angle=DEFAULT_DEGREES_VISUAL_ANGLE,
    supersample=4,
):
    """A single static grating frame, shape (image_size, image_size), values in [0, 255]."""
    cycles_per_pixel = spatial_frequency_cpd * (degrees_visual_angle / image_size)
    return _grating_frame(orientation_deg, cycles_per_pixel, phase, image_size, contrast, supersample=supersample)


def generate_drifting_grating_block(
    direction_deg,
    temporal_frequency_hz,
    n_frames,
    image_size=64,
    spatial_frequency_cpd=DRIFTING_SPATIAL_FREQUENCY_CPD,
    contrast=DRIFTING_DEFAULT_CONTRAST,
    degrees_visual_angle=DEFAULT_DEGREES_VISUAL_ANGLE,
    frame_rate_hz=DEFAULT_FRAME_RATE_HZ,
    supersample=4,
):
    """n_frames of a grating drifting in `direction_deg` at `temporal_frequency_hz`,
    shape (n_frames, image_size, image_size), values in [0, 255]."""
    cycles_per_pixel = spatial_frequency_cpd * (degrees_visual_angle / image_size)
    frames = np.empty((n_frames, image_size, image_size), dtype=np.float32)
    for t in range(n_frames):
        phase_cycles = temporal_frequency_hz * (t / frame_rate_hz)
        frames[t] = _grating_frame(
            direction_deg, cycles_per_pixel, phase_cycles, image_size, contrast, supersample=supersample
        )
    return frames


def generate_static_grating_conditions(
    image_size=64,
    orientations_deg=STATIC_ORIENTATIONS_DEG,
    spatial_frequencies_cpd=STATIC_SPATIAL_FREQUENCIES_CPD,
    phases=STATIC_PHASES,
    **kwargs,
):
    """Yields one static-grating image per condition, in the same (sf outer,
    orientation middle, phase inner) order as allen_data.get_activations_sg,
    so condition i here == condition i of that function's response matrix.
    """
    for sf in spatial_frequencies_cpd:
        for orientation_deg in orientations_deg:
            for phase in phases:
                yield generate_static_grating(orientation_deg, sf, phase, image_size=image_size, **kwargs)


def generate_drifting_grating_conditions(
    n_frames,
    image_size=64,
    directions_deg=DRIFTING_DIRECTIONS_DEG,
    temporal_frequencies_hz=DRIFTING_TEMPORAL_FREQUENCIES_HZ,
    **kwargs,
):
    """Yields one drifting-grating block (n_frames, image_size, image_size) per
    condition, in the same (tf outer, direction inner) order as
    allen_data.get_activations_dg, so condition i here == condition i
    of that function's response matrix.
    """
    for tf in temporal_frequencies_hz:
        for direction_deg in directions_deg:
            yield generate_drifting_grating_block(direction_deg, tf, n_frames, image_size=image_size, **kwargs)


if __name__ == "__main__":
    static = list(generate_static_grating_conditions())
    assert len(static) == len(STATIC_ORIENTATIONS_DEG) * len(STATIC_SPATIAL_FREQUENCIES_CPD) * len(STATIC_PHASES)
    drifting = list(generate_drifting_grating_conditions(n_frames=5))
    assert len(drifting) == len(DRIFTING_DIRECTIONS_DEG) * len(DRIFTING_TEMPORAL_FREQUENCIES_HZ)
    assert drifting[0].shape == (5, 64, 64)
    print(f"{len(static)} static conditions, {len(drifting)} drifting conditions. Shapes OK")
