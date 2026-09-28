"""Deterministic random-dot kinematogram (RDK) stimulus generator."""

import numpy as np

from config.stimuli import (
    RDK_DIRECTIONS_DEG,
    RDK_DOT_DENSITY,
    RDK_DOT_LIFETIME_FRAMES,
    RDK_DOT_SIZE_PX,
    RDK_SPEED_PX_PER_FRAME,
)


def dot_count(image_size, dot_density=RDK_DOT_DENSITY, dot_size_px=RDK_DOT_SIZE_PX):
    """How many dots give `dot_density` coverage of the circular aperture, given square dots of
    dot_size_px x dot_size_px. The paper specifies a density rather than a count, so the count is
    derived here and follows automatically if the image size ever changes."""
    aperture_radius = image_size / 2 - dot_size_px / 2
    aperture_area = np.pi * aperture_radius**2
    return max(1, round(dot_density * aperture_area / dot_size_px**2))


def generate_rdk_block(
    n_frames,
    direction_deg,
    coherence,
    seed=0,
    image_size=64,
    dot_density=RDK_DOT_DENSITY,
    dot_size_px=RDK_DOT_SIZE_PX,
    speed_px_per_frame=RDK_SPEED_PX_PER_FRAME,
    dot_lifetime_frames=RDK_DOT_LIFETIME_FRAMES,
):
    """Generate one RDK stimulus block as a (n_frames, image_size, image_size)
    float32 array in [0, 1] (0 = background, 1 = dot).

    Dot density, size and speed default to the paper's values (see config/stimuli.py). The dot
    count follows from the density and the aperture area rather than being set directly.

    coherence: fraction of dots (redrawn each frame) that move in
        `direction_deg` this frame ("different rule"); the rest take a fresh
        uniformly random direction this frame, at the same speed. Holding the
        noise dots at speed_px_per_frame is what keeps speed from cueing coherence.
    dot_lifetime_frames: a dot is respawned at a random position within the
        aperture after this many frames, so individual dots can't be tracked
        across the whole block. Independently of that, a dot leaving the aperture
        wraps to the far end of the chord it was crossing and continues.
    seed: makes this block reproducible; pass an int, or a
        numpy.random.SeedSequence for use with generate_rdk_blocks.
    """
    if not 0.0 <= coherence <= 1.0:
        raise ValueError("coherence must be in [0, 1]")
    if int(dot_size_px) != dot_size_px or dot_size_px < 1:
        raise ValueError(f"dot_size_px must be a positive integer number of pixels, got {dot_size_px}")
    dot_size_px = int(dot_size_px)

    rng = np.random.default_rng(seed)
    n_dots = dot_count(image_size, dot_density=dot_density, dot_size_px=dot_size_px)
    # keep whole dots inside the aperture: their centre stays a half-dot in from the edge
    aperture_radius = image_size / 2 - dot_size_px / 2
    center = image_size / 2

    def respawn(n):
        r = aperture_radius * np.sqrt(rng.random(n))
        theta = rng.random(n) * 2 * np.pi
        return r * np.cos(theta), r * np.sin(theta)

    x, y = respawn(n_dots)
    # stagger initial ages so dots don't all respawn on the same frame
    age = rng.integers(0, dot_lifetime_frames, n_dots)

    direction_rad = np.deg2rad(direction_deg)
    dx_signal = speed_px_per_frame * np.cos(direction_rad)
    dy_signal = speed_px_per_frame * np.sin(direction_rad)

    frames = np.zeros((n_frames, image_size, image_size), dtype=np.float32)

    for t in range(n_frames):
        is_signal = rng.random(n_dots) < coherence
        noise_theta = rng.random(n_dots) * 2 * np.pi
        dx = np.where(is_signal, dx_signal, speed_px_per_frame * np.cos(noise_theta))
        dy = np.where(is_signal, dy_signal, speed_px_per_frame * np.sin(noise_theta))
        x = x + dx
        y = y + dy
        age = age + 1

        r = np.sqrt(x**2 + y**2)
        outside = r > aperture_radius

        # Wraparound along the dot's own direction of travel: an exiting dot re-enters
        # at the far end of the chord it was crossing, keeping its perpendicular offset
        # to that chord. Every chord is then a 1D periodic domain, which makes the step
        # measure preserving, so a uniform dot density stays uniform.
        if outside.any():
            ux = dx[outside] / speed_px_per_frame
            uy = dy[outside] / speed_px_per_frame
            along = x[outside] * ux + y[outside] * uy
            perp = -x[outside] * uy + y[outside] * ux
            # invariant along the line of travel, and the dot started this frame inside
            # the aperture, so |perp| <= aperture_radius up to floating-point slack
            half_chord = np.sqrt(np.clip(aperture_radius**2 - perp**2, 0.0, None))
            chord = 2 * half_chord
            safe_chord = np.where(chord > 1e-9, chord, 1.0)
            # a dot leaving exactly tangentially has no chord left to travel; park it at
            # the tangent point rather than dividing by zero
            along = np.where(chord > 1e-9, ((along + half_chord) % safe_chord) - half_chord, 0.0)
            x[outside] = along * ux - perp * uy
            y[outside] = along * uy + perp * ux

        # respawn on age alone. Also masking out the dots that wrapped this frame makes the
        # respawn rate position dependent, since those dots all sit at the entry end of their
        # chord, and that reinstates a weaker version of the same density gradient
        aged_out = age >= dot_lifetime_frames
        n_respawn = int(aged_out.sum())
        if n_respawn:
            rx, ry = respawn(n_respawn)
            x[aged_out] = rx
            y[aged_out] = ry
            age[aged_out] = 0

        px = np.clip(np.round(x + center).astype(int), 0, image_size - 1)
        py = np.clip(np.round(y + center).astype(int), 0, image_size - 1)

        frame = np.zeros((image_size, image_size), dtype=np.float32)

        # stamp a dot_size_px x dot_size_px square on each dot's pixel centre, so a dot covers
        # exactly dot_size_px**2 pixels and dot_count's density arithmetic holds
        offsets = range(-(dot_size_px // 2), -(dot_size_px // 2) + dot_size_px)
        for ox in offsets:
            for oy in offsets:
                xx = np.clip(px + ox, 0, image_size - 1)
                yy = np.clip(py + oy, 0, image_size - 1)
                frame[yy, xx] = 1.0
        frames[t] = frame

    return frames


def generate_rdk_blocks(
    n_blocks_per_direction,
    n_frames,
    coherence=1.0,
    base_seed=0,
    directions_deg=RDK_DIRECTIONS_DEG,
    **block_kwargs,
):
    """Deterministically generate `n_blocks_per_direction` RDK blocks for
    each direction in `directions_deg` (all sharing `coherence`).

    Returns an array of shape
    (n_blocks_per_direction * len(directions_deg), n_frames, image_size, image_size).

    Reproducible and composable: seeds are derived from `base_seed` via
    numpy.random.SeedSequence.spawn, so the block for, e.g., direction=90,
    block index 2 is identical whether you ask for 1 direction or all 4, and
    whether you ask for 3 blocks per direction or 10.
    """
    base_seed_seq = np.random.SeedSequence(base_seed)
    blocks = []
    for direction_deg in directions_deg:
        (direction_seed_seq,) = base_seed_seq.spawn(1)
        block_seed_seqs = direction_seed_seq.spawn(n_blocks_per_direction)
        for block_seed_seq in block_seed_seqs:
            blocks.append(
                generate_rdk_block(
                    n_frames=n_frames,
                    direction_deg=direction_deg,
                    coherence=coherence,
                    seed=block_seed_seq,
                    **block_kwargs,
                )
            )
    return np.stack(blocks, axis=0)


if __name__ == "__main__":
    a = generate_rdk_blocks(2, 5, coherence=1.0, base_seed=0)
    b = generate_rdk_blocks(2, 5, coherence=1.0, base_seed=0)
    assert np.array_equal(a, b), "same seed must give bit-identical stimuli"
    c = generate_rdk_blocks(2, 5, coherence=1.0, base_seed=1)
    assert not np.array_equal(a, c), "different seed must give different stimuli"
    print("shape:", a.shape, "reproducibility check passed")
