"""Fetches mouse visual cortex responses from the Allen Brain Observatory and turns them into
representational similarity matrices (RSMs), the brain-side counterpart to the model RSMs
computed in deep_models.py. get_RSM is the main entry point: for a given Cre line,
cortical area/depth, and stimulus type, it pulls per-neuron trial responses across all matching
experiments, averages over random trial subsets, and returns a distribution of RSMs.
It also estimates a split-half noise ceiling: the best similarity any model could possibly
achieve against this (noisy, finite-trial) brain data.
ImagingDepth may name one cortical depth or several.
"""

import itertools

import allensdk
import numpy as np
import scipy as sp
from allensdk.brain_observatory.brain_observatory_exceptions import (
    NoEyeTrackingException,
)
from allensdk.core.brain_observatory_cache import BrainObservatoryCache

from config.allen import ALLEN_MANIFEST_FILE
from config.depths import normalize_depths
from config.rsm import RSM_COMPARISON_MEASURE
from config.stimuli import (
    DRIFTING_DIRECTIONS_DEG,
    DRIFTING_NUM_REPEATS,
    DRIFTING_TEMPORAL_FREQUENCIES_HZ,
    STATIC_NUM_REPEATS,
    STATIC_ORIENTATIONS_DEG,
    STATIC_PHASES,
    STATIC_SPATIAL_FREQUENCIES_CPD,
)

# from oasis_preprocessing import deconvolve_calcium_unknown
from RSM.similarity import cka_kernel, SSM_MEASURES, compute_rsms, center_activations

print(f"allen sdk version is {allensdk.__version__}")

# absolute, so which cache this finds doesn't depend on the working directory
boc = BrainObservatoryCache(manifest_file=ALLEN_MANIFEST_FILE)
# targeted_structures = boc.get_all_targeted_structures()
# imaging_depths = boc.get_all_imaging_depths()

# the query every entry point below defaults to: the paper's Cre line, area and depth. Held as
# tuples and copied into a list per call, so no caller can mutate another call's defaults.
DEFAULT_CRE_LINES = ("Cux2-CreERT2",)
DEFAULT_TARGETED_STRUCTS = ("VISp",)
DEFAULT_IMAGING_DEPTHS = (175,)


def _resolve_query(CreLine, TargetedStruct, ImagingDepth):
    """Fills in the defaults above for whichever of the three a caller left as None. allensdk's
    query methods want lists, so each default becomes a fresh list rather than the shared tuple."""
    return (
        list(DEFAULT_CRE_LINES) if CreLine is None else CreLine,
        list(DEFAULT_TARGETED_STRUCTS) if TargetedStruct is None else TargetedStruct,
        list(DEFAULT_IMAGING_DEPTHS) if ImagingDepth is None else ImagingDepth,
    )


def get_RSM(
    CreLine=None,
    TargetedStruct=None,
    ImagingDepth=None,
    StimType="natural_scenes",
    downsample_rate=1,
    pool_sessions=True,
    comparison_measures=None,
    session_types=None,
):
    # comparison_measures: which of config.rsm.RSM_COMPARISON_MEASURES to compute the
    # split-half noise ceiling for; defaults to config.rsm.RSM_COMPARISON_MEASURE
    #
    # session_types: which imaging sessions to pool over, or None for all of them. Only matters
    # for natural_movie_one, which is shown in sessions A, B and C: pooling all three concatenates
    # the same cells two or three times onto the neuron axis (Allen re-images one container's cells
    # across its sessions), inflating the apparent population and narrowing any bootstrap that
    # resamples neurons. Passing config.movies.single_session_for's value restricts a movie to one
    # session per container and removes those duplicates. Every other stimulus lives in a single
    # session already, so None and the filter agree there.
    if comparison_measures is None:
        comparison_measures = RSM_COMPARISON_MEASURE
    CreLine, TargetedStruct, ImagingDepth = _resolve_query(CreLine, TargetedStruct, ImagingDepth)
    # get the number of exp containers (length of num_exps) and the number of
    # experiments within each container (each element of num_exps)
    num_exps = get_number_experiments(
        CreLine=CreLine,
        TargetedStruct=TargetedStruct,
        ImagingDepth=ImagingDepth,
        StimType=StimType,
        session_types=session_types,
    )
    num_exp_containers = len(num_exps)

    num_stim_conditions = get_num_stim_conditions(StimType=StimType)
    totalexp = 0
    if pool_sessions:
        num_iter = 100
        print(num_stim_conditions)
        all_RSM = np.ndarray(
            (
                int(num_stim_conditions * downsample_rate),
                int(num_stim_conditions * downsample_rate),
                num_iter,
            )
        )
        # multiplied by downsample_rate
        # in the case the movie data is downsampled
        # (only matters for the movie data)
        for eccounter in range(num_exp_containers):
            for expcounter in range(num_exps[eccounter]):
                print(eccounter, expcounter)
                data_set, events = get_one_dataset(
                    CreLine=CreLine,
                    TargetedStruct=TargetedStruct,
                    ImagingDepth=ImagingDepth,
                    StimType=StimType,
                    ExpContainerIdx=eccounter,
                    ExpIdx=expcounter,
                    session_types=session_types,
                )

                if StimType == "natural_scenes":
                    activations = get_activations_natuscene(data_set, events)

                elif StimType == "drifting_gratings":
                    activations = get_activations_dg(data_set)

                elif StimType == "static_gratings":
                    activations = get_activations_sg(data_set)

                elif (
                    (StimType == "natural_movie_one")
                    | (StimType == "natural_movie_two")
                    | (StimType == "natural_movie_three")
                ):
                    activations = get_activations_nm(data_set, StimType, downsample_rate, events)

                if totalexp == 0:
                    V1_tmp = activations["V1"]
                else:
                    print("V1_tmp shape is ", V1_tmp.shape)
                    # Different sessions can have slightly different repeat
                    # counts for grating/natural-scene stimuli, so the trial
                    # axis (axis 0) isn't guaranteed to match across sessions
                    # even though it matches within one session's response
                    # array.
                    if V1_tmp.shape[0] != activations["V1"].shape[0]:
                        num_repeats = min(V1_tmp.shape[0], activations["V1"].shape[0])
                        print(
                            f"[warning] trial count differs between sessions "
                            f"({V1_tmp.shape[0]} vs {activations['V1'].shape[0]}); "
                            f"truncating both to {num_repeats}"
                        )
                        V1_tmp = V1_tmp[:num_repeats]
                        activations["V1"] = activations["V1"][:num_repeats]
                    V1_tmp = np.concatenate((V1_tmp, activations["V1"]), axis=2)

                del activations
                totalexp = totalexp + 1

        num_neurons = V1_tmp.shape[2]
        num_trials = V1_tmp.shape[0]
        print("total no. neurons =", num_neurons)

        noise_ceiling = estimate_RSM_noise_ceiling(V1_tmp, num_trials, comparison_measures)

        # raw per-iteration trial-averaged activations (not just their RSMs), so callers can
        # compare model activations against this same iteration's brain data via CKA as well
        # as via the RSM built from it below
        all_activations = np.ndarray((int(num_stim_conditions * downsample_rate), num_neurons, num_iter))

        for itercount in range(num_iter):
            trials_idx_permute = np.random.permutation(np.arange(0, num_trials))
            which_trials = trials_idx_permute[0 : int(num_trials / 2)]

            this_V1 = np.mean(V1_tmp[which_trials, :, :], axis=0)
            all_activations[:, :, itercount] = this_V1
            activations = {"V1": this_V1}
            activations_centered = center_activations(activations)
            sim_mats = compute_rsms(activations_centered)
            RSM_v1 = sim_mats["V1"]
            all_RSM[:, :, itercount] = RSM_v1

    elif not pool_sessions:
        # not populated in this branch: no caller currently consumes per-session raw
        # activations here (only the pool_sessions=True path feeds compare_reps.py's CKA path)
        noise_ceiling = None
        V1_tmp = None
        all_activations = None
        all_RSM = np.ndarray(
            (
                int(num_stim_conditions * downsample_rate),
                int(num_stim_conditions * downsample_rate),
                sum(num_exps),
            )
        )  # multiplied by downsample_rate in the case the movie data is downsampled (only matters for the movie data)
        for eccounter in range(num_exp_containers):
            for expcounter in range(num_exps[eccounter]):
                print(eccounter, expcounter)
                data_set, events = get_one_dataset(
                    CreLine=CreLine,
                    TargetedStruct=TargetedStruct,
                    ImagingDepth=ImagingDepth,
                    StimType=StimType,
                    ExpContainerIdx=eccounter,
                    ExpIdx=expcounter,
                    session_types=session_types,
                )

                if StimType == "natural_scenes":
                    activations = get_activations_natuscene(data_set, events)

                elif StimType == "drifting_gratings":
                    activations = get_activations_dg(data_set)

                elif StimType == "static_gratings":
                    activations = get_activations_sg(data_set)

                elif (
                    (StimType == "natural_movie_one")
                    | (StimType == "natural_movie_two")
                    | (StimType == "natural_movie_three")
                ):
                    activations = get_activations_nm(data_set, StimType, downsample_rate, events)

                sim_mats = compute_rsms(activations)
                RSM_v1 = sim_mats["V1"]
                all_RSM[:, :, totalexp] = RSM_v1
                totalexp = totalexp + 1

    return all_RSM, all_activations, noise_ceiling, V1_tmp


def estimate_RSM_noise_ceiling(activations, num_trial, comparison_measures=None):
    """activations is T (number of trials) x M (number of stimuli) x N (number of neurons)
    Split-half reliability: randomly split trials in half, average each half separately, and
    compare the two halves to each other under each of comparison_measures (see
    config/rsm.py). Since both halves are estimates of the same underlying representation,
    their agreement upper-bounds how well any model could possibly match this data, given the
    trial-to-trial noise in the neural responses."""
    if comparison_measures is None:
        comparison_measures = RSM_COMPARISON_MEASURE
    num_trials = activations.shape[0]
    print(activations[:, 0, 0])
    num_repetitions = num_trial
    noise_ceiling = {m: np.empty([num_repetitions, 1]) for m in comparison_measures}
    for iter in range(num_repetitions):
        trials_idx_permute = np.random.permutation(np.arange(0, num_trials))
        which_trials = trials_idx_permute[0 : int(num_trials / 2)]
        other_trials = np.setdiff1d(np.arange(0, num_trials), which_trials)

        responses1 = activations[which_trials, :, :]
        responses2 = activations[other_trials, :, :]
        activation1 = {"V1": np.mean(responses1, axis=0)}
        sim_mats1 = compute_rsms(activation1)
        activation2 = {"V1": np.mean(responses2, axis=0)}
        sim_mats2 = compute_rsms(activation2)

        RSM1 = sim_mats1["V1"]
        RSM2 = sim_mats2["V1"]
        for m in comparison_measures:
            if m == "cka":
                noise_ceiling[m][iter] = cka_kernel(responses1.mean(0), responses2.mean(0))
            else:
                noise_ceiling[m][iter] = compare_two_RSMs(RSM1, RSM2, measure=m)

    return noise_ceiling


def compare_two_RSMs(RSM1, RSM2, measure="kendall"):
    return SSM_MEASURES[measure](RSM1, RSM2)


def get_number_experiments(
    CreLine=None,
    TargetedStruct=None,
    ImagingDepth=None,
    StimType="natural_scenes",
    session_types=None,
):
    CreLine, TargetedStruct, ImagingDepth = _resolve_query(CreLine, TargetedStruct, ImagingDepth)
    # every container at any of the requested depths, so a multi-depth ImagingDepth yields one
    # longer container list that get_RSM pools over rather than two separate analyses
    all_ecs = boc.get_experiment_containers(
        cre_lines=CreLine,
        targeted_structures=TargetedStruct,
        imaging_depths=normalize_depths(ImagingDepth),
    )
    num_exp_containers = len(all_ecs)
    print("number of {} experiment containers: {}\n".format(", ".join(CreLine), num_exp_containers))
    num_exps = []
    for eccounter in range(num_exp_containers):
        ec_id = all_ecs[eccounter]["id"]
        exps = boc.get_ophys_experiments(
            experiment_container_ids=[ec_id], stimuli=[StimType], session_types=session_types
        )

        num_exps.append(len(exps))
        print(f"experiment container: {ec_id}\n: {len(exps)}")

    return num_exps


def get_one_dataset(
    CreLine=None,
    TargetedStruct=None,
    ImagingDepth=None,
    StimType="natural_scenes",
    ExpContainerIdx=0,
    ExpIdx=0,
    session_types=None,
):
    CreLine, TargetedStruct, ImagingDepth = _resolve_query(CreLine, TargetedStruct, ImagingDepth)
    # same query (and so the same container ordering) as get_number_experiments, which is what
    # makes ExpContainerIdx mean the same container in both
    all_ecs = boc.get_experiment_containers(
        cre_lines=CreLine,
        targeted_structures=TargetedStruct,
        imaging_depths=normalize_depths(ImagingDepth),
    )
    all_ec_id = all_ecs[ExpContainerIdx]["id"]
    # same session_types as get_number_experiments, so ExpIdx indexes the same filtered list
    exp = boc.get_ophys_experiments(
        experiment_container_ids=[all_ec_id], stimuli=[StimType], session_types=session_types
    )[ExpIdx]
    data_set = boc.get_ophys_experiment_data(exp["id"])
    events = boc.get_ophys_experiment_events(exp["id"])

    return data_set, events


# condition counts below (118 images, 900/3600 movie frames) are fixed properties of the
# Allen Brain Observatory's standard stimulus set, not derived from any experiment's data
def get_num_stim_conditions(StimType="natural_scenes"):
    if StimType == "static_gratings":
        num_stim_conditions = len(STATIC_ORIENTATIONS_DEG) * len(STATIC_SPATIAL_FREQUENCIES_CPD) * len(STATIC_PHASES)
        return num_stim_conditions

    elif StimType == "drifting_gratings":
        num_stim_conditions = len(DRIFTING_DIRECTIONS_DEG) * len(DRIFTING_TEMPORAL_FREQUENCIES_HZ)
        return num_stim_conditions

    elif StimType == "natural_scenes":
        numImages = 118
        num_stim_conditions = numImages
        return num_stim_conditions

    elif StimType == "natural_movie_one" or StimType == "natural_movie_two":
        num_stim_conditions = 900
        return num_stim_conditions

    elif StimType == "natural_movie_three":
        num_stim_conditions = 3600
        return num_stim_conditions


def _condition_groups(stim_table, columns, value_grids):
    """Rows of stim_table for each stimulus condition, as a list of (start, end) index arrays in
    the nested order `value_grids` gives (outermost value list first).
    """
    groups = []
    for combo in itertools.product(*value_grids):
        mask = np.ones(len(stim_table), dtype=bool)
        for column, value in zip(columns, combo, strict=True):
            mask &= (stim_table[column] == value).to_numpy()
        rows = stim_table[mask]
        groups.append((rows["start"].to_numpy(), rows["end"].to_numpy()))
    return groups


def _mean_response_per_trial(traces, groups, num_repeats):
    """(num_neurons, num_conditions, num_repeats) mean response, averaging each trial's frame
    window of every neuron's trace at once.

    traces is the whole session's (num_neurons, num_frames) dF/F array, so each condition/trial
    fills a whole neuron column in one np.nanmean instead of one scalar call per neuron.
    """
    responses = np.empty([traces.shape[0], len(groups), num_repeats])
    for condition, (starts, ends) in enumerate(groups):
        for tr in range(num_repeats):
            responses[:, condition, tr] = np.nanmean(traces[:, int(starts[tr]) : int(ends[tr])], axis=1)
    return responses


def get_activations_sg(data_set):
    """get_activations_sg/dg/natuscene/nm all follow the same shape: for each stimulus condition,
    average every neuron's response trace over that trial's frame window, and pack the results
    into a {"V1": array[trials, conditions, neurons]} dict (the "V1" key is just this pipeline's
    generic activation-dict convention, matching deep_models.py's model activations).
    They differ only in which stimulus table/timing columns each Allen stimulus type uses."""
    stim_table = data_set.get_stimulus_table("static_gratings")
    all_cell_ids = data_set.get_cell_specimen_ids()
    print("there are " + str(len(all_cell_ids)) + " neurons in this session")

    groups = _condition_groups(
        stim_table,
        ("spatial_frequency", "orientation", "phase"),
        (STATIC_SPATIAL_FREQUENCIES_CPD, STATIC_ORIENTATIONS_DEG, STATIC_PHASES),
    )
    num_repeats = min(len(starts) for starts, _ in groups)
    if num_repeats < STATIC_NUM_REPEATS:
        print(
            f"[warning] session has only {num_repeats} repeats for some static_gratings "
            f"conditions (nominal is {STATIC_NUM_REPEATS}); using {num_repeats} for all conditions"
        )

    _, traces = data_set.get_dff_traces(cell_specimen_ids=list(all_cell_ids))
    responses = _mean_response_per_trial(traces, groups, num_repeats)

    print(responses.shape)

    activations = {"V1": np.transpose(responses)}

    return activations


def get_activations_dg(data_set):
    stim_table = data_set.get_stimulus_table("drifting_gratings")
    all_cell_ids = data_set.get_cell_specimen_ids()
    print("there are " + str(len(all_cell_ids)) + " neurons in this session")

    groups = _condition_groups(
        stim_table,
        ("temporal_frequency", "orientation"),
        (DRIFTING_TEMPORAL_FREQUENCIES_HZ, DRIFTING_DIRECTIONS_DEG),
    )
    num_repeats = min(len(starts) for starts, _ in groups)
    if num_repeats < DRIFTING_NUM_REPEATS:
        print(
            f"[warning] session has only {num_repeats} repeats for some drifting_gratings "
            f"conditions (nominal is {DRIFTING_NUM_REPEATS}); using {num_repeats} for all conditions"
        )

    _, traces = data_set.get_dff_traces(cell_specimen_ids=list(all_cell_ids))
    responses = _mean_response_per_trial(traces, groups, num_repeats)

    print(responses.shape)

    activations = {"V1": np.transpose(responses)}

    return activations


def get_activations_natuscene(data_set, events):
    """get activations for natural scenes stimuli."""
    stim_table = data_set.get_stimulus_table("natural_scenes")
    all_cell_ids = data_set.get_cell_specimen_ids()
    num_neurons = len(all_cell_ids)
    print("there are " + str(num_neurons) + " neurons in this session")
    numImages = 118
    numTrials = 50

    responses = np.empty([num_neurons, numImages, numTrials])

    # filtered once per image, not once per (neuron, image). frame -1 is the blank sweep, so
    # range(numImages) covers exactly the real images
    frame_starts = [stim_table[stim_table.frame == im]["start"].to_numpy() for im in range(numImages)]
    for imcounter in range(numImages):
        starts = frame_starts[imcounter]
        for tr in range(numTrials):
            # events are deconvolved spikes, summed over a fixed 15-frame window from onset;
            # sliced across all neurons at once
            start = int(starts[tr])
            responses[:, imcounter, tr] = np.sum(events[:, start : start + 15], axis=1)

    print(responses.shape)

    activations_deconv = {"V1": np.transpose(responses)}

    return activations_deconv


def get_activations_nm(data_set, which_movie, downsample_rate, events):
    """get activations for natural movie stimuli."""
    stim_table = data_set.get_stimulus_table(which_movie)
    try:
        _timestamps, _locations = data_set.get_pupil_location()

    except NoEyeTrackingException:
        print(f"No eye tracking for experiment {data_set.get_metadata()['ophys_experiment_id']}.")

    all_cell_ids = data_set.get_cell_specimen_ids()
    num_neurons = len(all_cell_ids)
    print("there are " + str(num_neurons) + " neurons in this session")

    movie_len = len(stim_table[stim_table.repeat == 0])
    print("movie length is " + str(movie_len))
    numTrials = 10
    responses = np.empty([num_neurons, int(movie_len * downsample_rate), numTrials])

    for ncounter in range(num_neurons):
        sample_cell = events[ncounter, :]

        response_tmp = np.empty([max(stim_table.repeat) + 1, int(movie_len * downsample_rate)])
        for tr in range(max(stim_table.repeat) + 1):
            start_time = stim_table.start[stim_table.repeat == tr] + 0  # 4
            this_cell_resp = sample_cell[start_time]
            if downsample_rate != 1:
                this_cell_resp_ds = downsample_movie_data(this_cell_resp, downsample_rate)
            else:
                this_cell_resp_ds = this_cell_resp

            response_tmp[tr, :] = this_cell_resp_ds
        responses[ncounter, :, :] = np.transpose(response_tmp)

    print(responses.shape)

    activations = {"V1": np.transpose(responses)}

    return activations


def downsample_movie_data(response, rate):
    """Fourier-domain resampling (not simple decimation): resamples the response trace to
    rate * len(response) points, used to bring long natural-movie response traces down to a
    more manageable temporal resolution before building the (large) movie-frame RSM."""
    num_samples = int(len(response) * rate)
    response_ds = sp.signal.resample(response, num_samples)

    return response_ds


if __name__ == "__main__":
    print("can't be run from command line, please import and call functions")
