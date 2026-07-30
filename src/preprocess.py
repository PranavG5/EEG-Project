"""Loading, filtering, referencing, and epoching of EEG motor-imagery data.

This module implements milestone 1 of the project: get a single subject's
motor-imagery data loaded end-to-end, filtered into the motor-rhythm band,
re-referenced, and cut into labelled trials — then sanity-check that the
left-fist / right-fist event markers make sense before any features are built.

Dataset: PhysioNet EEG Motor Movement/Imagery Dataset (EEGMMIDB), fetched
through MNE's built-in downloader. See CLAUDE.md for the full project brief.

Event-marker convention (PhysioNet EEGMMIDB, imagery runs 4/8/12):
    T0 -> rest
    T1 -> onset of imagined *left* fist movement
    T2 -> onset of imagined *right* fist movement
This is the documented convention for the left/right-hand runs specifically
(it differs for the hands-vs-feet runs 6/10/14), and it is verified at runtime
by ``describe_events`` rather than assumed.
"""

from __future__ import annotations

from typing import Sequence

import matplotlib

# Use a non-interactive backend so figures render in a headless environment.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.datasets import eegbci
from mne.io import BaseRaw

# --- Constants -------------------------------------------------------------

# Mu (8-13 Hz) and beta (13-30 Hz) rhythms over sensorimotor cortex are the
# neural signatures of motor imagery: imagining a hand movement desynchronises
# (suppresses power in) these bands contralateral to the imagined hand
# (event-related desynchronisation, ERD). Band-passing to 8-30 Hz isolates that
# signal and rejects slow drifts and high-frequency EMG/line noise.
BANDPASS_LOW_HZ: float = 8.0
BANDPASS_HIGH_HZ: float = 30.0

# Epoch window relative to cue onset. We start at +0.5 s rather than 0 s so the
# stimulus-onset transient / evoked response does not contaminate the trial,
# and end at +3.5 s to capture the sustained ERD during the ~4 s imagery period.
EPOCH_TMIN: float = 0.5
EPOCH_TMAX: float = 3.5

# Human-readable labels for the two imagery classes we care about.
# Keys are the raw annotation descriptions; values are our class names.
IMAGERY_EVENT_LABELS: dict[str, str] = {"T1": "left_fist", "T2": "right_fist"}


# --- Loading ---------------------------------------------------------------

def load_raw(
    subject: int = 1,
    runs: Sequence[int] = (4,),
    edf_paths: Sequence[str] | None = None,
) -> BaseRaw:
    """Load one subject's EEG runs as a single Raw, downloading if needed.

    By default uses MNE's ``eegbci.load_data`` fetcher, so no manual download is
    required; files are cached under ``~/mne_data`` on first use. Note the
    parameter is ``subjects`` (plural) in MNE's API even for a single subject.

    If ``edf_paths`` is given, those local EDF files are loaded directly and the
    downloader is skipped — useful when the PhysioNet host is unreachable (e.g.
    a restricted network) and the files have been supplied out of band.

    Channel names in the raw EDF files carry BCI2000 quirks (trailing dots,
    e.g. ``Fc5.``). ``eegbci.standardize`` rewrites them to the canonical
    10-05 names so a standard montage can be attached, which is needed for any
    later spatial analysis / topomaps.

    Parameters
    ----------
    subject
        Subject id, 1-109. Ignored when ``edf_paths`` is provided.
    runs
        Run numbers to load and concatenate. Runs 4, 8, 12 are the imagined
        left/right fist runs. Ignored when ``edf_paths`` is provided.
    edf_paths
        Optional explicit list of local ``.edf`` files to load instead of
        fetching from PhysioNet.

    Returns
    -------
    Raw
        The concatenated raw recording with a standard 10-05 montage set.
    """
    paths = list(edf_paths) if edf_paths is not None else eegbci.load_data(
        subjects=subject, runs=list(runs)
    )
    raws = [mne.io.read_raw_edf(p, preload=True) for p in paths]
    raw = mne.concatenate_raws(raws)

    # Normalise channel names, then attach the electrode positions.
    eegbci.standardize(raw)
    montage = mne.channels.make_standard_montage("standard_1005")
    raw.set_montage(montage)

    return raw


# --- Preprocessing ---------------------------------------------------------

def preprocess_raw(
    raw: BaseRaw,
    l_freq: float = BANDPASS_LOW_HZ,
    h_freq: float = BANDPASS_HIGH_HZ,
) -> BaseRaw:
    """Band-pass filter and common-average-reference the raw signal in place.

    - **Band-pass 8-30 Hz** isolates the mu/beta sensorimotor rhythms that carry
      the motor-imagery signal (see ``BANDPASS_LOW_HZ``).
    - **Common Average Reference (CAR)** subtracts the mean across all EEG
      channels at each time point. It suppresses activity common to the whole
      scalp (reference drift, global noise) and sharpens the spatially focal
      contralateral ERD that distinguishes left- from right-hand imagery.

    Note on notch filtering: the brief mentions a 60 Hz US-powerline notch "if
    needed". Because the 8-30 Hz band-pass already sits well below 60 Hz, the
    line component is strongly attenuated by the band-pass itself, so a separate
    notch is redundant here and is intentionally omitted.

    Parameters
    ----------
    raw
        Raw recording (modified in place).
    l_freq, h_freq
        Band-pass edges in Hz.

    Returns
    -------
    Raw
        The same object, filtered and re-referenced (returned for chaining).
    """
    raw.filter(l_freq, h_freq, picks="eeg", verbose="WARNING")
    raw.set_eeg_reference("average", verbose="WARNING")
    return raw


# --- Epoching --------------------------------------------------------------

def make_epochs(
    raw: BaseRaw,
    tmin: float = EPOCH_TMIN,
    tmax: float = EPOCH_TMAX,
) -> mne.Epochs:
    """Cut the continuous recording into labelled left/right imagery trials.

    Extracts the T1 (left fist) and T2 (right fist) annotations and epochs the
    band around each cue onset. The resting T0 markers are dropped: this is a
    two-class left-vs-right imagery problem. ``baseline=None`` because the data
    is already band-passed (no slow baseline to correct) and the pre-cue period
    is excluded from the window.

    Parameters
    ----------
    raw
        A (preferably already preprocessed) raw recording with annotations.
    tmin, tmax
        Trial window in seconds relative to cue onset.

    Returns
    -------
    Epochs
        Epochs with classes ``left_fist`` and ``right_fist``.
    """
    events, event_id_from_annot = mne.events_from_annotations(raw, verbose="WARNING")

    # Map only the imagery annotations to our readable class names, resolving
    # each description to the integer code MNE assigned it.
    epoch_event_id = {
        label: event_id_from_annot[desc]
        for desc, label in IMAGERY_EVENT_LABELS.items()
        if desc in event_id_from_annot
    }

    epochs = mne.Epochs(
        raw,
        events,
        event_id=epoch_event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        picks="eeg",
        preload=True,
        verbose="WARNING",
    )
    return epochs


# --- Multi-subject loading -------------------------------------------------

def load_subject_epochs(
    subject: int,
    runs: Sequence[int] = (4, 8, 12),
    tmin: float = EPOCH_TMIN,
    tmax: float = EPOCH_TMAX,
) -> mne.Epochs:
    """Load, preprocess, and epoch a single subject end-to-end.

    Convenience wrapper that runs the full milestone-1 chain (download/load ->
    band-pass + CAR -> epoch T1/T2) for one subject and returns ready-to-use
    left/right imagery trials. Used to assemble the multi-subject dataset for
    cross-subject evaluation, where each subject must be processed independently
    (filtering and referencing are per-recording operations).

    Parameters
    ----------
    subject
        Subject id, 1-109.
    runs
        Imagery runs to concatenate (default: the three left/right fist runs).
    tmin, tmax
        Trial window in seconds relative to cue onset.

    Returns
    -------
    Epochs
        Epoched left/right imagery trials for this subject.
    """
    raw = load_raw(subject=subject, runs=runs)
    preprocess_raw(raw)
    return make_epochs(raw, tmin=tmin, tmax=tmax)


def epochs_from_edf_paths(
    edf_paths: Sequence[str],
    tmin: float = EPOCH_TMIN,
    tmax: float = EPOCH_TMAX,
) -> mne.Epochs:
    """Load, preprocess, and epoch one or more local EDF files end-to-end.

    The file-based counterpart to :func:`load_subject_epochs`: it runs the same
    load -> band-pass + CAR -> epoch chain on explicitly supplied ``.edf`` files
    instead of fetching a subject from PhysioNet. This is what the UI uses for
    uploaded recordings. The files must carry T1/T2 (left/right fist) annotations
    in the EEGMMIDB convention for the imagery classes to be found.

    Parameters
    ----------
    edf_paths
        Local ``.edf`` files to load and concatenate.
    tmin, tmax
        Trial window in seconds relative to cue onset.

    Returns
    -------
    Epochs
        Epoched left/right imagery trials.
    """
    raw = load_raw(edf_paths=list(edf_paths))
    preprocess_raw(raw)
    return make_epochs(raw, tmin=tmin, tmax=tmax)


def stack_subject_epochs(
    epochs_list: Sequence[mne.Epochs],
    subject_ids: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Stack per-subject epochs into one trials tensor with subject-id groups.

    A parallel ``groups`` array records which subject each trial came from — this
    is exactly what leave-one-subject-out cross-validation needs to guarantee no
    subject appears in both train and test.

    All subjects in EEGMMIDB share the same 64-channel montage and (for the
    subjects used here) a 160 Hz sampling rate, so the per-subject epoch tensors
    have matching shapes ``(n_trials_i, n_channels, n_times)`` and can be stacked
    directly. A defensive check rejects any subject whose time-axis length
    differs (a few EEGMMIDB subjects were recorded at 128 Hz), which would
    otherwise corrupt the stacked array.

    Parameters
    ----------
    epochs_list
        Per-subject epoched trials, one :class:`mne.Epochs` per subject.
    subject_ids
        Subject id for each entry in ``epochs_list`` (same order/length).

    Returns
    -------
    X : np.ndarray
        Trials tensor, shape (n_trials_total, n_channels, n_times).
    y : np.ndarray
        Integer class labels (0 = left_fist, 1 = right_fist).
    groups : np.ndarray
        Subject id for each trial (same length as ``y``).
    class_names : list[str]
        Class names indexed by label value.
    """
    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    group_parts: list[np.ndarray] = []
    class_names: list[str] | None = None
    n_times_ref: int | None = None

    for epochs, subject in zip(epochs_list, subject_ids):
        y_sub, names = epochs_to_labels(epochs)
        Xi = epochs.get_data(copy=False)

        if n_times_ref is None:
            n_times_ref = Xi.shape[2]
            class_names = names
        elif Xi.shape[2] != n_times_ref:
            raise ValueError(
                f"Subject {subject} has {Xi.shape[2]} time samples, expected "
                f"{n_times_ref} — likely a different sampling rate; exclude it "
                "or resample before stacking."
            )

        X_parts.append(Xi)
        y_parts.append(y_sub)
        group_parts.append(np.full(len(y_sub), subject, dtype=int))

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    groups = np.concatenate(group_parts, axis=0)
    assert class_names is not None
    return X, y, groups, class_names


def load_multisubject_dataset(
    subjects: Sequence[int],
    runs: Sequence[int] = (4, 8, 12),
    tmin: float = EPOCH_TMIN,
    tmax: float = EPOCH_TMAX,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load, preprocess, epoch, and stack many subjects into one dataset.

    Convenience wrapper: each subject is loaded and preprocessed independently
    (filtering/referencing are per-recording), then all are stacked via
    :func:`stack_subject_epochs`. Callers that also need the per-subject
    :class:`mne.Epochs` (e.g. to build a grand-average topomap) should instead
    load them with :func:`load_subject_epochs` and call
    :func:`stack_subject_epochs` directly, to avoid loading twice.

    Returns
    -------
    (X, y, groups, class_names)
        See :func:`stack_subject_epochs`.
    """
    epochs_list = [
        load_subject_epochs(s, runs=runs, tmin=tmin, tmax=tmax) for s in subjects
    ]
    return stack_subject_epochs(epochs_list, subjects)


def epochs_to_labels(epochs: mne.Epochs) -> tuple[np.ndarray, list[str]]:
    """Map epoch events to 0-based integer labels and their class names.

    The class order is fixed by sorting ``event_id`` on its integer code, so
    label ``i`` always maps to ``class_names[i]`` regardless of the arbitrary
    codes MNE assigns per recording (here 0 = left_fist, 1 = right_fist).

    Returns
    -------
    y : np.ndarray
        Integer labels for each trial.
    class_names : list[str]
        Class names indexed by label value.
    """
    name_by_code = {code: name for name, code in epochs.event_id.items()}
    ordered_codes = sorted(name_by_code)
    class_names = [name_by_code[c] for c in ordered_codes]
    code_to_label = {code: i for i, code in enumerate(ordered_codes)}
    y = np.array([code_to_label[c] for c in epochs.events[:, -1]])
    return y, class_names


# --- Sanity checks / reporting ---------------------------------------------

def describe_events(epochs: mne.Epochs) -> dict[str, int]:
    """Print and return the per-class trial counts for the epoched data.

    This is the milestone-1 verification step: confirm the left-fist / right-
    fist markers were parsed, roughly balanced, and reasonable in number
    (a single imagery run has ~7-8 trials per class).

    Returns
    -------
    dict
        Mapping of class name -> trial count.
    """
    counts = {name: len(epochs[name]) for name in epochs.event_id}

    print("\n=== Event / trial counts (T1 = left fist, T2 = right fist) ===")
    for name, n in counts.items():
        print(f"  {name:<12s}: {n} trials")
    print(f"  {'TOTAL':<12s}: {sum(counts.values())} trials")
    print(f"  epoch window : {epochs.tmin:.2f}s to {epochs.tmax:.2f}s "
          f"({epochs.times.size} samples at {epochs.info['sfreq']:.0f} Hz)\n")

    return counts


def plot_raw_segment(
    raw: BaseRaw,
    out_path: str,
    duration: float = 15.0,
    channels: Sequence[str] = ("C3", "Cz", "C4"),
) -> str:
    """Save a plot of a raw-signal segment over the sensorimotor channels.

    C3 / Cz / C4 sit over left / central / right motor cortex — the electrodes
    where hand motor imagery is most visible — so plotting them is the most
    informative eyeball check that the signal looks like clean band-passed EEG.

    Returns
    -------
    str
        The path the figure was written to.
    """
    picks = [ch for ch in channels if ch in raw.ch_names]
    data, times = raw.get_data(picks=picks, return_times=True)
    # Only plot the first `duration` seconds to keep the figure legible.
    mask = times <= (times[0] + duration)

    fig, axes = plt.subplots(len(picks), 1, figsize=(11, 2.0 * len(picks)),
                             sharex=True)
    if len(picks) == 1:
        axes = [axes]
    for ax, ch, row in zip(axes, picks, data):
        ax.plot(times[mask], row[mask] * 1e6, lw=0.6, color="tab:blue")
        ax.set_ylabel(f"{ch}\n(µV)")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time (s)")
    axes[0].set_title(
        f"Band-passed EEG ({BANDPASS_LOW_HZ:.0f}-{BANDPASS_HIGH_HZ:.0f} Hz), "
        f"sensorimotor channels — first {duration:.0f} s"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_events(raw: BaseRaw, out_path: str) -> str:
    """Save a plot of event markers over time to confirm cue timing/spacing.

    Returns
    -------
    str
        The path the figure was written to.
    """
    events, event_id_from_annot = mne.events_from_annotations(raw, verbose="WARNING")
    fig = mne.viz.plot_events(
        events,
        sfreq=raw.info["sfreq"],
        event_id=event_id_from_annot,
        show=False,
    )
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


# --- Milestone-1 entry point -----------------------------------------------

def main() -> None:
    """Run milestone 1 end-to-end for subject 1, run 4, and report results."""
    import os

    figures_dir = "results/figures"

    # Prefer a locally supplied EDF (e.g. when PhysioNet is unreachable);
    # otherwise fall back to MNE's downloader.
    local_edf = "data/S001R04.edf"
    edf_paths = [local_edf] if os.path.exists(local_edf) else None

    print("Loading subject 1, run 4 (imagined left/right fist)...")
    if edf_paths:
        print(f"  using local file: {local_edf}")
    raw = load_raw(subject=1, runs=(4,), edf_paths=edf_paths)
    print(f"  loaded: {len(raw.ch_names)} channels, "
          f"{raw.n_times} samples at {raw.info['sfreq']:.0f} Hz "
          f"({raw.times[-1]:.1f} s)")

    print(f"Filtering {BANDPASS_LOW_HZ:.0f}-{BANDPASS_HIGH_HZ:.0f} Hz and "
          "applying common average reference...")
    preprocess_raw(raw)

    print("Epoching around T1/T2 cues "
          f"(tmin={EPOCH_TMIN}, tmax={EPOCH_TMAX})...")
    epochs = make_epochs(raw)

    describe_events(epochs)

    raw_fig = plot_raw_segment(raw, f"{figures_dir}/milestone1_raw_signal.png")
    events_fig = plot_events(raw, f"{figures_dir}/milestone1_events.png")
    print(f"Saved raw-signal figure  -> {raw_fig}")
    print(f"Saved event-marker figure -> {events_fig}")


if __name__ == "__main__":
    main()
