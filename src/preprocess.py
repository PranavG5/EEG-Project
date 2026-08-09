"""Loading, filtering, referencing and epoching.

This module deliberately has two entry points that converge on the same
representation:

* :func:`load_physionet_subject` — research EDF files, 64 channels, 160 Hz.
* :func:`raw_from_array` — a chunk of samples from a headband, N channels,
  whatever rate the vendor chose.

Both produce an :class:`mne.io.Raw` carrying canonical 10-05 channel names and
a montage, and both are then passed through the *same* :func:`preprocess_raw`.
Keeping one filter/reference/epoch implementation is not tidiness for its own
sake: any difference between how training data and live data are processed
shows up as a silent accuracy collapse that is extremely hard to debug, because
both halves look correct in isolation.
"""

from __future__ import annotations

import warnings

import mne
import numpy as np
from mne.datasets import eegbci

from .config import (
    BANDPASS_HIGH,
    BANDPASS_LOW,
    BASELINE_TMAX,
    BASELINE_TMIN,
    CANONICAL_SFREQ,
    CLASS_TO_INT,
    EPOCH_TMAX,
    EPOCH_TMIN,
    IMAGERY_RUNS,
    MONTAGE_NAME,
    NOTCH_FREQ,
    PHYSIONET_EVENT_MAP,
    REJECT_PEAK_TO_PEAK_UV,
)
from .devices.channels import canonicalize_all


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def load_physionet_subject(
    subject: int,
    runs: tuple[int, ...] = IMAGERY_RUNS,
    *,
    local_files: list[str] | None = None,
) -> mne.io.BaseRaw:
    """Load and concatenate one subject's EEGMMIDB runs.

    Runs 4/8/12 contain *imagined* left- vs right-fist movement. Each run is a
    separate EDF; concatenating them gives ~45 trials per subject, which is
    thin for deep learning but adequate for CSP+LDA.

    The channel names in these files follow the BCI2000 convention with
    trailing dots ("Fc5."), so ``eegbci.standardize`` is applied before the
    montage is set — without it, MNE cannot match electrodes to 3D positions
    and every topographic plot silently fails.
    """
    if local_files:
        paths = local_files
    else:
        paths = eegbci.load_data(subjects=subject, runs=list(runs), update_path=True)

    raws = []
    for path in paths:
        raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
        eegbci.standardize(raw)
        raws.append(raw)

    raw = mne.concatenate_raws(raws, verbose="ERROR")
    return _apply_montage(raw)


def raw_from_array(
    data: np.ndarray,
    sfreq: float,
    ch_names: list[str],
) -> mne.io.BaseRaw:
    """Wrap a ``(n_channels, n_samples)`` array of volts as an MNE Raw object.

    This is the bridge from the hardware layer into the analysis code: once a
    headband chunk is a Raw with a montage attached, every downstream function
    written for the PhysioNet data works on it unchanged.
    """
    mapping = canonicalize_all(ch_names)
    canonical = [mapping.get(name, name) for name in ch_names]
    info = mne.create_info(ch_names=canonical, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    return _apply_montage(raw)


def _apply_montage(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """Attach 3D electrode positions, dropping channels the montage lacks.

    Positions are needed for the average reference to be meaningful, for
    topomaps, and for any interpolation. ``on_missing="warn"`` keeps a device
    with an unusual electrode (e.g. Muse's TP9/TP10) usable rather than
    raising, but the warning is worth reading.
    """
    montage = mne.channels.make_standard_montage(MONTAGE_NAME)
    known = set(montage.ch_names)
    unknown = [ch for ch in raw.ch_names if ch not in known]
    if unknown:
        warnings.warn(
            f"Dropping channels with no position in {MONTAGE_NAME}: {unknown}",
            stacklevel=2,
        )
        raw.drop_channels(unknown)
    raw.set_montage(montage, on_missing="warn", verbose="ERROR")
    return raw


# --------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------


def preprocess_raw(
    raw: mne.io.BaseRaw,
    *,
    pick_channels: list[str] | None = None,
    target_sfreq: float = CANONICAL_SFREQ,
    notch: bool = True,
    average_reference: bool = True,
    l_freq: float = BANDPASS_LOW,
    h_freq: float = BANDPASS_HIGH,
) -> mne.io.BaseRaw:
    """Filter, re-reference and resample, in the order that matters.

    Steps, and why each one is where it is:

    1. **Channel selection.** Restricting to the electrodes the target device
       has, *before* referencing, so that the average reference is computed
       over the same set of sensors at training and at inference time. An
       average reference over 64 channels is a different signal from an average
       over 8, so referencing first and subsetting after would make the model
       untransferable.
    2. **Notch at 60 Hz.** Removes US powerline contamination. Applied before
       the bandpass mostly for signal hygiene; the 8-30 Hz bandpass would
       reject 60 Hz anyway, but a huge line component can push the filter into
       numerical trouble.
    3. **Bandpass 8-30 Hz.** Isolates the mu and beta sensorimotor rhythms
       whose desynchronisation encodes motor imagery, while removing slow
       drift/sweat artefact below and EMG above.
    4. **Common average reference.** EEG voltage is meaningless without a
       reference, and whatever the vendor chose (mastoid, earlobe, Fpz) sits
       somewhere on the head that itself carries brain signal. Subtracting the
       mean across electrodes approximates a neutral reference and, more
       importantly for this task, sharpens spatial contrasts between the
       hemispheres.
    5. **Resample to the canonical rate.** Last, because resampling before
       filtering risks aliasing the very band we want.
    """
    raw = raw.copy()

    if pick_channels is not None:
        available = [ch for ch in pick_channels if ch in raw.ch_names]
        missing = [ch for ch in pick_channels if ch not in raw.ch_names]
        if missing:
            warnings.warn(f"Requested channels not present, skipping: {missing}", stacklevel=2)
        if not available:
            raise ValueError(
                f"None of the requested channels {pick_channels} are in this recording "
                f"({raw.ch_names})."
            )
        raw.pick(available)

    nyquist = raw.info["sfreq"] / 2.0
    if notch and NOTCH_FREQ < nyquist:
        raw.notch_filter(NOTCH_FREQ, verbose="ERROR")

    # Guard the upper edge: a 125 Hz board (Cyton+Daisy) has Nyquist 62.5 Hz so
    # 30 Hz is fine, but a low-rate source would otherwise raise deep inside MNE.
    h_eff = min(h_freq, nyquist - 1.0)
    raw.filter(l_freq, h_eff, fir_design="firwin", verbose="ERROR")

    if average_reference:
        if len(raw.ch_names) < 2:
            warnings.warn(
                "Average reference needs >=2 channels; skipping.", stacklevel=2
            )
        else:
            raw.set_eeg_reference("average", projection=False, verbose="ERROR")

    if target_sfreq and abs(raw.info["sfreq"] - target_sfreq) > 1e-6:
        raw.resample(target_sfreq, verbose="ERROR")

    return raw


# --------------------------------------------------------------------------
# Epoching
# --------------------------------------------------------------------------


def epoch_physionet(
    raw: mne.io.BaseRaw,
    *,
    tmin: float = EPOCH_TMIN,
    tmax: float = EPOCH_TMAX,
    baseline: tuple[float, float] | None = None,
    reject_uv: float | None = REJECT_PEAK_TO_PEAK_UV,
) -> mne.Epochs:
    """Cut the continuous recording into labelled left/right trials.

    ``baseline=None`` by default: baseline correction subtracts a pre-cue mean
    from each trial, which is right for evoked-potential analysis but actively
    harmful here. Motor imagery is decoded from *band power*, and the CSP
    filters that follow are variance-based — shifting each trial's DC level
    changes nothing useful and, when the baseline window contains an artefact,
    injects it into the whole trial.

    ``reject_uv`` drops trials with peak-to-peak amplitude above the threshold.
    On real headband data this is what removes blinks, jaw clenches and cable
    yanks. Set it to ``None`` to keep everything and inspect manually.
    """
    events, event_id = mne.events_from_annotations(raw, verbose="ERROR")

    wanted = {}
    for code, label in PHYSIONET_EVENT_MAP.items():
        if code in event_id:
            wanted[label] = event_id[code]
    if len(wanted) < 2:
        raise ValueError(
            f"Expected annotations {list(PHYSIONET_EVENT_MAP)} for left/right imagery, "
            f"found {sorted(event_id)}. Are these the right runs (4/8/12)?"
        )

    reject = {"eeg": reject_uv * 1e-6} if reject_uv else None

    epochs = mne.Epochs(
        raw,
        events,
        event_id=wanted,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        picks="eeg",
        preload=True,
        reject=reject,
        verbose="ERROR",
    )
    return epochs


def epochs_to_xy(epochs: mne.Epochs) -> tuple[np.ndarray, np.ndarray]:
    """Split epochs into an ``(n_trials, n_channels, n_times)`` array and labels.

    Labels are integers via :data:`~src.config.CLASS_TO_INT` so that 0 is always
    "left" and 1 always "right", regardless of the order MNE happened to assign
    event ids in — an ordering bug here inverts the classifier's output and
    presents as a suspiciously consistent ~35% accuracy.
    """
    x = epochs.get_data(copy=True)
    inv = {v: k for k, v in epochs.event_id.items()}
    labels = [inv[code] for code in epochs.events[:, -1]]
    y = np.array([CLASS_TO_INT[label] for label in labels], dtype=int)
    return x, y


def epoch_windows(
    raw: mne.io.BaseRaw,
    onsets_s: np.ndarray,
    labels: np.ndarray,
    *,
    tmin: float = EPOCH_TMIN,
    tmax: float = EPOCH_TMAX,
    reject_uv: float | None = REJECT_PEAK_TO_PEAK_UV,
) -> tuple[np.ndarray, np.ndarray]:
    """Epoch a headband recording around explicit cue times.

    Our own recordings carry cue timestamps in a sidecar file rather than EDF
    annotations, so this builds the MNE event array directly. Returns the same
    ``(X, y)`` contract as :func:`epochs_to_xy`, so calibration data and
    PhysioNet data are interchangeable from here on.
    """
    sfreq = raw.info["sfreq"]
    samples = np.round(np.asarray(onsets_s) * sfreq).astype(int)
    events = np.column_stack(
        [samples, np.zeros_like(samples), np.asarray(labels, dtype=int) + 1]
    )
    event_id = {name: idx + 1 for name, idx in CLASS_TO_INT.items()}
    reject = {"eeg": reject_uv * 1e-6} if reject_uv else None

    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        picks="eeg",
        preload=True,
        reject=reject,
        verbose="ERROR",
    )
    return epochs_to_xy(epochs)


def baseline_window() -> tuple[float, float]:
    """Pre-cue window used as the ERD reference level."""
    return BASELINE_TMIN, BASELINE_TMAX
