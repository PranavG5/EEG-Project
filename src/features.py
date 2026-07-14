"""Feature extraction for EEG motor-imagery trials.

Milestone 2 implements the simplest, most interpretable feature: **band power**
per channel. Later milestones add CSP and wavelet features here as separate
functions so the three can be compared on the same classifiers.

Neuroscience rationale for band power
-------------------------------------
Motor imagery suppresses the mu (8-13 Hz) and beta (13-30 Hz) rhythms over the
sensorimotor cortex *contralateral* to the imagined hand — event-related
desynchronisation (ERD). Concretely, imagining the left hand reduces 8-30 Hz
power over the right motor cortex (electrode C4) and vice versa. So the average
8-30 Hz power at each electrode is a direct, physically meaningful summary of
where that suppression is happening, and the left-vs-right contrast lives mostly
in the C3-vs-C4 power difference. This makes band power the natural baseline
feature before more elaborate spatial filters (CSP) are introduced.
"""

from __future__ import annotations

import mne
import numpy as np


def band_power(
    epochs: mne.Epochs,
    fmin: float = 8.0,
    fmax: float = 30.0,
) -> np.ndarray:
    """Compute log band power in ``fmin``-``fmax`` Hz for each channel and trial.

    Power spectral density is estimated per trial with Welch's method and then
    averaged across the frequency bins inside the band, giving one number per
    channel: the average mu/beta power during that trial. A log transform is
    applied because band-power values are strongly right-skewed (power is
    non-negative and roughly log-normal); logging makes the distribution closer
    to Gaussian, which suits a linear classifier such as LDA.

    Note that because the signal is already band-passed to 8-30 Hz upstream,
    this band power is essentially the per-channel signal variance; computing it
    explicitly over the band via Welch keeps the feature meaningful even if the
    upstream filter changes.

    Parameters
    ----------
    epochs
        Epoched trials, shape (n_trials, n_channels, n_times).
    fmin, fmax
        Frequency band to integrate power over, in Hz.

    Returns
    -------
    np.ndarray
        Feature matrix of shape (n_trials, n_channels): log band power.
    """
    spectrum = epochs.compute_psd(
        method="welch", fmin=fmin, fmax=fmax, verbose="WARNING"
    )
    # psds: (n_trials, n_channels, n_freqs)
    psds = spectrum.get_data()
    # Average power across the band -> (n_trials, n_channels), then log.
    band = psds.mean(axis=2)
    return np.log(band)
