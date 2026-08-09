"""Feature extraction: band power, CSP, and wavelet energy.

All three are exposed as scikit-learn transformers operating on
``(n_trials, n_channels, n_times)`` arrays, so they drop into a ``Pipeline``
interchangeably and every one of them is fit only on training folds — which is
the entire reason CSP is wrapped rather than called ad hoc. CSP is a
*supervised* spatial filter: fitting it on all the data before splitting leaks
test labels into the filters and inflates accuracy by 10-20 points. This is the
single most common bug in published student BCI projects.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import welch
from sklearn.base import BaseEstimator, TransformerMixin

from .config import CANONICAL_SFREQ, N_CSP_COMPONENTS, SUB_BANDS


# --------------------------------------------------------------------------
# Band power
# --------------------------------------------------------------------------


class BandPowerFeatures(BaseEstimator, TransformerMixin):
    """Log power in mu/beta sub-bands, per channel.

    The simplest thing that could possibly work, and the baseline every other
    method has to beat. The physiology it encodes directly: imagining a left
    hand movement suppresses mu (8-13 Hz) power over the *right* sensorimotor
    cortex (C4), and vice versa. So the feature that matters is essentially the
    C3-minus-C4 mu power difference — a linear classifier on these features can
    discover exactly that, which makes the model interpretable in a way CSP and
    EEGNet are not.

    The log transform matters. Band power is a variance, so it is
    chi-square-ish distributed with a long right tail; ``log`` makes the
    distribution roughly Gaussian, which is what LDA assumes.
    """

    def __init__(
        self,
        sfreq: float = CANONICAL_SFREQ,
        bands: dict[str, tuple[float, float]] | None = None,
        n_per_seg: int | None = None,
    ) -> None:
        self.sfreq = sfreq
        self.bands = bands
        self.n_per_seg = n_per_seg

    def _bands(self) -> dict[str, tuple[float, float]]:
        return self.bands if self.bands is not None else SUB_BANDS

    def fit(self, x: np.ndarray, y: np.ndarray | None = None) -> "BandPowerFeatures":
        self.n_channels_ = x.shape[1]
        self.feature_names_ = [
            f"ch{c}_{band}" for c in range(self.n_channels_) for band in self._bands()
        ]
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        n_trials, n_channels, n_times = x.shape
        # Welch trades frequency resolution for variance reduction by averaging
        # periodograms over overlapping segments. Half-second segments give
        # ~2 Hz resolution, plenty to separate mu from beta, and average ~11
        # segments across a 3 s trial.
        n_per_seg = self.n_per_seg or min(n_times, int(self.sfreq * 0.5))
        freqs, psd = welch(
            x, fs=self.sfreq, nperseg=n_per_seg, noverlap=n_per_seg // 2, axis=-1
        )

        out = []
        for lo, hi in self._bands().values():
            mask = (freqs >= lo) & (freqs < hi)
            if not mask.any():
                raise ValueError(
                    f"No FFT bins in band {lo}-{hi} Hz at fs={self.sfreq} with "
                    f"nperseg={n_per_seg}; the trial is too short."
                )
            out.append(psd[:, :, mask].mean(axis=-1))

        # (n_bands, n_trials, n_channels) -> (n_trials, n_channels * n_bands)
        stacked = np.stack(out, axis=-1).reshape(n_trials, n_channels * len(self._bands()))
        return np.log(stacked + 1e-20)


# --------------------------------------------------------------------------
# Common Spatial Patterns
# --------------------------------------------------------------------------


def make_csp(n_components: int = N_CSP_COMPONENTS, *, log: bool = True):
    """MNE's CSP, configured for two-class motor imagery.

    What CSP actually does, since this needs to be explainable rather than
    invoked: it finds a set of spatial filters — weighted sums across
    electrodes — chosen so that the filtered signal has maximum variance for
    one class and minimum for the other. Formally it simultaneously
    diagonalises the two classes' channel covariance matrices, solving the
    generalised eigenvalue problem ``C_left w = lambda (C_left + C_right) w``.
    The eigenvectors with the largest and smallest eigenvalues are the most
    discriminative filters.

    Why that is the right thing for this task: after an 8-30 Hz bandpass, the
    variance of a channel *is* its band power, and left-vs-right imagery is
    precisely a difference in the spatial distribution of band power. CSP
    therefore learns, from data, a lateralised weighting that typically ends up
    looking like a Laplacian centred on C3 versus one centred on C4 — the same
    contrast a neuroscientist would pick by hand, but optimised per subject,
    which matters because everyone's central sulcus sits in a slightly
    different place relative to the electrodes.

    ``n_components=6`` takes the 3 most extreme filters from each end.
    ``log=True`` returns log-variance per filter, for the same Gaussianity
    reason as in :class:`BandPowerFeatures`.
    """
    from mne.decoding import CSP

    return CSP(n_components=n_components, reg="ledoit_wolf", log=log, norm_trace=False)


# --------------------------------------------------------------------------
# Wavelets
# --------------------------------------------------------------------------


class WaveletEnergyFeatures(BaseEstimator, TransformerMixin):
    """Relative energy per DWT sub-band, per channel.

    Motivation over Welch: motor-imagery ERD is not stationary across a trial —
    it builds over roughly the first second and can wane. A Fourier band power
    averages that away. The discrete wavelet transform decomposes the signal
    into dyadic frequency bands with time localisation, so a burst of
    desynchronisation lasting 800 ms is represented differently from a uniform
    power drop.

    ``db4`` (Daubechies-4) is the conventional choice in the EEG literature: it
    is short enough to localise transients but smooth enough not to ring on the
    oscillatory mu rhythm. Energies are normalised per channel so the feature
    describes the *shape* of the spectrum rather than the overall amplitude,
    which drifts with electrode impedance over a session.
    """

    def __init__(
        self,
        wavelet: str = "db4",
        level: int = 4,
        sfreq: float = CANONICAL_SFREQ,
        normalize: bool = True,
    ) -> None:
        self.wavelet = wavelet
        self.level = level
        self.sfreq = sfreq
        self.normalize = normalize

    def fit(self, x: np.ndarray, y: np.ndarray | None = None) -> "WaveletEnergyFeatures":
        self.n_channels_ = x.shape[1]
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        import pywt

        n_trials, n_channels, _ = x.shape
        feats = []
        for trial in range(n_trials):
            row = []
            for ch in range(n_channels):
                coeffs = pywt.wavedec(x[trial, ch], self.wavelet, level=self.level)
                energies = np.array([float(np.sum(c**2)) for c in coeffs])
                if self.normalize:
                    energies = energies / (energies.sum() + 1e-20)
                row.append(np.log(energies + 1e-20))
            feats.append(np.concatenate(row))
        return np.asarray(feats)


# --------------------------------------------------------------------------
# ERD/ERS
# --------------------------------------------------------------------------


def erd_percentage(
    active: np.ndarray,
    baseline: np.ndarray,
    sfreq: float = CANONICAL_SFREQ,
    band: tuple[float, float] = (8.0, 13.0),
) -> np.ndarray:
    """Event-related desynchronisation as percent change from baseline, per channel.

    ``ERD% = 100 * (P_active - P_baseline) / P_baseline``

    Negative values are desynchronisation (power drop = cortex engaged),
    positive values are synchronisation. This is the quantity plotted on the
    topomap that makes the result legible as neuroscience: imagining the left
    hand should paint a blue (negative) blob over the right hemisphere.

    Parameters take ``(n_trials, n_channels, n_times)`` arrays for the active
    and baseline windows respectively.
    """
    def _power(arr: np.ndarray) -> np.ndarray:
        n_per_seg = min(arr.shape[-1], int(sfreq * 0.5))
        freqs, psd = welch(arr, fs=sfreq, nperseg=n_per_seg, axis=-1)
        mask = (freqs >= band[0]) & (freqs < band[1])
        return psd[:, :, mask].mean(axis=-1).mean(axis=0)

    p_active = _power(active)
    p_base = _power(baseline)
    return 100.0 * (p_active - p_base) / (p_base + 1e-20)
