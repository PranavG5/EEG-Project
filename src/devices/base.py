"""Hardware-agnostic EEG source interface.

Nothing above this layer knows what headband is plugged in. A source yields
(n_channels, n_samples) arrays in **volts** with a known sampling rate and
known 10-05 electrode names, and that is the entire contract. This is what
lets the same decoder run against a live Neurosity Crown, an OpenBCI Cyton, a
file being replayed, or a synthetic signal generator used for development
when no hardware is on the desk.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field

import numpy as np

from .channels import MontageAssessment, assess_montage, canonicalize_all


@dataclass
class StreamInfo:
    """Static description of what a source produces."""

    name: str
    sfreq: float
    ch_names: list[str]  # canonical 10-05 names, EEG only
    montage: MontageAssessment = field(init=False)

    def __post_init__(self) -> None:
        self.montage = assess_montage(self.ch_names)


class EEGSource(abc.ABC):
    """A live or replayed stream of EEG samples.

    Implementations must return data in volts. Consumer SDKs almost universally
    hand back microvolts, so the conversion belongs in the implementation, not
    scattered through the analysis code — a factor of 1e6 error is invisible to
    a scale-invariant classifier like CSP+LDA but silently destroys anything
    with an absolute threshold, including EEGNet's batch norm statistics.
    """

    @property
    @abc.abstractmethod
    def info(self) -> StreamInfo:
        """Static stream description."""

    @abc.abstractmethod
    def start(self) -> None:
        """Begin acquisition. Safe to call once."""

    @abc.abstractmethod
    def stop(self) -> None:
        """End acquisition and release the device."""

    @abc.abstractmethod
    def poll(self) -> np.ndarray:
        """Return samples acquired since the previous call.

        Shape ``(n_channels, n_new_samples)`` in volts, channel order matching
        ``info.ch_names``. May return an empty ``(n_channels, 0)`` array if no
        new data has arrived. Must never block for long: the live decoding loop
        calls this at a fixed cadence and a blocking read shows up directly as
        decoder latency.
        """

    # -- convenience ------------------------------------------------------

    def __enter__(self) -> "EEGSource":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()


class RingBuffer:
    """Fixed-capacity, per-channel circular buffer of recent samples.

    The live decoder needs "the last N seconds" on every tick. Re-allocating
    and concatenating arrays at 4 Hz for hours leaks memory and stutters, so we
    keep one preallocated array and overwrite it.
    """

    def __init__(self, n_channels: int, n_samples: int) -> None:
        self._buf = np.zeros((n_channels, n_samples), dtype=np.float64)
        self._n_samples = n_samples
        self._filled = 0

    def push(self, chunk: np.ndarray) -> None:
        """Append a ``(n_channels, k)`` chunk, discarding the oldest samples."""
        if chunk.size == 0:
            return
        k = chunk.shape[1]
        if k >= self._n_samples:
            self._buf[:] = chunk[:, -self._n_samples:]
            self._filled = self._n_samples
            return
        self._buf[:, :-k] = self._buf[:, k:]
        self._buf[:, -k:] = chunk
        self._filled = min(self._n_samples, self._filled + k)

    @property
    def is_full(self) -> bool:
        return self._filled >= self._n_samples

    def snapshot(self) -> np.ndarray:
        """Copy of the buffer, oldest sample first."""
        return self._buf.copy()


def eeg_channel_subset(raw_names: list[str]) -> tuple[list[int], list[str]]:
    """Split a device's raw channel list into EEG indices and canonical names.

    Devices interleave accelerometer, battery, timestamp and counter channels
    with the EEG. Feeding those into a spatial filter is not merely useless: a
    battery-level channel is near-constant, so its variance is ~0 and CSP's
    whitening step inverts a near-singular covariance matrix.
    """
    mapping = canonicalize_all(raw_names)
    indices = [i for i, name in enumerate(raw_names) if name in mapping]
    names = [mapping[raw_names[i]] for i in indices]
    return indices, names
