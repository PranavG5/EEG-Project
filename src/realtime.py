"""Live decoding: headband in, "left" or "right" out.

The offline pipeline classifies a trial that someone already cut out of a
recording and labelled. Online, there is no trial — just an unbroken stream —
so this module slides a window along it and classifies each window, then
smooths.

Three details separate a live decoder that works from one that appears to work
offline and then flickers uselessly on real hardware:

1. **The window must match training.** Same length, same filters, same
   reference, same sampling rate. :class:`LiveDecoder` reuses
   :func:`~src.preprocess.preprocess_raw` for exactly this reason.
2. **Filter warm-up must be outside the window.** An FIR bandpass needs
   history; applying it to a 3-second buffer and classifying all 3 seconds
   means the first ~0.5 s is filter transient, not signal. We therefore keep a
   longer buffer than we classify and discard the edges.
3. **Single windows are noisy.** Consecutive predictions from a
   70%-accurate classifier disagree constantly. Averaging probabilities over a
   second of windows and abstaining below a confidence threshold turns a
   flickering output into a usable one.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from collections import deque
from pathlib import Path

import numpy as np

from .config import (
    CANONICAL_SFREQ,
    INT_TO_CLASS,
    LIVE_CONFIDENCE_THRESHOLD,
    LIVE_SMOOTHING_WINDOWS,
    LIVE_STEP_S,
    LIVE_WINDOW_S,
)
from .devices.base import EEGSource, RingBuffer


@dataclass
class Prediction:
    """One smoothed decision from the live decoder."""

    label: str  # "left", "right", or "uncertain"
    confidence: float
    probabilities: dict[str, float]
    raw_label: str  # this window's own decision, before smoothing
    timestamp: float

    def bar(self, width: int = 24) -> str:
        """Terminal display: a bar leaning toward whichever side is winning."""
        p_right = self.probabilities.get("right", 0.5)
        pos = int(p_right * width)
        track = ["-"] * width
        track[min(pos, width - 1)] = "#"
        marker = "".join(track)
        tag = self.label.upper() if self.label != "uncertain" else "  ?  "
        return f"LEFT [{marker}] RIGHT   {tag:<7s} {self.confidence:.2f}"


class LiveDecoder:
    """Sliding-window motor-imagery decoder over an :class:`EEGSource`.

    Parameters
    ----------
    source
        Any live or replayed EEG stream.
    model
        A fitted estimator exposing ``predict_proba`` on
        ``(n_trials, n_channels, n_times)`` arrays — i.e. anything from
        :mod:`.models`.
    ch_names
        The channels, in order, that the model was trained on. The buffer is
        reordered to match; if the device is missing one of them the decoder
        refuses to start rather than silently feeding a permuted feature vector
        to the classifier, which produces plausible-looking nonsense.
    warmup_s
        Extra signal kept before the classified window to absorb filter
        transients.
    """

    def __init__(
        self,
        source: EEGSource,
        model,
        ch_names: list[str],
        *,
        window_s: float = LIVE_WINDOW_S,
        step_s: float = LIVE_STEP_S,
        smoothing: int = LIVE_SMOOTHING_WINDOWS,
        confidence_threshold: float = LIVE_CONFIDENCE_THRESHOLD,
        target_sfreq: float = CANONICAL_SFREQ,
        warmup_s: float = 1.0,
    ) -> None:
        self.source = source
        self.model = model
        self.window_s = window_s
        self.step_s = step_s
        self.confidence_threshold = confidence_threshold
        self.target_sfreq = target_sfreq
        self.warmup_s = warmup_s

        device_channels = list(source.info.ch_names)
        missing = [ch for ch in ch_names if ch not in device_channels]
        if missing:
            raise ValueError(
                f"Model expects channels {ch_names} but the device provides "
                f"{device_channels}. Missing: {missing}. Retrain the model against "
                "this device's montage (`train.py --device <key>`) rather than "
                "running it on a different electrode set."
            )
        self.ch_names = list(ch_names)

        buffer_s = window_s + warmup_s
        self._buffer = RingBuffer(
            n_channels=len(device_channels),
            n_samples=int(round(buffer_s * source.info.sfreq)),
        )
        self._history: deque[np.ndarray] = deque(maxlen=max(1, smoothing))

    # -- internals -------------------------------------------------------

    def _window_to_features(self, buf: np.ndarray) -> np.ndarray:
        """Preprocess a raw buffer into one model-ready epoch.

        Returns ``(1, n_channels, n_times)``.
        """
        from .preprocess import preprocess_raw, raw_from_array

        raw = raw_from_array(buf, self.source.info.sfreq, list(self.source.info.ch_names))
        raw = preprocess_raw(raw, pick_channels=self.ch_names, target_sfreq=self.target_sfreq)
        data = raw.get_data()

        # Drop the warm-up region: keep the trailing window_s seconds only.
        n_keep = int(round(self.window_s * self.target_sfreq))
        if data.shape[1] < n_keep:
            raise RuntimeError(
                f"Preprocessed window has {data.shape[1]} samples, need {n_keep}."
            )
        return data[:, -n_keep:][np.newaxis, :, :]

    def _smooth(self, probs: np.ndarray) -> tuple[str, float, dict[str, float]]:
        """Average recent windows' probabilities and apply the abstain rule."""
        self._history.append(probs)
        mean = np.mean(self._history, axis=0)
        idx = int(np.argmax(mean))
        confidence = float(mean[idx])
        label = INT_TO_CLASS[idx] if confidence >= self.confidence_threshold else "uncertain"
        prob_map = {INT_TO_CLASS[i]: float(p) for i, p in enumerate(mean)}
        return label, confidence, prob_map

    # -- public API ------------------------------------------------------

    def predict_once(self) -> Prediction | None:
        """Drain the source, and if the buffer is full, classify one window.

        Returns ``None`` while the buffer is still filling — expected for the
        first few seconds after start.
        """
        self._buffer.push(self.source.poll())
        if not self._buffer.is_full:
            return None

        epoch = self._window_to_features(self._buffer.snapshot())
        probs = np.asarray(self.model.predict_proba(epoch))[0]
        raw_label = INT_TO_CLASS[int(np.argmax(probs))]
        label, confidence, prob_map = self._smooth(probs)
        return Prediction(
            label=label,
            confidence=confidence,
            probabilities=prob_map,
            raw_label=raw_label,
            timestamp=time.time(),
        )

    def run(self, duration_s: float | None = None, callback=None) -> list[Prediction]:
        """Decode continuously until ``duration_s`` elapses or Ctrl-C.

        ``callback(Prediction)`` is invoked for each decision; the default
        prints a live confidence bar.
        """
        callback = callback or (lambda p: print("\r" + p.bar(), end="", flush=True))
        results: list[Prediction] = []
        deadline = None if duration_s is None else time.monotonic() + duration_s

        self.source.start()
        try:
            while deadline is None or time.monotonic() < deadline:
                tick = time.monotonic()
                prediction = self.predict_once()
                if prediction is not None:
                    results.append(prediction)
                    callback(prediction)
                # Fixed cadence: sleep off whatever the classification did not use.
                slack = self.step_s - (time.monotonic() - tick)
                if slack > 0:
                    time.sleep(slack)
        except KeyboardInterrupt:
            pass
        finally:
            self.source.stop()
            print()
        return results


# --------------------------------------------------------------------------
# Model persistence
# --------------------------------------------------------------------------


def save_decoder(model, ch_names: list[str], path: Path, **metadata: object) -> Path:
    """Persist a fitted model together with the montage it was trained on.

    Storing the channel list alongside the estimator is not optional bookkeeping:
    a model is only meaningful applied to the same electrodes in the same order,
    and a pickle that does not record them is a trap waiting to be sprung six
    weeks later.
    """
    import joblib

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "ch_names": list(ch_names),
            "sfreq": CANONICAL_SFREQ,
            "window_s": LIVE_WINDOW_S,
            "metadata": metadata,
        },
        path,
    )
    return path


def load_decoder(path: Path) -> tuple[object, list[str], dict]:
    """Load a model saved by :func:`save_decoder`."""
    import joblib

    bundle = joblib.load(Path(path))
    return bundle["model"], bundle["ch_names"], bundle.get("metadata", {})
