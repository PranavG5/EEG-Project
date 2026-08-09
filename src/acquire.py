"""Cued calibration recorder: collect your own labelled motor-imagery data.

This module exists because of a fact that is easy to miss until hardware
arrives: **a model trained on PhysioNet will not decode your headband.** Not
because the code is wrong, but because the mapping from cortical current to
recorded voltage differs between a 64-electrode gel cap on subject S001 in 2009
and dry electrodes on your head today — different electrode positions, different
skull, different reference, different amplifier, different impedance, different
mental strategy for "imagine your left hand". Cross-subject motor-imagery
accuracy is already only ~60-70% within one dataset; across hardware it
collapses toward chance.

The fix is not a better classifier. It is twenty minutes of calibration: record
yourself performing cued left/right imagery, train on *that*, and the model
becomes yours. This module runs that session and saves the result in a form
:mod:`.train` can consume directly.

The recording protocol follows the Graz BCI paradigm, which is the convention
this field standardised on:

    [ fixation 2 s ] -> [ cue arrow 1.25 s ] -> [ imagery 4 s ] -> [ rest 2-3 s ]

Randomised, jittered rest, balanced classes. The jitter matters: with a fixed
trial period, any periodic artefact (a fan, a heartbeat, the subject blinking
on a rhythm) becomes phase-locked to the cue, and the classifier learns that
instead of motor cortex.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .config import CLASS_TO_INT, EPOCH_TMAX, RECORDINGS_DIR, REJECT_PEAK_TO_PEAK_UV
from .devices.base import EEGSource


# --------------------------------------------------------------------------
# Protocol
# --------------------------------------------------------------------------


@dataclass
class CueProtocol:
    """Timing of one calibration session."""

    n_trials_per_class: int = 30
    fixation_s: float = 2.0
    cue_s: float = 1.25
    imagery_s: float = 4.0
    rest_min_s: float = 2.0
    rest_max_s: float = 3.5
    seed: int = 0

    @property
    def trial_s(self) -> float:
        return self.fixation_s + self.cue_s + self.imagery_s + self.rest_min_s

    @property
    def n_trials(self) -> int:
        return self.n_trials_per_class * len(CLASS_TO_INT)

    def estimated_duration_s(self) -> float:
        mean_rest = (self.rest_min_s + self.rest_max_s) / 2
        per_trial = self.fixation_s + self.cue_s + self.imagery_s + mean_rest
        return self.n_trials * per_trial

    def build_sequence(self) -> list[str]:
        """Balanced, shuffled class order.

        Shuffled rather than alternating: an alternating L/R/L/R sequence is
        predictable, so the subject prepares the next movement during rest and
        the "rest" period stops being neutral.
        """
        rng = np.random.default_rng(self.seed)
        seq = [name for name in CLASS_TO_INT for _ in range(self.n_trials_per_class)]
        rng.shuffle(seq)
        return seq


@dataclass
class TrialRecord:
    """One cue: when it fired (seconds into the recording) and what was asked."""

    onset_s: float
    label: str
    label_int: int


@dataclass
class SessionMetadata:
    """Everything needed to interpret a recording months later."""

    subject: str
    device_key: str
    device_name: str
    sfreq: float
    ch_names: list[str]
    task: str  # "imagery" or "executed"
    started_utc: str
    protocol: dict
    trials: list[dict] = field(default_factory=list)
    notes: str = ""


# --------------------------------------------------------------------------
# Recording
# --------------------------------------------------------------------------


class CalibrationRecorder:
    """Runs a cued session against any :class:`EEGSource` and saves the result.

    ``cue_callback`` receives ``(phase, label, seconds_remaining)`` so the
    presentation layer is pluggable — a terminal printer by default, but a
    pygame window or a browser page can be swapped in without touching the
    timing logic. Keeping presentation out of here matters because the cue
    timing is the one thing in the whole system that must not drift: labels are
    assigned by timestamp, so a late cue silently mislabels a trial.
    """

    def __init__(
        self,
        source: EEGSource,
        protocol: CueProtocol | None = None,
        *,
        subject: str = "self",
        device_key: str = "unknown",
        task: str = "imagery",
        cue_callback=None,
        poll_interval_s: float = 0.05,
    ) -> None:
        self.source = source
        self.protocol = protocol or CueProtocol()
        self.subject = subject
        self.device_key = device_key
        self.task = task
        self.cue_callback = cue_callback or _print_cue
        self.poll_interval_s = poll_interval_s

    def run(self) -> tuple[np.ndarray, SessionMetadata]:
        """Execute the session. Returns raw samples and the cue log.

        Cue onsets are recorded as offsets from the first sample rather than as
        wall-clock times, and the number of samples received is used as the
        clock. Wall-clock timing would drift against the device's own crystal;
        over a 15-minute session a 0.5% clock difference is 4.5 seconds, which
        is longer than a whole trial.
        """
        info = self.source.info
        sequence = self.protocol.build_sequence()
        chunks: list[np.ndarray] = []
        trials: list[TrialRecord] = []
        n_samples = 0

        started = datetime.now(timezone.utc).isoformat()
        rng = np.random.default_rng(self.protocol.seed + 1)

        def drain() -> None:
            nonlocal n_samples
            chunk = self.source.poll()
            if chunk.size:
                chunks.append(chunk)
                n_samples += chunk.shape[1]

        def wait(seconds: float, phase: str, label: str) -> None:
            deadline = time.monotonic() + seconds
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self.cue_callback(phase, label, remaining)
                time.sleep(min(self.poll_interval_s, remaining))
                drain()

        self.cue_callback("start", "", 0.0)
        drain()

        for i, label in enumerate(sequence):
            wait(self.protocol.fixation_s, "fixation", label)

            # The cue onset is the reference point for epoching, so it is
            # stamped from the sample count at the instant the cue appears.
            drain()
            onset_s = n_samples / info.sfreq
            trials.append(
                TrialRecord(onset_s=onset_s, label=label, label_int=CLASS_TO_INT[label])
            )

            # Imagery is timed from cue onset, matching the Graz convention and
            # matching EPOCH_TMIN/EPOCH_TMAX used for the PhysioNet data.
            wait(self.protocol.cue_s, "cue", label)
            wait(self.protocol.imagery_s, "imagery", label)
            rest = float(rng.uniform(self.protocol.rest_min_s, self.protocol.rest_max_s))
            wait(rest, "rest", "")

            self.cue_callback("progress", f"{i + 1}/{len(sequence)}", 0.0)

        # Tail: keep streaming past the final cue for at least a full epoch
        # window, otherwise the last trials extend beyond the end of the
        # recording and MNE silently drops them.
        wait(EPOCH_TMAX + 1.0, "rest", "")

        data = (
            np.concatenate(chunks, axis=1)
            if chunks
            else np.zeros((len(info.ch_names), 0))
        )

        meta = SessionMetadata(
            subject=self.subject,
            device_key=self.device_key,
            device_name=info.name,
            sfreq=info.sfreq,
            ch_names=list(info.ch_names),
            task=self.task,
            started_utc=started,
            protocol=asdict(self.protocol),
            trials=[asdict(t) for t in trials],
        )
        return data, meta


def _print_cue(phase: str, label: str, remaining: float) -> None:
    """Minimal terminal cue presentation.

    Deliberately blunt: a left/right arrow and nothing else. Anything animated
    or colourful on screen during the imagery window adds a visual evoked
    response that the classifier may learn instead of motor imagery.
    """
    symbols = {
        "start": "  ready...",
        "fixation": "        +",
        "cue": {"left": "  <<<<<  LEFT", "right": "  RIGHT  >>>>>"}.get(label, "   cue"),
        "imagery": {"left": "  <<<<<  imagine LEFT hand", "right": "  imagine RIGHT hand  >>>>>"}
        .get(label, "  imagine"),
        "rest": "     rest",
    }
    if phase == "progress":
        print(f"\r  trial {label} complete", end="", flush=True)
        return
    text = symbols.get(phase, phase)
    print(f"\r{text:<45s} {remaining:4.1f}s", end="", flush=True)


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------


def save_session(
    data: np.ndarray,
    meta: SessionMetadata,
    out_dir: Path = RECORDINGS_DIR,
) -> Path:
    """Write a session to ``<subject>_<device>_<timestamp>.npz`` plus a JSON sidecar.

    Raw, unfiltered samples are stored. Filtering is cheap and preprocessing
    choices change; a recording that has already been filtered at 8-30 Hz can
    never be reanalysed for, say, slow cortical potentials, and cannot be
    re-examined for the line noise that would have revealed a bad electrode.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = meta.started_utc.replace(":", "").replace("-", "").split(".")[0]
    stem = f"{meta.subject}_{meta.device_key}_{meta.task}_{stamp}"

    npz_path = out_dir / f"{stem}.npz"
    np.savez_compressed(npz_path, data=data)
    (out_dir / f"{stem}.json").write_text(json.dumps(asdict(meta), indent=2))
    return npz_path


def load_session(npz_path: Path) -> tuple[np.ndarray, SessionMetadata]:
    """Read back a saved calibration session."""
    npz_path = Path(npz_path)
    data = np.load(npz_path)["data"]
    meta_dict = json.loads(npz_path.with_suffix(".json").read_text())
    meta = SessionMetadata(**meta_dict)
    return data, meta


def session_to_xy(
    npz_path: Path,
    *,
    pick_channels: list[str] | None = None,
    reject_uv: float | None = REJECT_PEAK_TO_PEAK_UV,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load a session and turn it into ``(X, y, channel_names)`` training data.

    This is the join point where own-hardware recordings become
    indistinguishable from PhysioNet epochs, so every model and every
    evaluation routine works on them unchanged.
    """
    from .preprocess import epoch_windows, preprocess_raw, raw_from_array

    data, meta = load_session(npz_path)
    raw = raw_from_array(data, meta.sfreq, meta.ch_names)
    raw = preprocess_raw(raw, pick_channels=pick_channels)

    onsets = np.array([t["onset_s"] for t in meta.trials])
    labels = np.array([t["label_int"] for t in meta.trials])
    x, y = epoch_windows(raw, onsets, labels, reject_uv=reject_uv)
    return x, y, raw.ch_names
