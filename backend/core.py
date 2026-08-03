"""Decoder registry and decode logic for the EEG BCI API.

This module is the extensible heart of the backend. Everything the API can do is
driven by the ``DECODERS`` registry and the small set of functions below, which
reuse the project's existing ``src/`` pipeline (preprocessing, features, models,
evaluation) rather than re-implementing anything.

To add a new decoder, add one entry to ``DECODERS``. To add a new data source or
output, extend :func:`decode` — nothing in the HTTP layer needs to change.
"""

from __future__ import annotations

import base64
import os
import sys
import tempfile
from dataclasses import dataclass
from typing import Callable

import matplotlib

matplotlib.use("Agg")

import mne
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict

# Make the project's src/ importable regardless of where uvicorn is launched.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from evaluate import plot_erd_ers_topomap  # noqa: E402
from features import band_power  # noqa: E402
from models import make_csp_lda, make_csp_svm, make_eegnet, make_lda  # noqa: E402
from preprocess import (  # noqa: E402
    epochs_from_edf_paths,
    epochs_to_labels,
    load_subject_epochs,
)

mne.set_log_level("WARNING")

# Demo subjects that are known-good 64ch/160Hz left/right imagery recordings.
DEMO_SUBJECTS = list(range(1, 11))


@dataclass(frozen=True)
class Decoder:
    """One selectable decoding pipeline.

    Attributes
    ----------
    key
        Stable identifier used by the API.
    name
        Human-readable name for the UI.
    description
        One-line explanation shown in the UI.
    needs_epochs
        True if the estimator consumes the raw 3-D epoch tensor (CSP, EEGNet);
        False if it consumes the 2-D band-power feature matrix.
    factory
        Zero-of-one-arg callable ``factory(sfreq) -> estimator`` returning a
        fresh unfitted scikit-learn estimator.
    """

    key: str
    name: str
    description: str
    needs_epochs: bool
    factory: Callable[[float], object]


# --- The registry: add a line here to expose a new decoder ------------------
DECODERS: dict[str, Decoder] = {
    "csp_lda": Decoder(
        "csp_lda", "CSP + LDA",
        "Common Spatial Patterns spatial filtering into shrinkage LDA — the "
        "standard reference BCI pipeline.",
        needs_epochs=True, factory=lambda sfreq: make_csp_lda(n_components=6),
    ),
    "bandpower_lda": Decoder(
        "bandpower_lda", "Band power + LDA",
        "Per-channel log mu/beta power into LDA — the simplest interpretable "
        "baseline.",
        needs_epochs=False, factory=lambda sfreq: make_lda(),
    ),
    "csp_svm": Decoder(
        "csp_svm", "CSP + SVM (RBF)",
        "CSP features into a non-linear RBF support-vector machine.",
        needs_epochs=True, factory=lambda sfreq: make_csp_svm(n_components=6),
    ),
    "eegnet": Decoder(
        "eegnet", "EEGNet (CNN)",
        "Compact convolutional network that learns temporal + spatial filters "
        "end-to-end (~2.7K parameters).",
        needs_epochs=True, factory=lambda sfreq: make_eegnet(sfreq=sfreq),
    ),
}


def list_decoders() -> list[dict]:
    """Return decoder metadata for the API (order preserved)."""
    return [
        {"key": d.key, "name": d.name, "description": d.description}
        for d in DECODERS.values()
    ]


# --- Data acquisition -------------------------------------------------------

def epochs_for_demo_subject(subject: int) -> mne.Epochs:
    if subject not in DEMO_SUBJECTS:
        raise ValueError(f"Demo subject must be one of {DEMO_SUBJECTS}.")
    return load_subject_epochs(subject, runs=(4, 8, 12))


def epochs_for_uploaded_files(files: list[tuple[str, bytes]]) -> mne.Epochs:
    """Build epochs from uploaded EDF (filename, bytes) pairs.

    Files are written to a temporary directory and run through the same
    load -> band-pass + CAR -> epoch chain as the demo subjects.
    """
    if not files:
        raise ValueError("No files were uploaded.")
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for fname, data in files:
            safe = os.path.basename(fname) or "upload.edf"
            p = os.path.join(tmp, safe)
            with open(p, "wb") as fh:
                fh.write(data)
            paths.append(p)
        return epochs_from_edf_paths(paths)


# --- Decoding ---------------------------------------------------------------

def _topomap_data_uri(epochs: mne.Epochs) -> str | None:
    """Render the ERD/ERS topomap and return it as a base64 PNG data URI."""
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as fh:
            path = fh.name
        plot_erd_ers_topomap(epochs, path)
        with open(path, "rb") as fh:
            encoded = base64.b64encode(fh.read()).decode("ascii")
        os.unlink(path)
        return f"data:image/png;base64,{encoded}"
    except Exception:
        # Topomap is a nice-to-have; never fail the whole decode over it.
        return None


def decode(epochs: mne.Epochs, decoder_key: str, source_label: str) -> dict:
    """Calibrate ``decoder_key`` on ``epochs`` and return a JSON-ready result.

    Uses 5-fold stratified cross-validation so every trial is predicted by a
    model that never trained on it (an honest accuracy). Returns everything the
    UI needs: summary, per-trial predictions with confidence, confusion matrix,
    and the ERD/ERS topomap as an embedded image.
    """
    if decoder_key not in DECODERS:
        raise ValueError(f"Unknown decoder '{decoder_key}'. "
                         f"Choose from {list(DECODERS)}.")
    decoder = DECODERS[decoder_key]

    y, class_names = epochs_to_labels(epochs)
    if len(np.unique(y)) < 2:
        raise ValueError(
            "The recording does not contain both left (T1) and right (T2) "
            "imagery trials, so there is nothing to classify."
        )

    sfreq = float(epochs.info["sfreq"])
    X = epochs.get_data(copy=False) if decoder.needs_epochs else band_power(epochs)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    proba = cross_val_predict(decoder.factory(sfreq), X, y, cv=cv,
                              method="predict_proba")
    preds = proba.argmax(axis=1)

    chance = float(max(np.bincount(y)) / len(y))
    accuracy = float((preds == y).mean())
    cm = confusion_matrix(y, preds).tolist()

    trials = [
        {
            "index": int(i + 1),
            "imagined": class_names[y[i]],
            "decoded": class_names[preds[i]],
            "confidence": float(proba[i, preds[i]]),
            "correct": bool(preds[i] == y[i]),
        }
        for i in range(len(y))
    ]

    return {
        "decoder": decoder.key,
        "decoder_name": decoder.name,
        "source": source_label,
        "n_trials": int(len(y)),
        "n_channels": int(len(epochs.ch_names)),
        "sfreq": sfreq,
        "chance": chance,
        "accuracy": accuracy,
        "class_names": list(class_names),
        "confusion_matrix": cm,
        "trials": trials,
        "topomap_png": _topomap_data_uri(epochs),
    }
