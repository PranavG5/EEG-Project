"""Live-decoding demo: replay one subject's trials as if streaming from a BCI.

This is the "watch it work" entry point, meant for a screen-share or a talk. It
loads a single subject, gets an *honest* held-out prediction for every trial
(via stratified cross-validation, so no trial is predicted by a model that
trained on it), then replays the trials one at a time — printing, for each,
which hand the subject imagined, which hand the decoder guessed, how confident
it was, and a running accuracy. Because it uses the classical pipeline it runs
in a few seconds, so it is safe to run live in front of an audience.

Run from the repository root:

    python src/demo.py                       # subject 1, CSP + LDA
    python src/demo.py --subject 4 --method bandpower_lda
    python src/demo.py --method eegnet --delay 0.3   # slower, dramatic replay

Methods (all expose calibrated class probabilities for the confidence bar):
    csp_lda        CSP spatial filtering -> LDA        (the reference BCI method)
    bandpower_lda  per-channel mu/beta power -> LDA    (simplest baseline)
    eegnet         compact CNN, learned features       (slow: trains a net)
"""

from __future__ import annotations

import argparse
import time

import mne
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from features import band_power
from models import make_csp_lda, make_eegnet, make_lda
from preprocess import epochs_to_labels, load_subject_epochs

# method key -> (human name, needs raw epoch tensor?, estimator factory)
METHODS = {
    "csp_lda": ("CSP + LDA", True, lambda: make_csp_lda(n_components=6)),
    "bandpower_lda": ("Band power + LDA", False, make_lda),
    "eegnet": ("EEGNet", True, lambda: make_eegnet(sfreq=160.0)),
}

# Map class name -> a short label and an arrow, purely for the console display.
HAND_DISPLAY = {
    "left_fist": ("LEFT  hand", "<--"),
    "right_fist": ("RIGHT hand", "-->"),
}


def confidence_bar(prob: float, width: int = 20) -> str:
    """Render a probability in [0, 1] as a simple text meter, e.g. ``|#####___|``."""
    filled = int(round(prob * width))
    return "|" + "#" * filled + "-" * (width - filled) + "|"


def run_demo(subject: int, method: str, delay: float) -> None:
    mne.set_log_level("WARNING")
    name, needs_epochs, factory = METHODS[method]

    print(f"\nLoading subject {subject} (imagery runs 4/8/12) and calibrating "
          f"the '{name}' decoder...")
    epochs = load_subject_epochs(subject, runs=(4, 8, 12))
    y, class_names = epochs_to_labels(epochs)

    # Feature representation: the raw epoch tensor for CSP/EEGNet, or the
    # precomputed per-channel band-power matrix for the simple baseline.
    X = epochs.get_data(copy=False) if needs_epochs else band_power(epochs)

    # Honest per-trial predictions: 5-fold stratified CV means every trial is
    # decoded by a model that never saw it. We also grab class probabilities so
    # the demo can show the decoder's confidence, not just its final guess.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    proba = cross_val_predict(factory(), X, y, cv=cv, method="predict_proba")
    preds = proba.argmax(axis=1)

    n = len(y)
    print(f"Calibrated on {n} trials. Replaying them as a live decode "
          f"(chance = {max(np.bincount(y)) / n:.0%})...\n")
    print(f"{'trial':>5}  {'imagined':<11} {'decoded':<11} {'confidence':<24} "
          f"{'':<4} running")
    print("-" * 72)

    correct = 0
    for i in range(n):
        true_name = class_names[y[i]]
        pred_name = class_names[preds[i]]
        conf = proba[i, preds[i]]
        hit = preds[i] == y[i]
        correct += hit

        true_lbl, _ = HAND_DISPLAY[true_name]
        pred_lbl, arrow = HAND_DISPLAY[pred_name]
        mark = "OK " if hit else " X "
        print(f"{i + 1:>5}  {true_lbl:<11} {pred_lbl:<11} "
              f"{confidence_bar(conf)} {conf:>4.0%} {mark}  "
              f"{correct}/{i + 1} = {correct / (i + 1):.0%}")
        if delay:
            time.sleep(delay)

    acc = correct / n
    print("-" * 72)
    print(f"\nFinal held-out accuracy for subject {subject} "
          f"with {name}: {acc:.1%}  ({correct}/{n} trials)\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subject", type=int, default=1, help="EEGMMIDB subject id (1-109)")
    p.add_argument("--method", choices=list(METHODS), default="csp_lda",
                   help="decoder to calibrate and replay")
    p.add_argument("--delay", type=float, default=0.0,
                   help="seconds to pause between trials (e.g. 0.3 for a live feel)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_demo(subject=args.subject, method=args.method, delay=args.delay)
