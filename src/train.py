"""End-to-end training/evaluation entry point.

Run from the repository root:

    python src/train.py

Milestone 2 pipeline: load subject 1 / run 4 -> preprocess (8-30 Hz band-pass +
common average reference) -> epoch T1/T2 trials -> band-power features ->
shrinkage LDA -> subject-dependent cross-validated accuracy + confusion matrix.

Later milestones extend this driver with CSP features, an SVM, EEGNet, and
cross-subject evaluation.
"""

from __future__ import annotations

import os

import numpy as np

from evaluate import evaluate_subject_dependent
from features import band_power
from models import make_lda
from preprocess import load_raw, make_epochs, preprocess_raw

FIGURES_DIR = "results/figures"
LOCAL_EDF = "data/S001R04.edf"


def epochs_to_xy(epochs) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Turn epochs into (labels y, class_names), with y as 0-based integers.

    The class-name order is fixed by sorting the epoch ``event_id`` mapping by
    its integer code, so label ``i`` always corresponds to ``class_names[i]``
    (here 0 = left_fist, 1 = right_fist).
    """
    # name -> integer event code, e.g. {"left_fist": 2, "right_fist": 3}
    name_by_code = {code: name for name, code in epochs.event_id.items()}
    ordered_codes = sorted(name_by_code)
    class_names = [name_by_code[c] for c in ordered_codes]
    code_to_label = {code: i for i, code in enumerate(ordered_codes)}

    y = np.array([code_to_label[c] for c in epochs.events[:, -1]])
    return y, class_names


def main() -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # --- Data: load, preprocess, epoch -------------------------------------
    edf_paths = [LOCAL_EDF] if os.path.exists(LOCAL_EDF) else None
    print("Loading subject 1, run 4 (imagined left/right fist)...")
    raw = load_raw(subject=1, runs=(4,), edf_paths=edf_paths)
    preprocess_raw(raw)
    epochs = make_epochs(raw)

    # --- Features: band power ----------------------------------------------
    X = band_power(epochs)
    y, class_names = epochs_to_xy(epochs)
    print(f"Feature matrix: {X.shape[0]} trials x {X.shape[1]} channels "
          f"(log band power, 8-30 Hz)")

    # --- Classify + evaluate (subject-dependent, CV) -----------------------
    results = evaluate_subject_dependent(
        X,
        y,
        make_lda(),
        class_names=class_names,
        method_name="Band power + LDA",
        n_splits=5,
        out_path=f"{FIGURES_DIR}/milestone2_confusion_bandpower_lda.png",
    )

    return results


if __name__ == "__main__":
    main()
