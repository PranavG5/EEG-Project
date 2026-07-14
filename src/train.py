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

import glob
import os

import mne
import numpy as np

from evaluate import evaluate_subject_dependent, plot_accuracy_comparison
from features import band_power
from models import make_csp_lda, make_csp_svm, make_lda
from preprocess import load_raw, make_epochs, preprocess_raw

FIGURES_DIR = "results/figures"
DATA_DIR = "data"
# Subject 1's imagined left/right-fist runs. Any of these present under data/
# are loaded and concatenated, so dropping in more runs raises the trial count
# with no code change.
SUBJECT = 1
IMAGERY_RUNS = (4, 8, 12)


def find_local_edfs(subject: int, runs) -> list[str]:
    """Return existing local EDF paths for the given subject/runs, if any.

    Looks for files named like ``S001R04.edf`` under ``data/`` (case-insensitive
    on the extension). Returns an empty list if none are present, in which case
    the caller falls back to MNE's downloader.
    """
    found = []
    for run in runs:
        matches = glob.glob(os.path.join(DATA_DIR, f"S{subject:03d}R{run:02d}.edf"))
        matches += glob.glob(os.path.join(DATA_DIR, f"S{subject:03d}R{run:02d}.EDF"))
        found.extend(sorted(set(matches)))
    return found


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
    # CSP re-fits every CV fold and logs covariance details at INFO; quiet it so
    # the metrics summary is readable.
    mne.set_log_level("WARNING")
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # --- Data: load, preprocess, epoch -------------------------------------
    edf_paths = find_local_edfs(SUBJECT, IMAGERY_RUNS) or None
    if edf_paths:
        print(f"Loading subject {SUBJECT} from {len(edf_paths)} local run(s): "
              f"{', '.join(os.path.basename(p) for p in edf_paths)}")
    else:
        print(f"Loading subject {SUBJECT}, runs {IMAGERY_RUNS} via MNE "
              "downloader...")
    raw = load_raw(subject=SUBJECT, runs=IMAGERY_RUNS, edf_paths=edf_paths)
    preprocess_raw(raw)
    epochs = make_epochs(raw)

    y, class_names = epochs_to_xy(epochs)
    results = {}

    # --- Method 1: band power + LDA ----------------------------------------
    # 2-D features (n_trials, n_channels): precomputed, label-free, so safe to
    # build once outside CV.
    X_bandpower = band_power(epochs)
    print(f"Band-power feature matrix: {X_bandpower.shape[0]} trials x "
          f"{X_bandpower.shape[1]} channels (log band power, 8-30 Hz)")
    results["bandpower_lda"] = evaluate_subject_dependent(
        X_bandpower,
        y,
        make_lda(),
        class_names=class_names,
        method_name="Band power + LDA",
        n_splits=5,
        out_path=f"{FIGURES_DIR}/milestone2_confusion_bandpower_lda.png",
    )

    # --- Method 2: CSP + LDA (reference BCI pipeline) ----------------------
    # CSP is supervised, so it lives inside the pipeline and is fit per-fold.
    # Its input is the raw 3-D epoch data (n_trials, n_channels, n_times).
    X_epochs = epochs.get_data(copy=False)
    print(f"CSP input tensor: {X_epochs.shape[0]} trials x "
          f"{X_epochs.shape[1]} channels x {X_epochs.shape[2]} samples")
    results["csp_lda"] = evaluate_subject_dependent(
        X_epochs,
        y,
        make_csp_lda(n_components=6),
        class_names=class_names,
        method_name="CSP + LDA",
        n_splits=5,
        out_path=f"{FIGURES_DIR}/milestone3_confusion_csp_lda.png",
    )

    # --- Method 3: CSP + RBF SVM -------------------------------------------
    results["csp_svm"] = evaluate_subject_dependent(
        X_epochs,
        y,
        make_csp_svm(n_components=6),
        class_names=class_names,
        method_name="CSP + SVM (RBF)",
        n_splits=5,
        out_path=f"{FIGURES_DIR}/milestone4_confusion_csp_svm.png",
    )

    # --- Summary comparison + bar chart ------------------------------------
    display_names = {
        "bandpower_lda": "Band power\n+ LDA",
        "csp_lda": "CSP + LDA",
        "csp_svm": "CSP + SVM",
    }
    print(f"\n=== Subject-dependent accuracy comparison "
          f"(S{SUBJECT:03d}, {len(y)} trials) ===")
    for name, res in results.items():
        print(f"  {name:<16s}: {res['accuracy']:.3f} "
              f"± {res['fold_accuracies'].std():.3f}")

    chance = max(np.bincount(y)) / len(y)
    chart = plot_accuracy_comparison(
        {display_names.get(k, k): v for k, v in results.items()},
        out_path=f"{FIGURES_DIR}/accuracy_comparison_subject_dependent.png",
        chance=chance,
        title=f"Subject-dependent accuracy (S{SUBJECT:03d}, {len(y)} trials, "
              "5-fold CV)",
    )
    print(f"Saved accuracy comparison chart -> {chart}")

    return results


if __name__ == "__main__":
    main()
