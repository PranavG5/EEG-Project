"""End-to-end training/evaluation entry point.

Run from the repository root:

    python src/train.py

Full pipeline over the first ``N_SUBJECTS`` PhysioNet EEGMMIDB subjects
(imagery runs 4, 8, 12):

    load + preprocess (8-30 Hz band-pass + common average reference) + epoch
        -> four decoders:
             1. band power + LDA        (simplest baseline)
             2. CSP + LDA               (reference BCI pipeline)
             3. CSP + RBF-SVM           (non-linear variant)
             4. EEGNet                  (compact CNN, learned features)
        -> two evaluation regimes:
             A. subject-dependent  (within-subject k-fold CV, per subject,
                                     averaged across subjects)
             B. cross-subject      (leave-one-subject-out — the transfer test)
        -> figures: accuracy comparison, confusion matrices, ERD/ERS topomap.

Data downloads automatically via MNE on first run (cached under ~/mne_data).

The whole run trains EEGNet many times on CPU, so it takes a while; progress is
printed per method. Set ``N_SUBJECTS`` lower, or ``RUN_EEGNET = False``, for a
quick classical-only pass.
"""

from __future__ import annotations

import os
import time

import matplotlib
import mne
import numpy as np

matplotlib.use("Agg")

from evaluate import (  # noqa: E402
    evaluate_cross_subject,
    evaluate_subject_dependent_grouped,
    plot_accuracy_comparison,
    plot_erd_ers_topomap,
    plot_subjdep_vs_crosssubj,
)
from features import band_power  # noqa: E402
from models import make_csp_lda, make_csp_svm, make_eegnet, make_lda  # noqa: E402
from preprocess import (  # noqa: E402
    load_subject_epochs,
    stack_subject_epochs,
)

FIGURES_DIR = "results/figures"

# First N subjects of EEGMMIDB and their imagined left/right-fist runs. All are
# 64-channel, 160 Hz recordings, so their epochs stack without resampling.
N_SUBJECTS = 10
SUBJECTS = tuple(range(1, N_SUBJECTS + 1))
IMAGERY_RUNS = (4, 8, 12)
SFREQ = 160.0

# EEGNet is the expensive part (trained once per CV fold on CPU). Flip off for a
# fast classical-only run.
RUN_EEGNET = True

# Human-readable labels for tables/plots.
DISPLAY_NAMES = {
    "bandpower_lda": "Band power\n+ LDA",
    "csp_lda": "CSP + LDA",
    "csp_svm": "CSP + SVM",
    "eegnet": "EEGNet",
}


def build_methods() -> dict[str, dict]:
    """Return the method registry: how to featurise and which estimator to use.

    Each entry declares ``feature`` — either ``"epochs"`` (feed the raw 3-D epoch
    tensor, used by CSP and EEGNet) or ``"bandpower"`` (the precomputed 2-D
    per-channel log-power matrix) — and a zero-arg ``factory`` returning a fresh
    unfitted estimator.
    """
    methods = {
        "bandpower_lda": {"feature": "bandpower", "factory": make_lda,
                          "name": "Band power + LDA"},
        "csp_lda": {"feature": "epochs",
                    "factory": lambda: make_csp_lda(n_components=6),
                    "name": "CSP + LDA"},
        "csp_svm": {"feature": "epochs",
                    "factory": lambda: make_csp_svm(n_components=6),
                    "name": "CSP + SVM (RBF)"},
    }
    if RUN_EEGNET:
        methods["eegnet"] = {
            "feature": "epochs",
            "factory": lambda: make_eegnet(sfreq=SFREQ),
            "name": "EEGNet",
        }
    return methods


def main() -> dict:
    mne.set_log_level("WARNING")
    os.makedirs(FIGURES_DIR, exist_ok=True)
    t_start = time.time()

    # --- Load, preprocess, epoch every subject once ------------------------
    print(f"Loading {len(SUBJECTS)} subjects {SUBJECTS}, runs {IMAGERY_RUNS} "
          "(download on first run, cached thereafter)...")
    epochs_list = [load_subject_epochs(s, runs=IMAGERY_RUNS) for s in SUBJECTS]
    X_epochs, y, groups, class_names = stack_subject_epochs(epochs_list, SUBJECTS)
    print(f"  dataset: {X_epochs.shape[0]} trials x {X_epochs.shape[1]} channels "
          f"x {X_epochs.shape[2]} samples; classes {class_names}; "
          f"balance {np.bincount(y).tolist()}")

    # Band-power features (2-D, label-free) computed per subject and stacked in
    # the same trial order as the epoch tensor.
    X_bandpower = np.concatenate([band_power(ep) for ep in epochs_list], axis=0)

    features = {"epochs": X_epochs, "bandpower": X_bandpower}
    methods = build_methods()
    chance = max(np.bincount(y)) / len(y)

    # --- Evaluation A: subject-dependent (within-subject k-fold) -----------
    print("\n" + "=" * 70)
    print("A. SUBJECT-DEPENDENT EVALUATION (5-fold CV within each subject)")
    print("=" * 70)
    results_sd: dict[str, dict] = {}
    for key, spec in methods.items():
        t0 = time.time()
        results_sd[key] = evaluate_subject_dependent_grouped(
            features[spec["feature"]], y, groups,
            spec["factory"](), class_names=class_names,
            method_name=spec["name"], n_splits=5,
            out_path=f"{FIGURES_DIR}/subject_dependent_confusion_{key}.png",
        )
        print(f"  ({spec['name']} done in {time.time() - t0:.0f}s)")

    # --- Evaluation B: cross-subject (leave-one-subject-out) ---------------
    print("\n" + "=" * 70)
    print("B. CROSS-SUBJECT EVALUATION (leave-one-subject-out)")
    print("=" * 70)
    results_cs: dict[str, dict] = {}
    for key, spec in methods.items():
        t0 = time.time()
        results_cs[key] = evaluate_cross_subject(
            features[spec["feature"]], y, groups,
            spec["factory"](), class_names=class_names,
            method_name=spec["name"],
            out_path=f"{FIGURES_DIR}/cross_subject_confusion_{key}.png",
        )
        print(f"  ({spec['name']} done in {time.time() - t0:.0f}s)")

    # --- Summary table -----------------------------------------------------
    print("\n" + "=" * 70)
    print(f"RESULTS SUMMARY ({len(SUBJECTS)} subjects, {len(y)} trials, "
          f"chance {chance:.2f})")
    print("=" * 70)
    print(f"  {'method':<18s} {'subject-dependent':>20s} {'cross-subject':>18s}")
    for key in methods:
        sd = results_sd[key]
        cs = results_cs[key]
        print(f"  {key:<18s} "
              f"{sd['accuracy']:.3f} ± {sd['fold_accuracies'].std():.3f}   "
              f"    {cs['accuracy']:.3f} ± {cs['fold_accuracies'].std():.3f}")

    # --- Figures -----------------------------------------------------------
    plot_names = {k: DISPLAY_NAMES.get(k, k) for k in methods}

    sd_chart = plot_accuracy_comparison(
        {plot_names[k]: results_sd[k] for k in methods},
        out_path=f"{FIGURES_DIR}/accuracy_comparison_subject_dependent.png",
        chance=chance,
        title=f"Subject-dependent accuracy ({len(SUBJECTS)} subjects, 5-fold CV)",
    )
    cs_chart = plot_accuracy_comparison(
        {plot_names[k]: results_cs[k] for k in methods},
        out_path=f"{FIGURES_DIR}/accuracy_comparison_cross_subject.png",
        chance=chance,
        title=f"Cross-subject accuracy ({len(SUBJECTS)} subjects, LOSO)",
    )
    combo_chart = plot_subjdep_vs_crosssubj(
        {plot_names[k]: results_sd[k] for k in methods},
        {plot_names[k]: results_cs[k] for k in methods},
        out_path=f"{FIGURES_DIR}/accuracy_comparison_combined.png",
        chance=chance,
        title=f"Subject-dependent vs cross-subject ({len(SUBJECTS)} subjects)",
    )

    # Grand-average ERD/ERS topomap across all subjects (annotations are dropped
    # on concatenation; only the class events are needed here).
    combined_epochs = mne.concatenate_epochs(epochs_list, verbose="WARNING")
    topomap = plot_erd_ers_topomap(
        combined_epochs, out_path=f"{FIGURES_DIR}/erd_ers_topomap.png"
    )

    print(f"\nSaved figures:")
    for path in (sd_chart, cs_chart, combo_chart, topomap):
        print(f"  {path}")
    print(f"\nTotal runtime: {time.time() - t_start:.0f}s")

    return {"subject_dependent": results_sd, "cross_subject": results_cs}


if __name__ == "__main__":
    main()
