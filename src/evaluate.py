"""Evaluation utilities: cross-validated metrics, confusion matrices, topomaps.

Two evaluation regimes are provided:

- **Subject-dependent** — train and test within the same subject. With only a
  few dozen trials per subject a single split is very noisy, so we use stratified
  k-fold cross-validation (preserving the left/right balance in every fold).
  ``evaluate_subject_dependent`` runs this for one subject;
  ``evaluate_subject_dependent_grouped`` runs it per subject and averages across
  subjects for a robust "once calibrated to a user" number.
- **Cross-subject** — ``evaluate_cross_subject`` does leave-one-subject-out CV:
  train on all-but-one subject, test on the held-out one. This is the honest
  transfer test and is expected to be markedly harder.

Plus visualisation helpers: accuracy bar/grouped charts and the ERD/ERS topomap
that shows the spatial mu/beta power pattern distinguishing left vs right imagery.
"""

from __future__ import annotations

from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mne
import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import (
    LeaveOneGroupOut,
    StratifiedKFold,
    cross_val_predict,
)


def evaluate_subject_dependent(
    X: np.ndarray,
    y: np.ndarray,
    clf: BaseEstimator,
    class_names: Sequence[str],
    method_name: str = "model",
    n_splits: int = 5,
    out_path: str | None = None,
) -> dict:
    """Cross-validate ``clf`` within one subject and report classification metrics.

    Uses stratified k-fold CV so each fold holds the same left/right ratio as
    the full set. Predictions are collected out-of-fold via
    ``cross_val_predict`` so every trial is predicted exactly once by a model
    that never saw it, giving an honest aggregate confusion matrix and
    per-class precision/recall.

    Parameters
    ----------
    X
        Feature matrix, shape (n_trials, n_features).
    y
        Integer class labels, shape (n_trials,).
    clf
        An unfitted scikit-learn estimator/pipeline.
    class_names
        Human-readable names indexed by label value (e.g. ["left_fist",
        "right_fist"]).
    method_name
        Label used in printed output and the plot title.
    n_splits
        Number of stratified CV folds.
    out_path
        If given, the confusion-matrix figure is saved here.

    Returns
    -------
    dict
        Keys: ``accuracy`` (mean over trials), ``fold_accuracies``,
        ``confusion_matrix``, ``report`` (text), ``y_pred``.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Out-of-fold predictions: each trial predicted by a model that didn't train
    # on it. This is the basis for both the confusion matrix and the report.
    y_pred = cross_val_predict(clf, X, y, cv=cv)

    # Per-fold accuracy, to convey variance across the (small) dataset.
    fold_accuracies = []
    for _, test_idx in cv.split(X, y):
        fold_accuracies.append(accuracy_score(y[test_idx], y_pred[test_idx]))
    fold_accuracies = np.array(fold_accuracies)

    accuracy = accuracy_score(y, y_pred)
    cm = confusion_matrix(y, y_pred)
    report = classification_report(y, y_pred, target_names=class_names, digits=3)

    chance = max(np.bincount(y)) / len(y)
    print(f"\n=== Subject-dependent evaluation: {method_name} ===")
    print(f"  {n_splits}-fold CV accuracy : {accuracy:.3f} "
          f"(per-fold mean {fold_accuracies.mean():.3f} "
          f"± {fold_accuracies.std():.3f})")
    print(f"  majority-class chance   : {chance:.3f}  "
          f"(n={len(y)} trials)")
    print("  per-class metrics:")
    print("    " + report.replace("\n", "\n    "))

    if out_path is not None:
        disp = ConfusionMatrixDisplay(cm, display_labels=list(class_names))
        fig, ax = plt.subplots(figsize=(4.5, 4))
        disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
        ax.set_title(f"{method_name}\nsubject-dependent, {n_splits}-fold CV "
                     f"(acc {accuracy:.2f})")
        fig.tight_layout()
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        print(f"  saved confusion matrix -> {out_path}")

    return {
        "accuracy": accuracy,
        "fold_accuracies": fold_accuracies,
        "confusion_matrix": cm,
        "report": report,
        "y_pred": y_pred,
    }


def evaluate_subject_dependent_grouped(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    clf: BaseEstimator,
    class_names: Sequence[str],
    method_name: str = "model",
    n_splits: int = 5,
    out_path: str | None = None,
) -> dict:
    """Subject-dependent accuracy aggregated over many subjects.

    Runs stratified k-fold cross-validation *within* each subject separately
    (train and test always come from the same person) and then summarises across
    subjects as mean ± std of the per-subject accuracies. This is the standard
    "how well does it work once calibrated to a user" number, made robust by
    averaging over subjects instead of trusting a single one, and it is the
    natural baseline against which the cross-subject (transfer) accuracy is
    compared.

    A fresh clone of ``clf`` is fit on every fold of every subject, so supervised
    front-ends (CSP, EEGNet) never see their own test trials. Out-of-fold
    predictions are pooled across all subjects for one aggregate confusion matrix.

    Parameters
    ----------
    X
        Feature/epoch tensor, shape (n_trials, ...).
    y
        Integer class labels.
    groups
        Subject id per trial.
    clf
        Unfitted estimator/pipeline (cloned per fold).
    class_names
        Class names indexed by label value.
    method_name
        Label for printed output / plot title.
    n_splits
        Stratified CV folds per subject.
    out_path
        If given, the aggregate confusion-matrix figure is saved here.

    Returns
    -------
    dict
        Keys: ``accuracy`` (mean over subjects), ``fold_accuracies`` (per
        subject), ``subjects``, ``confusion_matrix``, ``report``, ``y_pred``.
    """
    y_pred = np.empty_like(y)
    subjects: list[int] = []
    per_subject_acc: list[float] = []

    for subj in np.unique(groups):
        mask = groups == subj
        Xs, ys = X[mask], y[mask]
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        preds = cross_val_predict(clone(clf), Xs, ys, cv=cv)
        y_pred[mask] = preds
        subjects.append(int(subj))
        per_subject_acc.append(accuracy_score(ys, preds))

    fold_accuracies = np.array(per_subject_acc)
    accuracy = fold_accuracies.mean()
    cm = confusion_matrix(y, y_pred)
    report = classification_report(y, y_pred, target_names=class_names, digits=3)
    chance = max(np.bincount(y)) / len(y)

    print(f"\n=== Subject-dependent ({n_splits}-fold CV, "
          f"{len(subjects)} subjects): {method_name} ===")
    print(f"  mean subject accuracy : {accuracy:.3f} "
          f"± {fold_accuracies.std():.3f}")
    print(f"  majority-class chance  : {chance:.3f}  (n={len(y)} trials)")
    print("  per-class metrics (pooled):")
    print("    " + report.replace("\n", "\n    "))

    if out_path is not None:
        disp = ConfusionMatrixDisplay(cm, display_labels=list(class_names))
        fig, ax = plt.subplots(figsize=(4.5, 4))
        disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
        ax.set_title(f"{method_name}\nsubject-dependent, {n_splits}-fold CV "
                     f"(mean acc {accuracy:.2f})")
        fig.tight_layout()
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        print(f"  saved confusion matrix -> {out_path}")

    return {
        "accuracy": accuracy,
        "fold_accuracies": fold_accuracies,
        "subjects": subjects,
        "confusion_matrix": cm,
        "report": report,
        "y_pred": y_pred,
    }


def plot_erd_ers_topomap(
    epochs: mne.Epochs,
    out_path: str,
    left_class: str = "left_fist",
    right_class: str = "right_fist",
    fmin: float = 8.0,
    fmax: float = 30.0,
) -> str:
    """Scalp maps of mu/beta power for left vs right imagery and their contrast.

    This is the figure that visually reads as neuroscience rather than generic
    ML. Motor imagery causes event-related desynchronisation (ERD): the 8-30 Hz
    (mu/beta) rhythm *drops* in power over the sensorimotor cortex
    *contralateral* to the imagined hand. So imagining the left hand should
    suppress power near the right-hemisphere hand area (electrode C4), and
    imagining the right hand should suppress it near the left hemisphere (C3).

    Three maps are drawn:

    - **left-hand imagery** and **right-hand imagery**: average log 8-30 Hz power
      per electrode for each class (shared colour scale), showing where the
      rhythm sits during each condition;
    - **left − right**: the lateralisation contrast. Because the suppression is
      contralateral, this difference map should be spatially antisymmetric across
      the midline around C3/C4 — the direct spatial signature that the two
      classes are physiologically distinguishable, and exactly the structure the
      CSP and depthwise-conv spatial filters exploit.

    Power is computed per trial with Welch's method, averaged over the band and
    then over trials within each class, and log-scaled (power is log-normal).

    Parameters
    ----------
    epochs
        Preprocessed, montaged epochs containing both classes.
    out_path
        Where to save the figure.
    left_class, right_class
        Event-id names for the two imagery conditions.
    fmin, fmax
        Band over which to integrate power (mu + beta).

    Returns
    -------
    str
        The path the figure was written to.
    """
    def class_log_power(name: str) -> tuple[np.ndarray, mne.Info]:
        spectrum = epochs[name].compute_psd(
            method="welch", fmin=fmin, fmax=fmax, verbose="WARNING"
        )
        psds = spectrum.get_data()          # (n_trials, n_channels, n_freqs)
        band = psds.mean(axis=2)            # (n_trials, n_channels)
        return np.log10(band).mean(axis=0), spectrum.info

    left_power, info = class_log_power(left_class)
    right_power, _ = class_log_power(right_class)
    diff = left_power - right_power

    # Shared symmetric scale for the two per-class maps so they are comparable.
    vmax = float(np.max(np.abs([left_power - left_power.mean(),
                                right_power - right_power.mean()])))
    # Centre each class map on its own mean so the *spatial* pattern (not the
    # overall offset) is what the colour encodes.
    left_c = left_power - left_power.mean()
    right_c = right_power - right_power.mean()
    dmax = float(np.max(np.abs(diff)))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    for ax, data, title, vlim, cmap in (
        (axes[0], left_c, f"Left-hand imagery\n(log {fmin:.0f}-{fmax:.0f} Hz power)",
         vmax, "RdBu_r"),
        (axes[1], right_c, f"Right-hand imagery\n(log {fmin:.0f}-{fmax:.0f} Hz power)",
         vmax, "RdBu_r"),
        (axes[2], diff, "Left − Right\n(lateralisation contrast)", dmax, "RdBu_r"),
    ):
        im, _ = mne.viz.plot_topomap(
            data, info, axes=ax, show=False, cmap=cmap,
            vlim=(-vlim, vlim), contours=4, sensors=True,
        )
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.06)

    fig.suptitle("ERD/ERS spatial distribution — mu/beta power during motor imagery",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def evaluate_cross_subject(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    clf: BaseEstimator,
    class_names: Sequence[str],
    method_name: str = "model",
    out_path: str | None = None,
) -> dict:
    """Leave-one-subject-out evaluation: the honest test of generalisation.

    For each subject in turn, the classifier is trained on *all other* subjects
    and tested on the held-out one, so no trial from the test subject is ever
    seen in training. This measures whether the decoder captures motor-imagery
    structure that transfers *across brains*, rather than fitting one person's
    idiosyncratic signal. It is substantially harder than subject-dependent
    evaluation — inter-subject differences in anatomy, electrode placement and
    strategy mean accuracy typically drops toward chance — and that gap is the
    point of the comparison, not a bug to fix.

    A fresh clone of ``clf`` is fit on every fold (so supervised steps such as
    CSP or EEGNet never leak the held-out subject), and predictions are pooled
    across folds for the aggregate confusion matrix while accuracy is summarised
    as the mean ± std over subjects (each subject weighted equally).

    Parameters
    ----------
    X
        Feature/epoch tensor, shape (n_trials, ...). Passed through to ``clf``.
    y
        Integer class labels, shape (n_trials,).
    groups
        Subject id per trial; defines the leave-one-out folds.
    clf
        An unfitted scikit-learn estimator/pipeline (cloned per fold).
    class_names
        Human-readable names indexed by label value.
    method_name
        Label used in printed output and the plot title.
    out_path
        If given, the aggregate confusion-matrix figure is saved here.

    Returns
    -------
    dict
        Keys: ``accuracy`` (mean over subjects), ``accuracy_pooled`` (over all
        trials), ``fold_accuracies`` (per-subject), ``subjects`` (fold order),
        ``confusion_matrix``, ``report``, ``y_pred``.
    """
    logo = LeaveOneGroupOut()
    y_pred = np.empty_like(y)
    subjects: list[int] = []
    fold_accuracies: list[float] = []

    for train_idx, test_idx in logo.split(X, y, groups):
        est = clone(clf)
        est.fit(X[train_idx], y[train_idx])
        pred = est.predict(X[test_idx])
        y_pred[test_idx] = pred

        subj = int(groups[test_idx][0])
        subjects.append(subj)
        fold_accuracies.append(accuracy_score(y[test_idx], pred))

    fold_accuracies = np.array(fold_accuracies)
    accuracy = fold_accuracies.mean()            # each subject weighted equally
    accuracy_pooled = accuracy_score(y, y_pred)  # each trial weighted equally
    cm = confusion_matrix(y, y_pred)
    report = classification_report(y, y_pred, target_names=class_names, digits=3)
    chance = max(np.bincount(y)) / len(y)

    print(f"\n=== Cross-subject (leave-one-subject-out): {method_name} ===")
    print(f"  mean subject accuracy : {accuracy:.3f} "
          f"± {fold_accuracies.std():.3f} over {len(subjects)} subjects")
    print(f"  pooled-trial accuracy : {accuracy_pooled:.3f}")
    print(f"  majority-class chance  : {chance:.3f}  (n={len(y)} trials)")
    print("  per-subject accuracy:")
    for subj, acc in zip(subjects, fold_accuracies):
        print(f"    S{subj:03d}: {acc:.3f}")
    print("  per-class metrics (pooled):")
    print("    " + report.replace("\n", "\n    "))

    if out_path is not None:
        disp = ConfusionMatrixDisplay(cm, display_labels=list(class_names))
        fig, ax = plt.subplots(figsize=(4.5, 4))
        disp.plot(ax=ax, cmap="Purples", colorbar=False, values_format="d")
        ax.set_title(f"{method_name}\ncross-subject LOSO "
                     f"(mean acc {accuracy:.2f})")
        fig.tight_layout()
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        print(f"  saved confusion matrix -> {out_path}")

    return {
        "accuracy": accuracy,
        "accuracy_pooled": accuracy_pooled,
        "fold_accuracies": fold_accuracies,
        "subjects": subjects,
        "confusion_matrix": cm,
        "report": report,
        "y_pred": y_pred,
    }


def plot_subjdep_vs_crosssubj(
    results_sd: dict[str, dict],
    results_cs: dict[str, dict],
    out_path: str,
    chance: float | None = None,
    title: str = "Subject-dependent vs cross-subject accuracy",
) -> str:
    """Grouped bar chart contrasting within-subject and across-subject accuracy.

    Draws two bars per method — subject-dependent and cross-subject — so the
    generalisation gap (how much accuracy is lost when the decoder must transfer
    to an unseen brain) is visible at a glance for every method.

    Parameters
    ----------
    results_sd, results_cs
        Mappings of method label -> result dict (must share the same keys and
        each contain ``accuracy`` and ``fold_accuracies``).
    out_path
        Where to save the figure.
    chance
        If given, a dashed line marks majority-class chance.
    title
        Figure title.

    Returns
    -------
    str
        The path the figure was written to.
    """
    labels = list(results_sd)
    x = np.arange(len(labels))
    width = 0.38

    sd_acc = [results_sd[k]["accuracy"] for k in labels]
    sd_err = [results_sd[k]["fold_accuracies"].std() for k in labels]
    cs_acc = [results_cs[k]["accuracy"] for k in labels]
    cs_err = [results_cs[k]["fold_accuracies"].std() for k in labels]

    fig, ax = plt.subplots(figsize=(1.7 * len(labels) + 2.5, 4.8))
    b1 = ax.bar(x - width / 2, sd_acc, width, yerr=sd_err, capsize=4,
                label="subject-dependent", color="tab:blue", alpha=0.85)
    b2 = ax.bar(x + width / 2, cs_acc, width, yerr=cs_err, capsize=4,
                label="cross-subject (LOSO)", color="tab:purple", alpha=0.85)
    if chance is not None:
        ax.axhline(chance, ls="--", color="gray", label=f"chance ({chance:.2f})")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend()
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02, f"{h:.2f}",
                    ha="center", va="bottom", fontsize=8)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_accuracy_comparison(
    results: dict[str, dict],
    out_path: str,
    chance: float | None = None,
    title: str = "Subject-dependent accuracy",
) -> str:
    """Bar chart comparing methods' CV accuracy, with per-fold error bars.

    Parameters
    ----------
    results
        Mapping of method label -> result dict (as returned by
        ``evaluate_subject_dependent``; must contain ``accuracy`` and
        ``fold_accuracies``).
    out_path
        Where to save the figure.
    chance
        If given, a dashed horizontal line marks the majority-class chance level
        so bars can be read against it.
    title
        Figure title.

    Returns
    -------
    str
        The path the figure was written to.
    """
    labels = list(results)
    accs = [results[k]["accuracy"] for k in labels]
    errs = [results[k]["fold_accuracies"].std() for k in labels]

    fig, ax = plt.subplots(figsize=(1.6 * len(labels) + 2, 4.5))
    bars = ax.bar(labels, accs, yerr=errs, capsize=5, color="tab:blue",
                  alpha=0.85)
    if chance is not None:
        ax.axhline(chance, ls="--", color="gray",
                   label=f"chance ({chance:.2f})")
        ax.legend()
    ax.set_ylabel("CV accuracy")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, acc + 0.02,
                f"{acc:.2f}", ha="center", va="bottom")
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
