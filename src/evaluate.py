"""Evaluation utilities: cross-validated metrics and confusion-matrix plots.

Milestone 2 covers subject-dependent evaluation — train and test within a
single subject. With only a few dozen trials a single train/test split is very
noisy, so we use stratified k-fold cross-validation (preserving the left/right
class balance in every fold) and report the mean accuracy across folds together
with an aggregated confusion matrix. Cross-subject (leave-one-subject-out)
evaluation is added in a later milestone.
"""

from __future__ import annotations

from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict


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
