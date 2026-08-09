"""Metrics, cross-validation schemes and figures.

Two evaluation regimes, and the gap between them is the interesting result:

* **Subject-dependent** — train and test on the same person. This is what a
  calibrated BCI actually does, and what your headband will do after a
  calibration session.
* **Cross-subject (leave-one-subject-out)** — train on N-1 people, test on the
  held-out one. This is the "no calibration needed" dream, and accuracy drops
  substantially. That drop is a real property of EEG, not a bug: sulcal
  anatomy, skull thickness and imagery strategy all differ between people, so
  the spatial filters that are optimal for one are wrong for another.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from .config import CLASSES, FIGURES_DIR, RANDOM_SEED


@dataclass
class EvalResult:
    """Scores for one model under one evaluation scheme."""

    model_name: str
    scheme: str
    accuracy: float
    kappa: float
    confusion: np.ndarray
    per_fold: list[float] = field(default_factory=list)
    report: str = ""
    n_trials: int = 0
    n_channels: int = 0

    def summary(self) -> str:
        spread = (
            f" (folds {min(self.per_fold):.3f}-{max(self.per_fold):.3f})"
            if self.per_fold
            else ""
        )
        return (
            f"{self.model_name:16s} {self.scheme:18s} "
            f"acc={self.accuracy:.3f}{spread}  kappa={self.kappa:.3f}  "
            f"n={self.n_trials} trials x {self.n_channels} ch"
        )


def _score(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, np.ndarray, str]:
    acc = float(accuracy_score(y_true, y_pred))
    # Cohen's kappa rather than accuracy alone: it corrects for agreement by
    # chance, which is the standard reporting convention in the BCI literature
    # precisely because two-class accuracy near 50% is so easy to over-read.
    kappa = float(cohen_kappa_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES))))
    report = classification_report(
        y_true, y_pred, labels=list(range(len(CLASSES))),
        target_names=list(CLASSES), zero_division=0,
    )
    return acc, kappa, cm, report


def evaluate_subject_dependent(
    model_factory,
    x: np.ndarray,
    y: np.ndarray,
    *,
    model_name: str = "model",
    n_splits: int = 5,
    seed: int = RANDOM_SEED,
) -> EvalResult:
    """Stratified k-fold within one subject.

    ``model_factory`` is a callable returning a *fresh* unfitted estimator, not
    an estimator instance. This forces CSP (and EEGNet's weights) to be refit
    inside every fold; reusing one fitted object across folds is the label-leak
    that makes student BCI projects report 95%.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    y_pred = cross_val_predict(model_factory(), x, y, cv=cv, n_jobs=1)

    per_fold = []
    for _, test_idx in cv.split(x, y):
        per_fold.append(float(accuracy_score(y[test_idx], y_pred[test_idx])))

    acc, kappa, cm, report = _score(y, y_pred)
    return EvalResult(
        model_name=model_name,
        scheme=f"within-subject {n_splits}-fold",
        accuracy=acc,
        kappa=kappa,
        confusion=cm,
        per_fold=per_fold,
        report=report,
        n_trials=len(y),
        n_channels=x.shape[1],
    )


def evaluate_per_subject(
    model_factory,
    subject_data: dict[object, tuple[np.ndarray, np.ndarray]],
    *,
    model_name: str = "model",
    n_splits: int = 5,
    seed: int = RANDOM_SEED,
) -> EvalResult:
    """Within-subject CV run separately for each subject, then aggregated.

    This is the correct multi-subject version of subject-dependent evaluation,
    and it is *not* the same as pooling everyone's trials and running k-fold
    over the pool. Pooling lets a fold contain subject A's trials in both train
    and test while also containing subject B — the model can then learn one
    average spatial filter that suits nobody, which penalises exactly the
    methods (CSP) whose strength is per-subject adaptation.

    Measured on 10 PhysioNet subjects, that distinction moves CSP+LDA from
    60.7% (pooled) to 68.7% (per-subject) — the pooled number is not a harder
    version of the same question, it is a different and less relevant one,
    because a deployed BCI is always calibrated to one person.

    ``per_fold`` holds one accuracy per subject, so its spread shows
    between-subject variability — including the subjects who sit at chance.
    """
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    per_subject_acc: list[float] = []

    for subject in sorted(subject_data, key=str):
        x, y = subject_data[subject]
        counts = np.bincount(y, minlength=2)
        if counts.min() < n_splits:
            # Not enough trials of some class to stratify; skip rather than
            # silently reduce the fold count and report an incomparable number.
            continue
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        pred = cross_val_predict(model_factory(), x, y, cv=cv, n_jobs=1)
        all_true.append(y)
        all_pred.append(pred)
        per_subject_acc.append(float(accuracy_score(y, pred)))

    if not all_true:
        raise ValueError("No subject had enough trials per class to cross-validate.")

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    acc, kappa, cm, report = _score(y_true, y_pred)
    return EvalResult(
        model_name=model_name,
        scheme=f"within-subject {n_splits}-fold (n={len(per_subject_acc)})",
        accuracy=acc,
        kappa=kappa,
        confusion=cm,
        per_fold=per_subject_acc,
        report=report,
        n_trials=len(y_true),
        n_channels=next(iter(subject_data.values()))[0].shape[1],
    )


def evaluate_cross_subject(
    model_factory,
    subject_data: dict[int, tuple[np.ndarray, np.ndarray]],
    *,
    model_name: str = "model",
) -> EvalResult:
    """Leave-one-subject-out evaluation.

    Every subject takes a turn as the test set while the rest are pooled for
    training. Note that no per-subject normalisation is applied here beyond what
    the pipeline itself does — adding it (e.g. Euclidean alignment of the
    covariance matrices) is the standard trick for closing part of this gap and
    is a natural extension.
    """
    subjects = sorted(subject_data)
    if len(subjects) < 2:
        raise ValueError(
            f"Leave-one-subject-out needs >=2 subjects, got {len(subjects)}."
        )

    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    per_fold: list[float] = []

    for held_out in subjects:
        train_x = np.concatenate([subject_data[s][0] for s in subjects if s != held_out])
        train_y = np.concatenate([subject_data[s][1] for s in subjects if s != held_out])
        test_x, test_y = subject_data[held_out]

        model = model_factory()
        model.fit(train_x, train_y)
        pred = model.predict(test_x)

        all_true.append(test_y)
        all_pred.append(pred)
        per_fold.append(float(accuracy_score(test_y, pred)))

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    acc, kappa, cm, report = _score(y_true, y_pred)
    return EvalResult(
        model_name=model_name,
        scheme=f"cross-subject LOSO (n={len(subjects)})",
        accuracy=acc,
        kappa=kappa,
        confusion=cm,
        per_fold=per_fold,
        report=report,
        n_trials=len(y_true),
        n_channels=next(iter(subject_data.values()))[0].shape[1],
    )


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------


def plot_accuracy_comparison(
    results: list[EvalResult],
    out_path: Path | None = None,
    title: str = "Motor imagery decoding accuracy",
) -> Path:
    """Bar chart of every method, with the chance line drawn in.

    The chance line is the point of the figure. A 68% accuracy bar means
    nothing to a reader who does not know the task is two-class; drawn against
    50%, the same bar is immediately interpretable.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path or FIGURES_DIR / "accuracy_comparison.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    schemes = sorted({r.scheme for r in results}, reverse=True)
    models = list(dict.fromkeys(r.model_name for r in results))
    width = 0.8 / max(1, len(schemes))

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, scheme in enumerate(schemes):
        vals, errs = [], []
        for m in models:
            match = [r for r in results if r.model_name == m and r.scheme == scheme]
            vals.append(match[0].accuracy if match else np.nan)
            errs.append(
                np.std(match[0].per_fold) if match and match[0].per_fold else 0.0
            )
        pos = np.arange(len(models)) + i * width
        ax.bar(pos, vals, width=width, yerr=errs, capsize=3, label=scheme)

    ax.axhline(0.5, linestyle="--", color="0.4", linewidth=1.2, label="chance (2-class)")
    ax.set_xticks(np.arange(len(models)) + width * (len(schemes) - 1) / 2)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_confusion(
    result: EvalResult, out_path: Path | None = None
) -> Path:
    """Confusion matrix, annotated with counts and row-normalised colour."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(
        out_path or FIGURES_DIR / f"confusion_{result.model_name}.png"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cm = result.confusion
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, f"{cm[i, j]}\n{norm[i, j]:.0%}",
                ha="center", va="center",
                color="white" if norm[i, j] > 0.5 else "black",
                fontsize=11,
            )
    ax.set_xticks(range(len(CLASSES)), labels=[c.capitalize() for c in CLASSES])
    ax.set_yticks(range(len(CLASSES)), labels=[c.capitalize() for c in CLASSES])
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"{result.model_name}  ({result.accuracy:.1%})", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, label="proportion of true class")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_erd_topomap(
    epochs,
    out_path: Path | None = None,
    band: tuple[float, float] = (8.0, 13.0),
) -> Path:
    """ERD topomap: mu-band power change for left vs right imagery.

    This is the figure that shows the project decoded *neuroscience* rather
    than an arbitrary pattern. What should appear, if everything is working:
    imagining the LEFT hand produces a blue (negative, desynchronised) patch
    over the RIGHT hemisphere near C4, and imagining the RIGHT hand mirrors it
    over C3. The crossing is the signature of contralateral motor control.

    If the blobs are symmetric, midline, or frontal, the classifier is not
    reading motor cortex — check for eye artefact and electrode placement
    before trusting any accuracy number.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .features import erd_percentage

    out_path = Path(out_path or FIGURES_DIR / "erd_topomap.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sfreq = epochs.info["sfreq"]
    baseline = epochs.copy().crop(tmin=epochs.tmin, tmax=min(0.0, epochs.tmax))
    active = epochs.copy().crop(tmin=max(epochs.tmin, 0.5), tmax=epochs.tmax)

    fig, axes = plt.subplots(1, len(CLASSES), figsize=(4.4 * len(CLASSES), 4.0))
    images = []
    erds = {
        cls: erd_percentage(
            active[cls].get_data(copy=True),
            baseline[cls].get_data(copy=True),
            sfreq=sfreq,
            band=band,
        )
        for cls in CLASSES
        if cls in epochs.event_id
    }
    if not erds:
        raise ValueError(f"Epochs contain none of {CLASSES}: {sorted(epochs.event_id)}")

    vmax = float(np.max([np.abs(v).max() for v in erds.values()])) or 1.0

    import mne

    for ax, (cls, values) in zip(np.atleast_1d(axes), erds.items()):
        im, _ = mne.viz.plot_topomap(
            values, epochs.info, axes=ax, show=False,
            cmap="RdBu_r", vlim=(-vmax, vmax), contours=4,
        )
        images.append(im)
        ax.set_title(f"imagine {cls} hand", fontsize=11)

    fig.suptitle(
        f"Mu-band ({band[0]:g}-{band[1]:g} Hz) ERD — blue = power decrease = cortex engaged",
        fontsize=11,
    )
    fig.colorbar(images[0], ax=np.atleast_1d(axes).tolist(), fraction=0.05, label="% change from baseline")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def results_table(results: list[EvalResult]) -> str:
    """Markdown table, ready to paste into the README."""
    schemes = sorted({r.scheme for r in results}, reverse=True)
    models = list(dict.fromkeys(r.model_name for r in results))

    header = "| Method | " + " | ".join(schemes) + " |"
    sep = "|---|" + "---|" * len(schemes)
    lines = [header, sep]
    for m in models:
        cells = []
        for s in schemes:
            match = [r for r in results if r.model_name == m and r.scheme == s]
            cells.append(f"{match[0].accuracy:.1%}" if match else "—")
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
