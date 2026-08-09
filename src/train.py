"""Training entry point.

Trains on either source of data, with the same code path:

* ``--source physionet`` — the research dataset, for benchmarking and for
  developing the pipeline before hardware arrives.
* ``--source recordings`` — your own headband calibration sessions, which is
  what you will actually deploy.

The ``--device`` flag is what ties the two together. Passing ``--device crown``
restricts the PhysioNet data to the electrodes a Neurosity Crown physically
has, so the resulting accuracy is an honest *upper bound* on what that headband
can achieve, measured before spending any money.

    python -m src.train --source physionet --subjects 1 --device crown
    python -m src.train --source recordings --save models/me_crown.joblib
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from .config import (
    EXECUTED_RUNS,
    FIGURES_DIR,
    IMAGERY_RUNS,
    MODELS_DIR,
    MOTOR_CHANNELS_EXTENDED,
    RECORDINGS_DIR,
)
from .devices.channels import shared_channels
from .devices.registry import DEVICE_PROFILES, get_profile
from .evaluate import (
    evaluate_cross_subject,
    evaluate_subject_dependent,
    plot_accuracy_comparison,
    plot_confusion,
    plot_erd_topomap,
    results_table,
)
from .models import CLASSICAL_MODELS, MODEL_BUILDERS, build_model


# --------------------------------------------------------------------------
# Channel selection
# --------------------------------------------------------------------------


def resolve_channels(
    device_key: str | None,
    dataset_channels: list[str],
    *,
    motor_only: bool = True,
) -> list[str]:
    """Decide which electrodes to train on.

    With ``--device``, the answer is "the ones that device has", intersected
    with the dataset. Without it, the default is the sensorimotor strip rather
    than all 64 channels — measured on subject 1, restricting to those 21
    electrodes moves CSP+LDA from 0.60 to 0.73, because CSP fitted on 64
    channels with 45 trials is estimating a 64x64 covariance from far too
    little data and overfits.
    """
    if device_key:
        profile = get_profile(device_key)
        if not profile.default_channels:
            raise ValueError(
                f"{profile.display_name} has no fixed electrode layout. Train from "
                "your own recordings instead, or pass --channels explicitly."
            )
        picks = shared_channels(list(profile.default_channels), dataset_channels)
        if not picks:
            raise ValueError(
                f"{profile.display_name}'s electrodes {profile.default_channels} do not "
                f"overlap the dataset montage at all — this device cannot be simulated "
                "against PhysioNet."
            )
        return picks

    if motor_only:
        return [ch for ch in MOTOR_CHANNELS_EXTENDED if ch in dataset_channels]
    return list(dataset_channels)


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------


def load_physionet_xy(
    subjects: list[int],
    *,
    device_key: str | None,
    runs: tuple[int, ...],
    channels: list[str] | None = None,
    local_files: list[str] | None = None,
) -> tuple[dict[int, tuple[np.ndarray, np.ndarray]], list[str], object]:
    """Load and preprocess each subject. Returns per-subject (X, y) plus example epochs."""
    from .preprocess import (
        epoch_physionet,
        epochs_to_xy,
        load_physionet_subject,
        preprocess_raw,
    )

    per_subject: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    picks: list[str] | None = channels
    example_epochs = None

    for subject in subjects:
        raw = load_physionet_subject(
            subject, runs=runs, local_files=local_files if len(subjects) == 1 else None
        )
        if picks is None:
            picks = resolve_channels(device_key, raw.ch_names)
            print(f"  training on {len(picks)} channels: {', '.join(picks)}")

        clean = preprocess_raw(raw, pick_channels=picks)
        epochs = epoch_physionet(clean)
        x, y = epochs_to_xy(epochs)
        if len(np.unique(y)) < 2:
            print(f"  subject {subject}: only one class survived rejection, skipping")
            continue
        per_subject[subject] = (x, y)
        if example_epochs is None:
            # Keep a wider window for the ERD topomap, which needs pre-cue baseline.
            example_epochs = epoch_physionet(clean, tmin=-2.0, tmax=4.0)
        print(f"  subject {subject}: {x.shape[0]} trials x {x.shape[1]} ch x {x.shape[2]} samples")

    if not per_subject:
        raise RuntimeError("No usable data loaded.")
    return per_subject, picks or [], example_epochs


def load_recordings_xy(
    recordings_dir: Path,
    *,
    channels: list[str] | None = None,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], list[str]]:
    """Load every calibration session in a directory, grouped by subject."""
    from .acquire import load_session, session_to_xy

    files = sorted(Path(recordings_dir).glob("*.npz"))
    if not files:
        raise FileNotFoundError(
            f"No recordings in {recordings_dir}. Record one first:\n"
            "  python -m src.cli record --device <key>"
        )

    grouped: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    resolved: list[str] = channels or []
    for path in files:
        _, meta = load_session(path)
        x, y, ch_names = session_to_xy(path, pick_channels=channels)
        if not resolved:
            resolved = ch_names
        if x.shape[0] == 0:
            print(f"  {path.name}: all trials rejected as artefact, skipping")
            continue
        grouped.setdefault(meta.subject, []).append((x, y))
        print(f"  {path.name}: {x.shape[0]} trials x {x.shape[1]} ch ({meta.device_name})")

    per_subject = {
        subject: (
            np.concatenate([x for x, _ in parts]),
            np.concatenate([y for _, y in parts]),
        )
        for subject, parts in grouped.items()
    }
    if not per_subject:
        raise RuntimeError("Every recording was rejected — check electrode contact.")
    return per_subject, resolved


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def run_training(args: argparse.Namespace) -> int:
    channels = args.channels.split(",") if args.channels else None

    if args.source == "physionet":
        runs = EXECUTED_RUNS if args.task == "executed" else IMAGERY_RUNS
        local = args.local_edf.split(",") if args.local_edf else None
        print(f"Loading PhysioNet subjects {args.subjects} (runs {runs})...")
        per_subject, picks, example_epochs = load_physionet_xy(
            args.subjects,
            device_key=args.device,
            runs=runs,
            channels=channels,
            local_files=local,
        )
    else:
        print(f"Loading recordings from {args.recordings_dir}...")
        per_subject, picks = load_recordings_xy(Path(args.recordings_dir), channels=channels)
        example_epochs = None

    model_names = args.models or list(CLASSICAL_MODELS)
    if "eegnet" in model_names:
        try:
            import torch  # noqa: F401
        except ImportError:
            print("  torch not installed; dropping eegnet from the comparison")
            model_names = [m for m in model_names if m != "eegnet"]

    results = []
    pooled_x = np.concatenate([x for x, _ in per_subject.values()])
    pooled_y = np.concatenate([y for _, y in per_subject.values()])

    print(f"\nEvaluating {len(model_names)} models on {len(pooled_y)} pooled trials")
    print("-" * 78)
    for name in model_names:
        factory = lambda n=name: build_model(n)  # noqa: E731
        result = evaluate_subject_dependent(
            factory, pooled_x, pooled_y, model_name=name, n_splits=args.folds
        )
        results.append(result)
        print(result.summary())

    if args.cross_subject and len(per_subject) >= 2:
        print("\nCross-subject (leave-one-subject-out)")
        print("-" * 78)
        for name in model_names:
            factory = lambda n=name: build_model(n)  # noqa: E731
            result = evaluate_cross_subject(factory, per_subject, model_name=name)
            results.append(result)
            print(result.summary())
    elif args.cross_subject:
        print("\nSkipping cross-subject evaluation: needs >=2 subjects.")

    # -- figures -------------------------------------------------------
    if not args.no_figures:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        paths = [plot_accuracy_comparison(results)]
        best = max(results, key=lambda r: r.accuracy)
        paths.append(plot_confusion(best))
        if example_epochs is not None:
            try:
                paths.append(plot_erd_topomap(example_epochs))
            except Exception as exc:  # topomap needs valid sensor positions
                print(f"  ERD topomap skipped: {exc}")
        print("\nFigures written:")
        for p in paths:
            print(f"  {p}")

    print("\n" + results_table(results))

    # -- persistence ---------------------------------------------------
    if args.save:
        from .realtime import save_decoder

        best_name = max(
            (r for r in results if "within" in r.scheme), key=lambda r: r.accuracy
        ).model_name
        model = build_model(best_name)
        model.fit(pooled_x, pooled_y)
        path = save_decoder(
            model,
            picks,
            Path(args.save),
            model_name=best_name,
            source=args.source,
            device=args.device,
            n_trials=int(len(pooled_y)),
        )
        print(f"\nSaved best model ({best_name}) to {path}")
        print("Run it live with:")
        print(f"  python -m src.cli live --device <key> --model {path}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and evaluate motor-imagery decoders.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source", choices=("physionet", "recordings"), default="physionet",
        help="Research dataset, or your own headband calibration sessions.",
    )
    parser.add_argument(
        "--subjects", type=int, nargs="+", default=[1],
        help="PhysioNet subject numbers (1-109).",
    )
    parser.add_argument(
        "--task", choices=("imagery", "executed"), default="imagery",
        help="Imagined movement (runs 4/8/12) or executed movement (runs 3/7/11).",
    )
    parser.add_argument(
        "--device", choices=sorted(DEVICE_PROFILES), default=None,
        help="Restrict channels to this headset's montage, to simulate it.",
    )
    parser.add_argument(
        "--channels", default=None,
        help="Explicit comma-separated electrode list, overriding --device.",
    )
    parser.add_argument(
        "--models", nargs="+", choices=sorted(MODEL_BUILDERS), default=None,
        help="Models to compare.",
    )
    parser.add_argument("--folds", type=int, default=5, help="Within-subject CV folds.")
    parser.add_argument(
        "--cross-subject", action="store_true",
        help="Also run leave-one-subject-out evaluation.",
    )
    parser.add_argument(
        "--recordings-dir", default=str(RECORDINGS_DIR),
        help="Directory of .npz calibration sessions.",
    )
    parser.add_argument(
        "--local-edf", default=None,
        help="Comma-separated EDF paths, bypassing the MNE downloader (single subject).",
    )
    parser.add_argument(
        "--save", default=None,
        help=f"Path to persist the best model, e.g. {MODELS_DIR / 'decoder.joblib'}",
    )
    parser.add_argument("--no-figures", action="store_true", help="Skip figure generation.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run_training(args)


if __name__ == "__main__":
    sys.exit(main())
