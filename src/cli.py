"""Unified command-line entry point.

    python -m src.cli devices                  # which headset should I buy?
    python -m src.cli check   --device crown   # is the one I bought working?
    python -m src.cli record  --device crown   # collect calibration data
    python -m src.cli train   --source recordings --save models/me.joblib
    python -m src.cli live    --device crown --model models/me.joblib

The ordering above is the intended workflow, and each step gates the next.
Skipping ``check`` is the most common way to waste an afternoon: a calibration
session recorded through a badly seated electrode produces data that no
classifier can rescue, and you will not find out until training reports 51%.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from .config import MODELS_DIR, RECORDINGS_DIR
from .devices.registry import DEVICE_PROFILES, describe_all, describe_device, get_profile


# --------------------------------------------------------------------------
# devices
# --------------------------------------------------------------------------


def cmd_devices(args: argparse.Namespace) -> int:
    if args.device:
        print(describe_device(args.device))
    else:
        print(describe_all())
        print(
            "\nRatings describe whether the electrode layout can physically observe\n"
            "left-vs-right hand motor imagery, not overall device quality."
        )
    return 0


# --------------------------------------------------------------------------
# check
# --------------------------------------------------------------------------


def cmd_check(args: argparse.Namespace) -> int:
    """Signal-quality check: connect, stream briefly, report per-channel health.

    What the numbers mean:

    * **RMS amplitude** — resting EEG in the 8-30 Hz band is roughly 2-15 uV.
      Below ~0.5 uV the channel is probably not connected to skin (flat line).
      Above ~50 uV it is picking up movement, muscle, or a floating electrode.
    * **60 Hz ratio** — powerline pickup relative to the EEG band. A well
      seated electrode sits under ~2; a poorly contacting one acts as an
      antenna and this climbs sharply. This is the most sensitive single
      indicator of contact quality available without an impedance check.
    """
    from scipy.signal import welch

    from .devices import open_source

    profile = get_profile(args.device)
    print(describe_device(args.device))
    print("\nConnecting...")

    source = open_source(
        args.device,
        serial_port=args.serial_port,
        mac_address=args.mac_address,
        ip_address=args.ip_address,
        channel_override=args.channels.split(",") if args.channels else None,
    )

    import time

    chunks = []
    with source:
        info = source.info
        print(f"Streaming {info.sfreq:g} Hz from {info.name} for {args.seconds:g}s")
        print("Sit still, eyes open, relaxed jaw.\n")
        deadline = time.monotonic() + args.seconds
        while time.monotonic() < deadline:
            time.sleep(0.2)
            chunk = source.poll()
            if chunk.size:
                chunks.append(chunk)

    if not chunks:
        print("No data received. Check pairing, dongle, and that the device is on.")
        return 1

    data = np.concatenate(chunks, axis=1)
    sfreq = source.info.sfreq
    print(f"Collected {data.shape[1]} samples ({data.shape[1] / sfreq:.1f}s)\n")

    nperseg = min(data.shape[1], int(sfreq * 2))
    freqs, psd = welch(data, fs=sfreq, nperseg=nperseg, axis=-1)
    band = (freqs >= 8) & (freqs <= 30)
    line = (freqs >= 58) & (freqs <= 62)

    print(f"{'channel':>8s} {'RMS (uV)':>10s} {'60Hz ratio':>12s}   status")
    print("-" * 52)
    all_ok = True
    for i, ch in enumerate(source.info.ch_names):
        rms = float(np.sqrt(psd[i, band].sum() * (freqs[1] - freqs[0]))) * 1e6
        line_power = float(psd[i, line].mean()) if line.any() else 0.0
        band_power = float(psd[i, band].mean()) + 1e-30
        ratio = line_power / band_power

        if rms < 0.5:
            status, ok = "FLAT — not contacting skin", False
        elif rms > 50:
            status, ok = "NOISY — movement or floating electrode", False
        elif ratio > 2.0:
            status, ok = "LINE NOISE — reseat, add gel/saline", False
        else:
            status, ok = "ok", True
        all_ok &= ok
        print(f"{ch:>8s} {rms:>10.2f} {ratio:>12.2f}   {status}")

    print()
    if all_ok:
        print("All channels look usable. Next: python -m src.cli record --device "
              f"{args.device}")
    else:
        print("Fix the flagged channels before recording — no classifier recovers\n"
              "from an electrode that was not touching skin.")

    if profile.montage_rating == "unsuitable":
        print(
            f"\nNote: {profile.display_name} has no sensorimotor electrodes, so even "
            "perfect signal quality will not yield left-vs-right hand decoding."
        )
    return 0 if all_ok else 1


# --------------------------------------------------------------------------
# record
# --------------------------------------------------------------------------


def cmd_record(args: argparse.Namespace) -> int:
    from .acquire import CalibrationRecorder, CueProtocol, save_session
    from .devices import open_source

    profile = get_profile(args.device)
    if profile.montage_rating == "unsuitable" and not args.force:
        print(describe_device(args.device))
        print(
            "\nRefusing to record: this device cannot observe the signal you are "
            "trying to decode.\nPass --force to record anyway (e.g. to demonstrate "
            "that it does not work)."
        )
        return 2

    protocol = CueProtocol(
        n_trials_per_class=args.trials_per_class,
        imagery_s=args.imagery_seconds,
        seed=args.seed,
    )
    minutes = protocol.estimated_duration_s() / 60
    print(f"Session: {protocol.n_trials} trials, about {minutes:.1f} minutes.")
    print(
        "\nWhen the arrow points LEFT, imagine squeezing your left fist — feel the\n"
        "movement kinaesthetically, do not just picture a hand. Keep still, keep\n"
        "your jaw loose, and try not to blink during the imagery window.\n"
    )
    if args.task == "executed":
        print("TASK = EXECUTED: actually squeeze the fist. Do this first; it produces\n"
              "a much stronger signal and proves the whole chain works.\n")
    input("Press Enter to begin...")

    source = open_source(
        args.device,
        serial_port=args.serial_port,
        mac_address=args.mac_address,
        ip_address=args.ip_address,
        channel_override=args.channels.split(",") if args.channels else None,
    )

    with source:
        recorder = CalibrationRecorder(
            source,
            protocol,
            subject=args.subject,
            device_key=args.device,
            task=args.task,
        )
        data, meta = recorder.run()

    print()
    path = save_session(data, meta, out_dir=Path(args.out_dir))
    duration = data.shape[1] / meta.sfreq if meta.sfreq else 0
    print(f"Saved {data.shape[0]} channels x {data.shape[1]} samples ({duration:.0f}s)")
    print(f"  {path}")
    print("\nNext: python -m src.cli train --source recordings --save "
          f"{MODELS_DIR / 'me.joblib'}")
    return 0


# --------------------------------------------------------------------------
# live
# --------------------------------------------------------------------------


def cmd_live(args: argparse.Namespace) -> int:
    from .devices import open_source
    from .realtime import LiveDecoder, load_decoder

    model, ch_names, metadata = load_decoder(Path(args.model))
    print(f"Loaded {metadata.get('model_name', 'model')} trained on "
          f"{metadata.get('n_trials', '?')} trials")
    print(f"Channels: {', '.join(ch_names)}")

    source = open_source(
        args.device,
        serial_port=args.serial_port,
        mac_address=args.mac_address,
        ip_address=args.ip_address,
        channel_override=args.channels.split(",") if args.channels else None,
    )

    decoder = LiveDecoder(
        source, model, ch_names,
        confidence_threshold=args.threshold,
        smoothing=args.smoothing,
    )
    print("\nDecoding. Imagine left or right hand movement. Ctrl-C to stop.\n")
    predictions = decoder.run(duration_s=args.seconds)

    if predictions:
        decided = [p for p in predictions if p.label != "uncertain"]
        print(f"\n{len(predictions)} windows, {len(decided)} above threshold")
        for cls in ("left", "right"):
            n = sum(1 for p in decided if p.label == cls)
            print(f"  {cls:>5s}: {n}")
    return 0


# --------------------------------------------------------------------------
# app
# --------------------------------------------------------------------------


def cmd_app(args: argparse.Namespace) -> int:
    from .app import serve

    serve(args.host, args.port, open_browser=not args.no_browser)
    return 0


# --------------------------------------------------------------------------
# import / export — collecting data from several people
# --------------------------------------------------------------------------


def cmd_import(args: argparse.Namespace) -> int:
    from .ingest import IngestError, import_any

    channels = [c.strip() for c in args.channels.split(",")] if args.channels else None
    imported, failed = [], []

    for pattern in args.files:
        matches = sorted(Path().glob(pattern)) if any(c in pattern for c in "*?[") else [Path(pattern)]
        if not matches:
            failed.append(f"{pattern}: no such file")
            continue
        for path in matches:
            try:
                kwargs: dict = {}
                if path.suffix.lower() not in {".npz", ".json", ".zip"}:
                    if not channels:
                        raise IngestError(
                            "This file type carries no electrode names, so --channels is "
                            "required: the positions you actually recorded from, in the "
                            "order the file stores them (e.g. --channels C3,C4,CP3,CP4)."
                        )
                    kwargs = {
                        "subject": args.subject,
                        "channels": channels,
                        "labels_path": Path(args.labels) if args.labels else None,
                        "device_key": args.device,
                        "task": args.task,
                        "sfreq": args.sfreq,
                        "eeg_columns": (
                            [int(c) for c in args.eeg_columns.split(",")]
                            if args.eeg_columns else None
                        ),
                    }
                results = import_any(path, out_dir=Path(args.out_dir), **kwargs)
                imported.extend(results)
                for r in results:
                    print(f"  imported {r.name}")
            except Exception as exc:
                failed.append(f"{path.name}: {exc}")

    print(f"\n{len(imported)} session(s) imported into {args.out_dir}")
    for message in failed:
        print(f"  FAILED  {message}")
    if imported:
        print("\nNext: python -m src.cli train --source recordings --cross-subject")
    return 0 if imported and not failed else (1 if failed else 0)


def cmd_export(args: argparse.Namespace) -> int:
    from .ingest import export_bundle

    path = export_bundle(Path(args.recordings_dir), Path(args.out), subject=args.subject)
    size_kb = path.stat().st_size / 1024
    print(f"Wrote {path} ({size_kb:.0f} KB)")
    print("Send this to whoever is training the model; they run:")
    print(f"  python -m src.cli import {path.name}")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    from .ingest import inventory

    sessions = inventory(Path(args.recordings_dir))
    if not sessions:
        print(f"No recordings in {args.recordings_dir}.")
        return 0

    print(f"{'person':>10s} {'task':>9s} {'trials':>7s} {'montage':>11s}  electrodes")
    print("-" * 78)
    for s in sessions:
        if "error" in s:
            print(f"{s['file']}: {s['error']}")
            continue
        print(
            f"{s['subject']:>10s} {s['task']:>9s} "
            f"{s['n_trials']:>3d} ({s['n_left']}L/{s['n_right']}R) "
            f"{s['montage_rating']:>11s}  {', '.join(s['channels'])}"
        )
    people = sorted({s.get("subject") for s in sessions if "error" not in s})
    total = sum(s.get("n_trials", 0) for s in sessions if "error" not in s)
    print(f"\n{len(people)} people, {total} trials: {', '.join(str(p) for p in people)}")
    return 0


# --------------------------------------------------------------------------
# train (delegates)
# --------------------------------------------------------------------------


def cmd_train(argv: list[str]) -> int:
    """Hand the remaining arguments straight to the training parser.

    Dispatched before the top-level parser runs rather than through a
    subparser: ``argparse.REMAINDER`` does not capture arguments that begin
    with ``-``, so ``cli train --source recordings`` would otherwise be
    rejected as an unrecognised option.
    """
    from .train import main as train_main

    return train_main(argv)


# --------------------------------------------------------------------------
# parser
# --------------------------------------------------------------------------


def _add_connection_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--device", choices=sorted(DEVICE_PROFILES), required=True)
    p.add_argument("--serial-port", default="", help="e.g. /dev/ttyUSB0, COM3 (OpenBCI)")
    p.add_argument("--mac-address", default="", help="Bluetooth MAC (Muse on some platforms)")
    p.add_argument("--ip-address", default="", help="Device IP (Neurosity over Wi-Fi)")
    p.add_argument(
        "--channels", default=None,
        help="Comma-separated 10-05 names for where you ACTUALLY placed the "
             "electrodes. Required for OpenBCI if you moved them.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.cli",
        description="EEG motor-imagery decoding, from dataset to live headband.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_dev = sub.add_parser("devices", help="Compare headsets for this task")
    p_dev.add_argument("--device", choices=sorted(DEVICE_PROFILES), default=None)
    p_dev.set_defaults(func=cmd_devices)

    p_check = sub.add_parser("check", help="Signal-quality check on connected hardware")
    _add_connection_args(p_check)
    p_check.add_argument("--seconds", type=float, default=20.0)
    p_check.set_defaults(func=cmd_check)

    p_rec = sub.add_parser("record", help="Run a cued calibration session")
    _add_connection_args(p_rec)
    p_rec.add_argument("--subject", default="self")
    p_rec.add_argument("--task", choices=("imagery", "executed"), default="imagery")
    p_rec.add_argument("--trials-per-class", type=int, default=30)
    p_rec.add_argument("--imagery-seconds", type=float, default=4.0)
    p_rec.add_argument("--seed", type=int, default=0)
    p_rec.add_argument("--out-dir", default=str(RECORDINGS_DIR))
    p_rec.add_argument("--force", action="store_true",
                       help="Record even on a device with no motor coverage.")
    p_rec.set_defaults(func=cmd_record)

    p_live = sub.add_parser("live", help="Decode continuously from the headband")
    _add_connection_args(p_live)
    p_live.add_argument("--model", required=True, help="Path from `train --save`")
    p_live.add_argument("--threshold", type=float, default=0.65)
    p_live.add_argument("--smoothing", type=int, default=4)
    p_live.add_argument("--seconds", type=float, default=None)
    p_live.set_defaults(func=cmd_live)

    p_app = sub.add_parser("app", help="Open the local web app in a browser")
    p_app.add_argument("--host", default="127.0.0.1",
                       help="0.0.0.0 exposes it to your network — see the docs first")
    p_app.add_argument("--port", type=int, default=8000)
    p_app.add_argument("--no-browser", action="store_true")
    p_app.set_defaults(func=cmd_app)

    p_imp = sub.add_parser("import", help="Import recordings from friends or other software")
    p_imp.add_argument("files", nargs="+", help=".npz/.json pair, .zip, .csv, .edf (globs ok)")
    p_imp.add_argument("--subject", default="friend", help="Whose head the data came from")
    p_imp.add_argument("--channels", default=None,
                       help="Electrode positions in file order, e.g. C3,C4,CP3,CP4")
    p_imp.add_argument("--labels", default=None, help="Cue times CSV: onset_s,label")
    p_imp.add_argument("--device", default="imported")
    p_imp.add_argument("--task", choices=("imagery", "executed"), default="imagery")
    p_imp.add_argument("--sfreq", type=float, default=None, help="Required for BrainFlow CSV")
    p_imp.add_argument("--eeg-columns", default=None,
                       help="Comma-separated column indices (BrainFlow CSV only)")
    p_imp.add_argument("--out-dir", default=str(RECORDINGS_DIR))
    p_imp.set_defaults(func=cmd_import)

    p_exp = sub.add_parser("export", help="Zip your recordings to send to the group")
    p_exp.add_argument("--out", default="my_recordings.zip")
    p_exp.add_argument("--subject", default=None, help="Only this person's sessions")
    p_exp.add_argument("--recordings-dir", default=str(RECORDINGS_DIR))
    p_exp.set_defaults(func=cmd_export)

    p_ls = sub.add_parser("list", help="Show what has been collected so far")
    p_ls.add_argument("--recordings-dir", default=str(RECORDINGS_DIR))
    p_ls.set_defaults(func=cmd_list)

    # `train` is documented here for discoverability but intercepted in main().
    sub.add_parser(
        "train", add_help=False,
        help="Train/evaluate (all options: `python -m src.train --help`)",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "train":
        return cmd_train(argv[1:])
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
