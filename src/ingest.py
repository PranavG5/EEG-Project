"""Importing recordings made elsewhere — by your friends, or by other software.

A group project has a data problem the single-user case does not: recordings
arrive from several laptops, in whatever format the person's software produced,
with electrode names spelled however their vendor spells them, and with no
guarantee the cue labels survived at all. This module is the funnel. Everything
it accepts comes out the far side as the same ``.npz`` + ``.json`` pair that
:mod:`.acquire` writes, so :mod:`.train` cannot tell the difference.

Supported inputs:

* **Native session** — ``.npz`` + ``.json`` from ``cli record``. Just copied in.
* **Zip bundle** — what a friend gets from the web app's "export" button.
* **OpenBCI GUI CSV** — what you get from OpenBCI's own recording software.
* **BrainFlow CSV** — ``DataFilter.write_file`` output.
* **EDF/BDF** — the clinical/research interchange format, read via MNE.

For everything except the native format, the cue labels live outside the signal
file, so a **label sidecar** is required: a two-column CSV of ``onset_s,label``.
Without it the recording is unlabelled and cannot train anything — which is the
single most common thing to get wrong when someone records with third-party
software and hands you the file.
"""

from __future__ import annotations

import csv
import json
import shutil
import zipfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .acquire import SessionMetadata, load_session
from .config import CLASS_TO_INT, RECORDINGS_DIR
from .devices.channels import assess_montage, canonicalize_all


class IngestError(Exception):
    """Raised when a file cannot be imported, with a message aimed at the human."""


# --------------------------------------------------------------------------
# Label sidecars
# --------------------------------------------------------------------------


def read_label_sidecar(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a ``onset_s,label`` CSV of cue times.

    Accepts ``left``/``right`` (any case) or ``0``/``1``. Onsets are seconds
    from the start of the recording — not wall-clock time, because the two
    almost never agree once you account for the delay between starting the
    software and starting the stream.
    """
    onsets: list[float] = []
    labels: list[int] = []

    with open(path, newline="") as fh:
        for row_number, row in enumerate(csv.reader(fh), start=1):
            if not row or not row[0].strip():
                continue
            first = row[0].strip().lower()
            if first in {"onset_s", "onset", "time", "time_s"}:
                continue  # header
            if len(row) < 2:
                raise IngestError(
                    f"{path.name} line {row_number}: expected 'onset_s,label', got {row!r}"
                )
            try:
                onset = float(first)
            except ValueError:
                raise IngestError(
                    f"{path.name} line {row_number}: '{row[0]}' is not a number of seconds"
                ) from None

            raw_label = row[1].strip().lower()
            if raw_label in CLASS_TO_INT:
                label = CLASS_TO_INT[raw_label]
            elif raw_label in {"0", "1"}:
                label = int(raw_label)
            elif raw_label in {"l", "lh"}:
                label = CLASS_TO_INT["left"]
            elif raw_label in {"r", "rh"}:
                label = CLASS_TO_INT["right"]
            else:
                raise IngestError(
                    f"{path.name} line {row_number}: label '{row[1]}' is not "
                    f"one of {sorted(CLASS_TO_INT)} or 0/1"
                )
            onsets.append(onset)
            labels.append(label)

    if not onsets:
        raise IngestError(f"{path.name} contains no cue rows.")
    return np.asarray(onsets), np.asarray(labels, dtype=int)


def _find_sidecar(signal_path: Path, explicit: Path | None) -> Path:
    """Locate the label CSV for a signal file, or explain what is missing."""
    if explicit is not None:
        if not explicit.exists():
            raise IngestError(f"Label file {explicit} does not exist.")
        return explicit

    for candidate in (
        signal_path.with_suffix(".labels.csv"),
        signal_path.with_name(signal_path.stem + "_labels.csv"),
        signal_path.with_name("labels.csv"),
    ):
        if candidate.exists():
            return candidate

    raise IngestError(
        f"No cue labels found for {signal_path.name}. A signal file alone cannot "
        "train anything — it has no idea which trials were left and which right.\n"
        f"Provide a CSV of 'onset_s,label' rows, named {signal_path.stem}.labels.csv "
        "or passed explicitly."
    )


# --------------------------------------------------------------------------
# Format readers
# --------------------------------------------------------------------------


def read_openbci_csv(path: Path) -> tuple[np.ndarray, float, list[str]]:
    """Read a recording saved by the OpenBCI GUI.

    The GUI writes '%'-prefixed header lines carrying the sample rate, then a
    column header row, then samples. EXG channel columns hold microvolts; we
    convert to volts at the boundary as everywhere else.
    """
    sfreq: float | None = None
    header: list[str] | None = None
    rows: list[list[float]] = []

    with open(path, newline="") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith("%"):
                if "sample rate" in line.lower():
                    for token in line.replace("=", " ").split():
                        try:
                            sfreq = float(token.rstrip("Hz"))
                            break
                        except ValueError:
                            continue
                continue
            parts = [p.strip() for p in line.split(",")]
            if header is None and any(c.isalpha() for c in line):
                header = parts
                continue
            try:
                rows.append([float(p) if p else np.nan for p in parts])
            except ValueError:
                continue  # trailing junk line

    if not rows or header is None:
        raise IngestError(f"{path.name} does not look like an OpenBCI GUI CSV.")
    if sfreq is None:
        raise IngestError(
            f"{path.name} has no '%Sample Rate' header line; cannot determine the "
            "sampling rate, and guessing it would silently corrupt every filter."
        )

    exg_cols = [i for i, name in enumerate(header) if "exg channel" in name.lower()]
    if not exg_cols:
        raise IngestError(f"{path.name} has no 'EXG Channel' columns.")

    table = np.asarray(rows, dtype=float)
    data = table[:, exg_cols].T * 1e-6
    names = [f"EXG{i}" for i in range(len(exg_cols))]
    return data, sfreq, names


def read_brainflow_csv(path: Path, sfreq: float, eeg_rows: list[int]) -> tuple[np.ndarray, float, list[str]]:
    """Read ``DataFilter.write_file`` output (tab-separated, one row per sample).

    BrainFlow's CSV carries no header at all, so the caller must say which
    columns are EEG and what the rate was. That is genuinely unknowable from
    the file, which is why the native format stores metadata alongside.
    """
    table = np.loadtxt(path, delimiter="\t")
    if table.ndim != 2:
        raise IngestError(f"{path.name} is not a 2-D table.")
    try:
        data = table[:, eeg_rows].T * 1e-6
    except IndexError:
        raise IngestError(
            f"{path.name} has {table.shape[1]} columns; requested EEG columns {eeg_rows}."
        ) from None
    return data, sfreq, [f"EXG{i}" for i in range(len(eeg_rows))]


def read_edf(path: Path) -> tuple[np.ndarray, float, list[str]]:
    """Read an EDF/BDF via MNE, keeping only recognisable scalp electrodes."""
    import mne

    raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    mapping = canonicalize_all(raw.ch_names)
    if not mapping:
        raise IngestError(
            f"{path.name}: none of {raw.ch_names} are recognisable 10-05 electrode names."
        )
    raw.pick(list(mapping))
    return raw.get_data(), float(raw.info["sfreq"]), list(mapping.values())


# --------------------------------------------------------------------------
# Import
# --------------------------------------------------------------------------


def import_signal_file(
    path: Path,
    *,
    subject: str,
    channels: list[str],
    labels_path: Path | None = None,
    device_key: str = "imported",
    task: str = "imagery",
    sfreq: float | None = None,
    eeg_columns: list[int] | None = None,
    out_dir: Path = RECORDINGS_DIR,
    notes: str = "",
) -> Path:
    """Convert a third-party recording into a native session.

    ``channels`` is required and is the thing people get wrong: the file says
    "EXG Channel 0", and only the person who placed the electrodes knows that
    it was C3. Passing the wrong order here silently trains a model on permuted
    electrodes, which still produces a plausible accuracy number.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        if eeg_columns is not None:
            if sfreq is None:
                raise IngestError("BrainFlow CSV import needs --sfreq.")
            data, sfreq_out, _ = read_brainflow_csv(path, sfreq, eeg_columns)
        else:
            data, sfreq_out, _ = read_openbci_csv(path)
    elif suffix in {".edf", ".bdf"}:
        data, sfreq_out, _ = read_edf(path)
    else:
        raise IngestError(
            f"Don't know how to read {path.name}. Supported: .csv (OpenBCI GUI or "
            "BrainFlow), .edf, .bdf, or a native .npz/.json pair."
        )

    if sfreq is not None and suffix == ".csv" and eeg_columns is None:
        sfreq_out = sfreq  # explicit override wins over the header

    if len(channels) != data.shape[0]:
        raise IngestError(
            f"{path.name} has {data.shape[0]} EEG channels but you named "
            f"{len(channels)}: {channels}. These must correspond one-to-one, in "
            "the order the file stores them."
        )

    mapping = canonicalize_all(channels)
    unknown = [c for c in channels if c not in mapping]
    if unknown:
        raise IngestError(
            f"Not valid 10-05 electrode names: {unknown}. Use names like C3, C4, "
            "CP3, FC4 — the position on the head, not the amplifier's channel number."
        )
    canonical = [mapping[c] for c in channels]

    onsets, label_ints = read_label_sidecar(_find_sidecar(path, labels_path))
    duration_s = data.shape[1] / sfreq_out
    late = onsets[onsets > duration_s]
    if late.size:
        raise IngestError(
            f"{len(late)} cue onsets fall past the end of the {duration_s:.1f}s "
            "recording. The label file and the signal file are probably not from "
            "the same session."
        )

    meta = SessionMetadata(
        subject=subject,
        device_key=device_key,
        device_name=f"imported from {path.name}",
        sfreq=float(sfreq_out),
        ch_names=canonical,
        task=task,
        started_utc=datetime.now(timezone.utc).isoformat(),
        protocol={"imported": True, "source_file": path.name},
        trials=[
            {"onset_s": float(o), "label": _label_name(v), "label_int": int(v)}
            for o, v in zip(onsets, label_ints)
        ],
        notes=notes,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = meta.started_utc.replace(":", "").replace("-", "").split(".")[0]
    stem = f"{subject}_{device_key}_{task}_{stamp}"
    np.savez_compressed(out_dir / f"{stem}.npz", data=data)
    (out_dir / f"{stem}.json").write_text(json.dumps(asdict(meta), indent=2))
    return out_dir / f"{stem}.npz"


def _label_name(value: int) -> str:
    for name, idx in CLASS_TO_INT.items():
        if idx == value:
            return name
    raise IngestError(f"Label {value} is not a known class.")


def import_native(path: Path, out_dir: Path = RECORDINGS_DIR) -> Path:
    """Copy in an ``.npz`` + ``.json`` pair produced by this project elsewhere."""
    path = Path(path)
    sidecar = path.with_suffix(".json")
    if not sidecar.exists():
        raise IngestError(
            f"{path.name} has no matching {sidecar.name}. Both files are needed — "
            "the .npz holds the samples and the .json holds the cue times and "
            "electrode names, and neither is useful alone."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    for src in (path, sidecar):
        dest = out_dir / src.name
        if dest.resolve() != src.resolve():
            shutil.copy2(src, dest)
    return out_dir / path.name


def import_zip(path: Path, out_dir: Path = RECORDINGS_DIR) -> list[Path]:
    """Unpack a zip of native sessions — the format the web app exports.

    Members are extracted by basename only, so a zip built on someone else's
    machine cannot write outside the recordings directory.
    """
    imported: list[Path] = []
    out_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(path) as zf:
        members = [m for m in zf.namelist() if m.lower().endswith((".npz", ".json"))]
        if not members:
            raise IngestError(f"{Path(path).name} contains no .npz/.json session files.")
        for member in members:
            name = Path(member).name
            if not name:
                continue
            with zf.open(member) as src, open(out_dir / name, "wb") as dst:
                shutil.copyfileobj(src, dst)
            if name.lower().endswith(".npz"):
                imported.append(out_dir / name)

    for npz in imported:
        if not npz.with_suffix(".json").exists():
            raise IngestError(f"{npz.name} arrived without its .json sidecar.")
    return imported


#: File types that carry their own metadata and can be imported with no extra
#: information from the user.
SELF_DESCRIBING_SUFFIXES = {".zip", ".npz", ".json"}


def import_any(path: Path, out_dir: Path = RECORDINGS_DIR, **kwargs: object) -> list[Path]:
    """Dispatch on file type. The entry point the web upload and CLI both use."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".zip":
        return import_zip(path, out_dir)
    if suffix == ".npz":
        return [import_native(path, out_dir)]
    if suffix == ".json":
        return [import_native(path.with_suffix(".npz"), out_dir)]

    # Third-party formats know nothing about electrode positions or whose head
    # this was, so refuse with an explanation rather than letting a TypeError
    # about missing keyword arguments reach the user.
    missing = [name for name in ("subject", "channels") if not kwargs.get(name)]
    if missing:
        raise IngestError(
            f"{path.name} is a raw signal file, so it cannot say which electrode "
            f"each column came from or whose head it was. Missing: {', '.join(missing)}.\n"
            "Supply them — in the app's upload form, or on the command line:\n"
            f"  python -m src.cli import {path.name} --subject NAME "
            "--channels C3,C4,CP3,CP4"
        )
    return [import_signal_file(path, out_dir=out_dir, **kwargs)]  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# Inventory
# --------------------------------------------------------------------------


def describe_recording(npz_path: Path) -> dict:
    """Summarise one session for display, without loading the full signal.

    Includes the montage verdict, because a group inevitably ends up with
    recordings from more than one electrode layout and it needs to be obvious
    which of them can contribute to a left-vs-right model.
    """
    npz_path = Path(npz_path)
    _, meta = load_session(npz_path)
    labels = [t["label"] for t in meta.trials]
    montage = assess_montage(meta.ch_names)

    with np.load(npz_path) as bundle:
        n_samples = bundle["data"].shape[1]

    return {
        "file": npz_path.name,
        "subject": meta.subject,
        "device": meta.device_key,
        "device_name": meta.device_name,
        "task": meta.task,
        "recorded": meta.started_utc,
        "sfreq": meta.sfreq,
        "channels": meta.ch_names,
        "n_trials": len(meta.trials),
        "n_left": labels.count("left"),
        "n_right": labels.count("right"),
        "duration_s": round(n_samples / meta.sfreq, 1) if meta.sfreq else 0.0,
        "montage_rating": montage.rating,
        "montage_note": montage.rationale,
    }


def inventory(recordings_dir: Path = RECORDINGS_DIR) -> list[dict]:
    """Every session on disk, newest first, with per-person grouping in mind."""
    recordings_dir = Path(recordings_dir)
    if not recordings_dir.exists():
        return []
    out = []
    for npz in sorted(recordings_dir.glob("*.npz")):
        try:
            out.append(describe_recording(npz))
        except Exception as exc:
            out.append({"file": npz.name, "error": str(exc)})
    return sorted(out, key=lambda d: d.get("recorded", ""), reverse=True)


def export_bundle(recordings_dir: Path, out_path: Path, subject: str | None = None) -> Path:
    """Zip up sessions so one person can send theirs to whoever trains the model."""
    recordings_dir = Path(recordings_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for npz in sorted(recordings_dir.glob("*.npz")):
            sidecar = npz.with_suffix(".json")
            if not sidecar.exists():
                continue
            if subject is not None:
                meta = json.loads(sidecar.read_text())
                if meta.get("subject") != subject:
                    continue
            zf.write(npz, npz.name)
            zf.write(sidecar, sidecar.name)
    return out_path
