"""Tests for importing recordings from other people and other software.

The failure mode this guards against is not a crash. It is a file that imports
cleanly with the electrode names in the wrong order, or with cue times that
belong to a different session — both of which train a model that reports a
believable accuracy and means nothing.
"""

from __future__ import annotations

import json
import zipfile

import numpy as np
import pytest

from src.acquire import SessionMetadata, save_session
from src.ingest import (
    IngestError,
    describe_recording,
    export_bundle,
    import_any,
    import_signal_file,
    import_zip,
    inventory,
    read_label_sidecar,
)


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


def _write_openbci_csv(path, n_channels=4, sfreq=250, seconds=40, sample_rate_header=True):
    rng = np.random.default_rng(0)
    n = int(sfreq * seconds)
    data = rng.normal(0, 20, (n_channels, n))
    with open(path, "w") as fh:
        fh.write("%OpenBCI Raw EEG Data\n")
        fh.write(f"%Number of channels = {n_channels}\n")
        if sample_rate_header:
            fh.write(f"%Sample Rate = {sfreq} Hz\n")
        header = ["Sample Index"] + [f" EXG Channel {i}" for i in range(n_channels)]
        fh.write(", ".join(header) + ", Timestamp\n")
        for i in range(n):
            fh.write(f"{i % 256}, " + ", ".join(f"{v:.4f}" for v in data[:, i]) + f", {i / sfreq:.4f}\n")
    return path


def _write_labels(path, onsets, labels):
    with open(path, "w") as fh:
        fh.write("onset_s,label\n")
        for onset, label in zip(onsets, labels):
            fh.write(f"{onset},{label}\n")
    return path


def _native_session(directory, subject="alice", n_trials=6, sfreq=250.0):
    rng = np.random.default_rng(1)
    data = rng.normal(0, 20e-6, (4, int(sfreq * 60)))
    meta = SessionMetadata(
        subject=subject,
        device_key="synthetic",
        device_name="test",
        sfreq=sfreq,
        ch_names=["C3", "C4", "CP3", "CP4"],
        task="imagery",
        started_utc="2026-01-01T00:00:00+00:00",
        protocol={},
        trials=[
            {"onset_s": 5.0 + 6 * i, "label": ["left", "right"][i % 2], "label_int": i % 2}
            for i in range(n_trials)
        ],
    )
    return save_session(data, meta, out_dir=directory)


# --------------------------------------------------------------------------
# Label sidecars
# --------------------------------------------------------------------------


def test_label_sidecar_accepts_names_and_integers(tmp_path):
    path = tmp_path / "l.csv"
    path.write_text("onset_s,label\n1.0,left\n2.0,RIGHT\n3.0,0\n4.0,1\n5.0,L\n6.0,r\n")
    onsets, labels = read_label_sidecar(path)
    np.testing.assert_allclose(onsets, [1, 2, 3, 4, 5, 6])
    np.testing.assert_array_equal(labels, [0, 1, 0, 1, 0, 1])


def test_label_sidecar_rejects_unknown_label(tmp_path):
    path = tmp_path / "l.csv"
    path.write_text("onset_s,label\n1.0,foot\n")
    with pytest.raises(IngestError, match="foot"):
        read_label_sidecar(path)


def test_label_sidecar_rejects_non_numeric_onset(tmp_path):
    path = tmp_path / "l.csv"
    path.write_text("onset_s,label\nwhenever,left\n")
    with pytest.raises(IngestError, match="not a number"):
        read_label_sidecar(path)


def test_empty_sidecar_is_an_error(tmp_path):
    path = tmp_path / "l.csv"
    path.write_text("onset_s,label\n")
    with pytest.raises(IngestError, match="no cue rows"):
        read_label_sidecar(path)


# --------------------------------------------------------------------------
# Third-party signal files
# --------------------------------------------------------------------------


def test_openbci_csv_roundtrip(tmp_path):
    csv = _write_openbci_csv(tmp_path / "rec.csv")
    _write_labels(tmp_path / "rec.labels.csv", [5, 11, 17, 23], ["left", "right", "left", "right"])

    out = import_signal_file(
        csv, subject="carol", channels=["C3", "C4", "CP3", "CP4"], out_dir=tmp_path / "out"
    )
    described = describe_recording(out)
    assert described["subject"] == "carol"
    assert described["n_trials"] == 4
    assert described["channels"] == ["C3", "C4", "CP3", "CP4"]
    assert described["montage_rating"] == "good"


def test_missing_sidecar_explains_itself(tmp_path):
    csv = _write_openbci_csv(tmp_path / "rec.csv")
    with pytest.raises(IngestError, match="No cue labels found"):
        import_signal_file(csv, subject="x", channels=["C3", "C4", "CP3", "CP4"],
                           out_dir=tmp_path / "out")


def test_channel_count_mismatch_is_rejected(tmp_path):
    """Naming 2 electrodes for a 4-column file must fail, not silently truncate."""
    csv = _write_openbci_csv(tmp_path / "rec.csv", n_channels=4)
    _write_labels(tmp_path / "rec.labels.csv", [5, 11], ["left", "right"])
    with pytest.raises(IngestError, match="4 EEG channels but you named 2"):
        import_signal_file(csv, subject="x", channels=["C3", "C4"], out_dir=tmp_path / "out")


def test_amplifier_channel_numbers_are_rejected_as_names(tmp_path):
    """'Channel 1' is not a place on the head; refuse rather than guess."""
    csv = _write_openbci_csv(tmp_path / "rec.csv", n_channels=2)
    _write_labels(tmp_path / "rec.labels.csv", [5, 11], ["left", "right"])
    with pytest.raises(IngestError, match="Not valid 10-05 electrode names"):
        import_signal_file(csv, subject="x", channels=["1", "2"], out_dir=tmp_path / "out")


def test_cues_past_end_of_recording_are_rejected(tmp_path):
    """Catches a label file paired with the wrong signal file."""
    csv = _write_openbci_csv(tmp_path / "rec.csv", seconds=30)
    _write_labels(tmp_path / "rec.labels.csv", [5, 900], ["left", "right"])
    with pytest.raises(IngestError, match="past the end"):
        import_signal_file(csv, subject="x", channels=["C3", "C4", "CP3", "CP4"],
                           out_dir=tmp_path / "out")


def test_missing_sample_rate_header_is_rejected(tmp_path):
    """Guessing the sampling rate would silently corrupt every filter."""
    csv = _write_openbci_csv(tmp_path / "rec.csv", sample_rate_header=False)
    _write_labels(tmp_path / "rec.labels.csv", [5, 11], ["left", "right"])
    with pytest.raises(IngestError, match="sampling rate"):
        import_signal_file(csv, subject="x", channels=["C3", "C4", "CP3", "CP4"],
                           out_dir=tmp_path / "out")


def test_import_any_refuses_raw_file_without_metadata(tmp_path):
    csv = _write_openbci_csv(tmp_path / "rec.csv")
    with pytest.raises(IngestError, match="Missing: subject, channels"):
        import_any(csv, out_dir=tmp_path / "out")


# --------------------------------------------------------------------------
# Native sessions and bundles
# --------------------------------------------------------------------------


def test_native_npz_without_json_is_rejected(tmp_path):
    (tmp_path / "orphan.npz").write_bytes(b"not really an npz")
    with pytest.raises(IngestError, match="no matching"):
        import_any(tmp_path / "orphan.npz", out_dir=tmp_path / "out")


def test_export_then_import_roundtrip(tmp_path):
    """The path a friend actually follows: export a zip, someone else imports it."""
    theirs = tmp_path / "theirs"
    _native_session(theirs, subject="alice")
    _native_session(theirs, subject="bob")

    bundle = export_bundle(theirs, tmp_path / "from_alice.zip", subject="alice")
    with zipfile.ZipFile(bundle) as zf:
        assert len([n for n in zf.namelist() if n.endswith(".npz")]) == 1

    ours = tmp_path / "ours"
    imported = import_zip(bundle, out_dir=ours)
    assert len(imported) == 1
    assert describe_recording(imported[0])["subject"] == "alice"


def test_zip_cannot_write_outside_the_recordings_directory(tmp_path):
    """A zip from someone else's machine must not be able to path-traverse.

    Members are extracted by basename, so '../../escaped.npz' lands inside the
    target directory as 'escaped.npz' rather than two levels up. The import
    then succeeds — being safe is not the same as being an error.
    """
    evil = tmp_path / "evil.zip"
    with zipfile.ZipFile(evil, "w") as zf:
        zf.writestr("../../escaped.npz", b"x")
        zf.writestr("../../escaped.json", json.dumps({"subject": "x"}))

    out = tmp_path / "out"
    imported = import_zip(evil, out_dir=out)

    assert not (tmp_path.parent / "escaped.npz").exists()
    assert not (tmp_path / "escaped.npz").exists()
    assert (out / "escaped.npz").exists()
    assert [p.parent for p in imported] == [out]


def test_zip_without_sessions_is_rejected(tmp_path):
    empty = tmp_path / "e.zip"
    with zipfile.ZipFile(empty, "w") as zf:
        zf.writestr("readme.txt", "hello")
    with pytest.raises(IngestError, match="no .npz/.json"):
        import_zip(empty, out_dir=tmp_path / "out")


# --------------------------------------------------------------------------
# Inventory
# --------------------------------------------------------------------------


def test_inventory_groups_multiple_people(tmp_path):
    _native_session(tmp_path, subject="alice", n_trials=6)
    _native_session(tmp_path, subject="bob", n_trials=8)

    rows = inventory(tmp_path)
    assert {r["subject"] for r in rows} == {"alice", "bob"}
    assert sum(r["n_trials"] for r in rows) == 14
    assert all(r["montage_rating"] == "good" for r in rows)


def test_inventory_of_empty_directory(tmp_path):
    assert inventory(tmp_path / "nothing-here") == []
