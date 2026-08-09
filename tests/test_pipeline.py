"""Tests for the parts that fail silently rather than loudly.

Priority here is the hardware boundary. A bug in the classifier shows up as
bad accuracy, which is visible. A bug in channel-name harmonisation or buffer
ordering shows up as *plausible* accuracy computed on permuted electrodes,
which is not.

Run with:  python -m pytest tests/ -q
"""

from __future__ import annotations

import numpy as np
import pytest

from src.config import CANONICAL_SFREQ, CLASS_TO_INT
from src.devices.base import RingBuffer, eeg_channel_subset
from src.devices.channels import assess_montage, canonical_name, shared_channels
from src.devices.registry import DEVICE_PROFILES, get_profile


# --------------------------------------------------------------------------
# Channel naming
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("C3", "C3"),
        ("c3", "C3"),
        ("Fc5.", "FC5"),          # PhysioNet EDF spelling
        ("EEG C3-REF", "C3"),     # clinical EDF spelling
        ("Cz..", "Cz"),
        ("FCZ", "FCz"),           # midline z must be lowercase for MNE montages
        ("FP1", "Fp1"),
        ("T3", "T7"),             # legacy 10-20 alias
        ("A1", "TP9"),            # mastoid reference alias
    ],
)
def test_canonical_name_normalises_vendor_spellings(raw, expected):
    assert canonical_name(raw) == expected


@pytest.mark.parametrize("junk", ["Accel X", "battery", "timestamp", "OTHER", "Counter"])
def test_canonical_name_rejects_non_electrodes(junk):
    # Non-EEG channels must be filtered out; a near-constant battery channel
    # would give CSP a singular covariance matrix to invert.
    assert canonical_name(junk) is None


def test_eeg_channel_subset_keeps_indices_aligned():
    raw_names = ["Counter", "C3", "Accel X", "C4", "battery"]
    idx, names = eeg_channel_subset(raw_names)
    assert idx == [1, 3]
    assert names == ["C3", "C4"]


# --------------------------------------------------------------------------
# Montage assessment — the buying decision
# --------------------------------------------------------------------------


def test_muse_montage_is_unsuitable():
    """Muse has no sensorimotor coverage; this must not silently pass."""
    result = assess_montage(["TP9", "AF7", "AF8", "TP10"])
    assert result.rating == "unsuitable"
    assert not result.usable
    assert result.core_motor == ()


def test_crown_montage_is_good():
    result = assess_montage(["CP3", "C3", "F5", "PO3", "PO4", "F6", "C4", "CP4"])
    assert result.rating == "good"
    assert set(result.core_motor) == {"C3", "C4"}
    assert ("C3", "C4") in result.lateral_pairs


def test_single_hemisphere_is_not_good():
    """Left-vs-right is a lateralised contrast; one side cannot resolve it."""
    result = assess_montage(["C3", "CP3", "FC3"])
    assert result.rating == "marginal"


def test_channel_count_does_not_override_placement():
    """A 14-channel frontal montage must rate worse than 2 central electrodes."""
    many_frontal = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "AF3", "AF4",
                    "O1", "O2", "P7", "P8", "Pz"]
    assert assess_montage(many_frontal).rating == "unsuitable"
    assert assess_montage(["C3", "C4"]).rating == "good"


def test_registry_ratings_are_self_consistent():
    for key, profile in DEVICE_PROFILES.items():
        if not profile.default_channels:
            continue
        assert profile.montage_rating == assess_montage(
            list(profile.default_channels)
        ).rating, f"{key} rating disagrees with its own channel list"


def test_get_profile_rejects_unknown_device():
    with pytest.raises(KeyError):
        get_profile("definitely-not-a-headset")


# --------------------------------------------------------------------------
# Dataset intersection
# --------------------------------------------------------------------------


def test_shared_channels_uses_dataset_order():
    """Feature layout must be stable, so ordering comes from the dataset."""
    dataset = ["FC5", "C3", "Cz", "C4", "CP3", "CP4"]
    device = ["C4", "C3", "CP4", "CP3"]  # deliberately different order
    assert shared_channels(device, dataset) == ["C3", "C4", "CP3", "CP4"]


def test_shared_channels_survives_spelling_differences():
    assert shared_channels(["c3", "c4"], ["C3.", "Cz.", "C4."]) == ["C3", "C4"]


# --------------------------------------------------------------------------
# Ring buffer — ordering bugs here permute the live feature vector
# --------------------------------------------------------------------------


def test_ring_buffer_preserves_temporal_order():
    buf = RingBuffer(n_channels=2, n_samples=5)
    buf.push(np.array([[1, 2, 3], [1, 2, 3]], dtype=float))
    buf.push(np.array([[4, 5, 6], [4, 5, 6]], dtype=float))
    assert buf.is_full
    # Oldest sample first, newest last.
    np.testing.assert_array_equal(buf.snapshot()[0], [2, 3, 4, 5, 6])


def test_ring_buffer_handles_oversized_chunk():
    buf = RingBuffer(n_channels=1, n_samples=3)
    buf.push(np.arange(10, dtype=float)[None, :])
    np.testing.assert_array_equal(buf.snapshot()[0], [7, 8, 9])


def test_ring_buffer_not_full_until_enough_samples():
    buf = RingBuffer(n_channels=1, n_samples=4)
    buf.push(np.ones((1, 3)))
    assert not buf.is_full
    buf.push(np.ones((1, 1)))
    assert buf.is_full


# --------------------------------------------------------------------------
# End-to-end on synthetic data
# --------------------------------------------------------------------------


def _synthetic_lateralised_epochs(n_per_class=30, n_times=384, seed=0):
    """Epochs with a planted contralateral ERD, for testing the plumbing.

    Channel 0 stands in for C3 and channel 1 for C4. "Left" trials get reduced
    mu amplitude on C4, "right" trials on C3 — the real contralateral pattern.
    Any working pipeline must decode this far above chance; if it cannot, the
    bug is in the plumbing, not the physiology.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_times) / CANONICAL_SFREQ
    x, y = [], []
    for label, idx in CLASS_TO_INT.items():
        for _ in range(n_per_class):
            mu = np.sin(2 * np.pi * 10 * t + rng.uniform(0, 2 * np.pi))
            suppressed, normal = 0.3, 1.0
            c3 = (suppressed if label == "right" else normal) * mu
            c4 = (suppressed if label == "left" else normal) * mu
            trial = np.stack([c3, c4]) + rng.normal(0, 0.25, (2, n_times))
            x.append(trial)
            y.append(idx)
    return np.asarray(x), np.asarray(y)


@pytest.mark.parametrize("model_name", ["bandpower_lda", "csp_lda", "csp_svm"])
def test_classical_models_decode_planted_erd(model_name):
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    from src.models import build_model

    x, y = _synthetic_lateralised_epochs()
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    score = cross_val_score(build_model(model_name), x, y, cv=cv).mean()
    assert score > 0.9, f"{model_name} scored {score:.3f} on a planted signal"


def test_shuffled_labels_give_chance_accuracy():
    """The leak control: with random labels nothing should be learnable.

    If this test ever fails, a transform is being fitted outside the CV folds.
    """
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    from src.models import build_model

    x, _ = _synthetic_lateralised_epochs()
    rng = np.random.default_rng(1)
    y_shuffled = rng.permutation(np.repeat([0, 1], 30))
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    score = cross_val_score(build_model("csp_lda"), x, y_shuffled, cv=cv).mean()
    assert score < 0.7, f"csp_lda scored {score:.3f} on shuffled labels — label leak"


def test_preprocess_resamples_to_canonical_rate():
    """A model trained at one rate is meaningless at another."""
    from src.preprocess import preprocess_raw, raw_from_array

    rng = np.random.default_rng(0)
    data = rng.normal(0, 20e-6, (4, 256 * 10))  # 10 s at a headband's 256 Hz
    raw = raw_from_array(data, 256.0, ["C3", "C4", "CP3", "CP4"])
    out = preprocess_raw(raw)
    assert out.info["sfreq"] == pytest.approx(CANONICAL_SFREQ)
    assert out.ch_names == ["C3", "C4", "CP3", "CP4"]


def test_epoch_labels_are_stable_regardless_of_event_order():
    """0 must always be 'left' and 1 always 'right'.

    An inverted mapping presents as a suspiciously consistent below-chance
    accuracy, which is easy to misread as a modelling problem.
    """
    from src.preprocess import epoch_windows, preprocess_raw, raw_from_array

    rng = np.random.default_rng(0)
    data = rng.normal(0, 20e-6, (2, 256 * 40))
    raw = preprocess_raw(raw_from_array(data, 256.0, ["C3", "C4"]))

    onsets = np.array([5.0, 12.0, 19.0, 26.0])
    labels = np.array([1, 0, 1, 0])  # right, left, right, left
    _, y = epoch_windows(raw, onsets, labels, reject_uv=None)
    np.testing.assert_array_equal(y, labels)
