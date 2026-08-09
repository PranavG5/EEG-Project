"""Electrode-name harmonisation and montage assessment.

The hard problem when moving from a research dataset to a consumer headband
is not the classifier — it is that the two systems do not record the same
places on the head. A model trained on 64 channels cannot be applied to a
4-channel headband, and more subtly, a model trained on channels the headband
*does not have* is not merely less accurate, it is measuring a different
physical quantity.

This module answers two questions:

1. Which electrodes does this device share with the training montage?
2. Does this device's montage physically permit left-vs-right hand decoding?

Question 2 is the one that decides whether a purchase was a good idea.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..config import MOTOR_CHANNELS_CORE, MOTOR_CHANNELS_EXTENDED

# --------------------------------------------------------------------------
# Name canonicalisation
# --------------------------------------------------------------------------

#: Canonical 10-05 spelling for every label we expect to meet. EEG vendors are
#: gloriously inconsistent: PhysioNet EDF headers write "Fc5.", BrainFlow
#: writes "FC5", some SDKs write "fc5" or "EEG FC5-REF".
_CANONICAL_FORMS = {
    # frontopolar / frontal
    "FP1", "FPZ", "FP2", "AF7", "AF3", "AFZ", "AF4", "AF8",
    "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8",
    # fronto-central
    "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8",
    # central
    "T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8", "T9", "T10",
    # centro-parietal
    "TP7", "TP9", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8", "TP10",
    # parietal / occipital
    "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8",
    "PO7", "PO3", "POZ", "PO4", "PO8", "O1", "OZ", "O2", "IZ",
}

def _display_form(upper_name: str) -> str:
    """Spell an all-caps label the way MNE's standard_1005 montage does.

    MNE's montage lookup is case sensitive, so "FCZ" silently fails to match
    while "FCz" works. Two rules cover the whole 10-05 system: a trailing
    midline "Z" is lowercase, and the frontopolar prefix is "Fp".
    """
    name = upper_name[:-1] + "z" if upper_name.endswith("Z") else upper_name
    if name.startswith("FP"):
        name = "Fp" + name[2:]
    return name


#: Correct capitalisation for display and for matching MNE's montage.
_DISPLAY_FORM = {name: _display_form(name) for name in _CANONICAL_FORMS}

#: Legacy 10-20 names that map onto modern 10-05 positions.
_ALIASES = {
    "T3": "T7",
    "T4": "T8",
    "T5": "P7",
    "T6": "P8",
    "A1": "TP9",  # left mastoid, commonly used as reference
    "A2": "TP10",  # right mastoid
    "M1": "TP9",
    "M2": "TP10",
}


def canonical_name(raw_name: str) -> str | None:
    """Normalise one vendor-specific electrode label to 10-05 spelling.

    Strips the junk that real headers carry — trailing dots from the PhysioNet
    EDF spec, ``EEG `` prefixes, ``-REF``/``-LE`` reference suffixes — then
    resolves legacy 10-20 aliases. Returns ``None`` for anything that is not a
    recognisable scalp electrode (accelerometer, battery, counter, ...), which
    is how callers filter non-EEG channels out of a device's channel list.
    """
    name = raw_name.strip().upper()
    name = re.sub(r"^EEG[\s_-]*", "", name)
    # "FC5-REF", "C3-LE" -> everything before the reference suffix.
    name = name.split("-")[0]
    name = name.replace(".", "").replace(" ", "").replace("_", "").strip()

    name = _ALIASES.get(name, name)
    if name in _CANONICAL_FORMS:
        return _DISPLAY_FORM[name]
    return None


def canonicalize_all(raw_names: list[str]) -> dict[str, str]:
    """Map each recognised input label to its canonical form, dropping the rest."""
    out: dict[str, str] = {}
    for raw in raw_names:
        canon = canonical_name(raw)
        if canon is not None:
            out[raw] = canon
    return out


# --------------------------------------------------------------------------
# Montage assessment
# --------------------------------------------------------------------------

#: Approximate left-right mirror pairs over the sensorimotor strip. The
#: left-vs-right decoding signal lives almost entirely in the *difference*
#: between members of these pairs, because imagining the left hand
#: desynchronises the right hemisphere and vice versa.
LATERAL_MOTOR_PAIRS = (
    ("C3", "C4"),
    ("C5", "C6"),
    ("C1", "C2"),
    ("FC3", "FC4"),
    ("FC5", "FC6"),
    ("FC1", "FC2"),
    ("CP3", "CP4"),
    ("CP5", "CP6"),
    ("CP1", "CP2"),
)


@dataclass(frozen=True)
class MontageAssessment:
    """Verdict on whether a montage can support left-vs-right hand decoding."""

    n_channels: int
    core_motor: tuple[str, ...]
    extended_motor: tuple[str, ...]
    lateral_pairs: tuple[tuple[str, str], ...]
    rating: str  # "good" | "marginal" | "unsuitable"
    rationale: str

    @property
    def usable(self) -> bool:
        return self.rating != "unsuitable"

    def describe(self) -> str:
        lines = [
            f"Montage assessment: {self.rating.upper()}",
            f"  channels                : {self.n_channels}",
            f"  core motor (C3/C4/Cz)   : {', '.join(self.core_motor) or 'NONE'}",
            f"  sensorimotor strip total: {len(self.extended_motor)}",
            "  lateral mirror pairs    : "
            + (", ".join(f"{a}/{b}" for a, b in self.lateral_pairs) or "NONE"),
            f"  {self.rationale}",
        ]
        return "\n".join(lines)


def assess_montage(channel_names: list[str]) -> MontageAssessment:
    """Judge a device's electrode layout for left-vs-right hand motor imagery.

    The rating is driven by *lateral coverage of the sensorimotor strip*, not
    by channel count. A 14-channel headset with nothing over central sites
    scores worse than a 4-channel one with C3 and C4, because the neural
    effect being decoded — contralateral mu/beta ERD over the hand knob of
    the precentral gyrus — is simply not observable elsewhere. Frontal and
    occipital electrodes pick it up only as smeared volume conduction, and
    what a classifier usually latches onto there is eye or jaw muscle
    artefact correlated with the cue, not motor cortex.
    """
    canon = list(canonicalize_all(channel_names).values())
    present = set(canon)

    core = tuple(c for c in MOTOR_CHANNELS_CORE if c in present)
    extended = tuple(c for c in MOTOR_CHANNELS_EXTENDED if c in present)
    pairs = tuple((a, b) for a, b in LATERAL_MOTOR_PAIRS if a in present and b in present)

    has_c3_c4 = "C3" in present and "C4" in present

    if has_c3_c4 and len(extended) >= 4:
        rating = "good"
        rationale = (
            "C3 and C4 both present with additional sensorimotor coverage — this is "
            "the standard montage for left-vs-right hand imagery and should work."
        )
    elif has_c3_c4:
        rating = "good"
        rationale = (
            "C3 and C4 both present. This is the minimal viable montage; accuracy "
            "will be lower than a full cap but the contrast is physically observable."
        )
    elif len(pairs) >= 1:
        rating = "marginal"
        rationale = (
            "No C3/C4, but at least one lateral sensorimotor pair is present. Expect "
            "a weaker and more variable effect; verify with executed movement first."
        )
    elif len(extended) >= 1:
        rating = "marginal"
        rationale = (
            "Only one side of the sensorimotor strip is covered. Left-vs-right is a "
            "lateralised contrast, so single-hemisphere coverage loses most of the signal."
        )
    else:
        rating = "unsuitable"
        rationale = (
            "No electrodes over the sensorimotor strip. Left-vs-right hand imagery "
            "cannot be decoded from this montage; any accuracy above chance is almost "
            "certainly eye or muscle artefact, not motor cortex."
        )

    return MontageAssessment(
        n_channels=len(canon),
        core_motor=core,
        extended_motor=extended,
        lateral_pairs=pairs,
        rating=rating,
        rationale=rationale,
    )


def shared_channels(device_channels: list[str], dataset_channels: list[str]) -> list[str]:
    """Electrodes common to a device and a training dataset, in dataset order.

    Training on this intersection is what makes a PhysioNet-pretrained model
    even nominally applicable to headband data. Order is taken from the
    dataset so that the feature vector layout is stable and reproducible.
    """
    device_canon = set(canonicalize_all(device_channels).values())
    dataset_canon = canonicalize_all(dataset_channels)
    return [canon for canon in dataset_canon.values() if canon in device_canon]
