"""Hardware layer: everything that knows what a headband is.

The rest of the pipeline depends only on :class:`EEGSource` and the canonical
electrode naming in :mod:`.channels`, so swapping headsets is a flag change.
"""

from .base import EEGSource, RingBuffer, StreamInfo, eeg_channel_subset
from .channels import (
    MontageAssessment,
    assess_montage,
    canonical_name,
    canonicalize_all,
    shared_channels,
)
from .registry import (
    DEVICE_PROFILES,
    DeviceProfile,
    describe_all,
    describe_device,
    get_profile,
)

__all__ = [
    "DEVICE_PROFILES",
    "DeviceProfile",
    "EEGSource",
    "MontageAssessment",
    "RingBuffer",
    "StreamInfo",
    "assess_montage",
    "canonical_name",
    "canonicalize_all",
    "describe_all",
    "describe_device",
    "eeg_channel_subset",
    "get_profile",
    "shared_channels",
]


def open_source(device_key: str, **kwargs: object) -> EEGSource:
    """Construct a live source for a device key.

    Imported lazily so that the offline training pipeline never pays the cost
    of loading BrainFlow's native library.
    """
    from .brainflow_device import BrainFlowSource

    return BrainFlowSource(device_key, **kwargs)  # type: ignore[arg-type]
