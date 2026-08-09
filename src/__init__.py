"""EEG motor-imagery decoding: offline research pipeline + live headband decoding."""

import os

__version__ = "0.2.0"

# MNE logs rank estimation and filter design for every CSP fit, which buries the
# actual results under hundreds of lines during cross-validation. Set
# MNE_LOG_LEVEL to override (e.g. "INFO" when debugging a preprocessing bug).
try:  # pragma: no cover - MNE is a hard dependency, but importing src should not explode
    import mne

    mne.set_log_level(os.environ.get("MNE_LOG_LEVEL", "ERROR"))
except ImportError:
    pass
