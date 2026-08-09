"""Central configuration for the motor-imagery pipeline.

Everything downstream (offline training on PhysioNet, live decoding from a
consumer headband) is parameterised from here so that the *same* signal
processing runs on both. The single most important idea in this file is
``CANONICAL_SFREQ``: research data and consumer hardware sample at different
rates, and a classifier trained at one rate produces garbage at another
because every filter and every learned temporal kernel is defined in samples,
not seconds. We therefore resample everything to one common rate at the
boundary of the system.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
RECORDINGS_DIR = DATA_DIR / "recordings"  # own headband recordings
MODELS_DIR = REPO_ROOT / "models"
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------

#: Every signal is resampled to this rate before filtering/feature extraction.
#: 128 Hz is chosen because (a) it is comfortably above the Nyquist rate for
#: the 30 Hz upper edge of the beta band we care about, leaving headroom for
#: the anti-alias filter, and (b) it divides the 256 Hz rate of most consumer
#: headbands (Muse, Neurosity Crown) exactly by 2, so decimation introduces no
#: resampling artefacts on the live path where latency matters.
CANONICAL_SFREQ = 128.0

#: PhysioNet EEGMMIDB native rate (BCI2000 system).
PHYSIONET_SFREQ = 160.0

# --------------------------------------------------------------------------
# Spectral bands
# --------------------------------------------------------------------------

#: Motor imagery is decoded from event-related desynchronisation (ERD): when
#: you imagine moving a limb, the sensorimotor cortex contralateral to that
#: limb *drops* in power in the mu (8-13 Hz) and beta (13-30 Hz) rhythms. The
#: mu rhythm is the idling rhythm of motor cortex, so engaging that cortex
#: desynchronises it. Bandpassing 8-30 Hz keeps exactly this and discards
#: slow drift, eye movement (<4 Hz) and muscle EMG (mostly >40 Hz).
BANDPASS_LOW = 8.0
BANDPASS_HIGH = 30.0

#: Sub-bands used by the band-power baseline. Splitting mu from beta helps
#: because their ERD time courses and spatial extents differ slightly.
SUB_BANDS: dict[str, tuple[float, float]] = {
    "mu": (8.0, 13.0),
    "beta_low": (13.0, 20.0),
    "beta_high": (20.0, 30.0),
}

#: US powerline frequency. Consumer headbands are battery powered and often
#: cleaner than mains-powered rigs, but a laptop on a charger still radiates.
NOTCH_FREQ = 60.0

# --------------------------------------------------------------------------
# Epoching
# --------------------------------------------------------------------------

#: Seconds relative to cue onset. We skip the first 0.5 s because the visual
#: cue itself evokes a large potential (a visual ERP) that has nothing to do
#: with motor imagery, and because subjects take a few hundred milliseconds to
#: actually start imagining. ERD is sustained, so 0.5-3.5 s is the meat of it.
EPOCH_TMIN = 0.5
EPOCH_TMAX = 3.5

#: Baseline window (relative to cue) used for ERD/ERS percentage plots. This
#: is the "resting" mu power that ERD is expressed as a decrease *from*.
BASELINE_TMIN = -2.0
BASELINE_TMAX = -0.2

EPOCH_LENGTH_S = EPOCH_TMAX - EPOCH_TMIN

#: Peak-to-peak amplitude above which a trial is discarded as artefact, in
#: microvolts, measured after the 8-30 Hz bandpass. Calibrated against the
#: PhysioNet data, where the median trial's worst channel swings ~150 uV and
#: the 90th percentile is ~205 uV: 250 uV therefore removes the clear outliers
#: without throwing away half the dataset. Dry-electrode headband recordings
#: are noisier and may need this raised — but if you find yourself raising it
#: past ~400 uV, the electrodes need reseating, not the threshold loosening.
REJECT_PEAK_TO_PEAK_UV = 250.0

# --------------------------------------------------------------------------
# Classes
# --------------------------------------------------------------------------

#: PhysioNet EEGMMIDB annotation convention for runs 4/8/12 (imagined fist
#: movement): T0 = rest, T1 = imagine LEFT fist, T2 = imagine RIGHT fist.
#: Verified against the annotations embedded in the EDF files themselves.
PHYSIONET_EVENT_MAP = {"T1": "left", "T2": "right"}

CLASSES = ("left", "right")
CLASS_TO_INT = {"left": 0, "right": 1}
INT_TO_CLASS = {v: k for k, v in CLASS_TO_INT.items()}

#: Runs containing *imagined* left vs right fist movement.
IMAGERY_RUNS = (4, 8, 12)
#: Runs containing *executed* left vs right fist movement. Executed movement
#: produces a far stronger, more reliable ERD than imagery and is the correct
#: thing to test hardware with before blaming your classifier.
EXECUTED_RUNS = (3, 7, 11)

# --------------------------------------------------------------------------
# Montage
# --------------------------------------------------------------------------

MONTAGE_NAME = "standard_1005"

#: Electrodes sitting over or immediately adjacent to the hand region of the
#: primary motor cortex / sensorimotor strip. Left-vs-right hand imagery is a
#: fundamentally *lateralised* contrast, so a montage that does not straddle
#: the midline over these sites cannot see the effect at all, no matter how
#: good the classifier is.
MOTOR_CHANNELS_CORE = ("C3", "C4", "Cz")
MOTOR_CHANNELS_EXTENDED = (
    "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6",
)

# --------------------------------------------------------------------------
# Live decoding
# --------------------------------------------------------------------------

#: Length of the sliding window fed to the classifier online. Matches
#: EPOCH_LENGTH_S so that the live feature distribution matches training.
LIVE_WINDOW_S = EPOCH_LENGTH_S

#: How often we slide the window forward. 250 ms gives 4 predictions/second,
#: which feels responsive without the windows being so correlated that
#: smoothing becomes meaningless.
LIVE_STEP_S = 0.25

#: Number of consecutive window predictions averaged before a decision is
#: emitted. Single-window motor-imagery predictions are noisy; majority
#: voting over ~1 s of windows trades latency for a large accuracy gain.
LIVE_SMOOTHING_WINDOWS = 4

#: Below this smoothed probability the decoder reports "uncertain" rather than
#: guessing. A BCI that abstains is far more usable than one that flickers.
LIVE_CONFIDENCE_THRESHOLD = 0.65

# --------------------------------------------------------------------------
# Misc
# --------------------------------------------------------------------------

RANDOM_SEED = 42
N_CSP_COMPONENTS = 6
