# EEG motor imagery classifier — project brief

## Goal
Build a working pipeline that classifies left-hand vs right-hand motor imagery
from EEG signals, using the PhysioNet EEG Motor Movement/Imagery Dataset
(EEGMMIDB). Implement both a classical pipeline (CSP + LDA/SVM) and a deep
learning pipeline (EEGNet), then compare them.

Author: Pranav Ganji (UT Austin, Statistics and Data Science)
Context: portfolio project for a competitive neurotech club application.
Complements an existing project (CerebroDX, medical imaging classifier) by
covering the signal-processing/BCI side of neurotech instead of imaging.

## Dataset
- PhysioNet EEGMMIDB: https://physionet.org/content/eegmmidb/1.0.0/
- 109 subjects, 64-channel EEG, 160 Hz sampling rate, EDF format, BCI2000 system.
- Use `MNE-Python`'s built-in dataset fetcher:
  `mne.datasets.eegbci.load_data(subjects=1, runs=[4])` — this downloads directly,
  no manual download needed.
- Relevant runs per subject: runs 4, 8, 12 are imagined left/right fist movement.
  Runs 6, 10, 14 are imagined fists/feet (for later 4-class extension).
- Start with ONE subject to get the pipeline working end-to-end, then scale
  to multiple subjects.

## Pipeline stages (implement in this order, get each working before moving on)

### 1. Data loading (`src/preprocess.py`)
- Load EDF via `mne.io.read_raw_edf`.
- Extract annotations/events (T1 = left fist, T2 = right fist in this dataset's
  convention — verify against MNE's eegbci documentation, don't assume).
- Set montage to standard 64-channel layout (`mne.channels.make_standard_montage('standard_1005')`).

### 2. Preprocessing (`src/preprocess.py`)
- Bandpass filter 8-30 Hz (`raw.filter(8, 30)`).
- Notch filter 60 Hz if needed (US powerline).
- Common Average Reference: `raw.set_eeg_reference('average')`.
- Epoch into trials: `mne.Epochs`, tmin=0.5, tmax=3.5 relative to event onset (avoid
  onset transient).

### 3. Feature extraction (`src/features.py`)
Implement three approaches, each as a separate function, so they can be compared:
- **Band power**: Welch's method power in 8-30 Hz per channel, or bandpass +
  variance per channel. This is the simplest baseline — get this working first.
- **CSP**: Use `mne.decoding.CSP` (n_components=6). This is the core BCI method —
  read MNE's CSP source/docs and be able to explain what it's doing, not just
  call it as a black box.
- **Wavelet features** (stretch goal, do last): discrete wavelet transform (db4),
  energy per sub-band per channel via `pywt`.

### 4. Classification (`src/models.py`)
- Classical: CSP features -> LDA (`sklearn.discriminant_analysis.LinearDiscriminantAnalysis`)
  and separately -> SVM with RBF kernel (`sklearn.svm.SVC`). CSP+LDA is the standard
  reference pipeline in BCI literature — get this working and treat it as the
  primary baseline to beat.
- Deep learning: implement EEGNet (compact CNN for EEG, ~2K params). Use PyTorch.
  Look up the original EEGNet paper architecture (temporal conv -> depthwise
  spatial conv -> separable conv -> classifier) rather than copying an
  unofficial implementation verbatim — should be able to explain each layer's
  purpose (temporal conv = frequency filters, depthwise spatial conv = CSP-like
  spatial filtering, learned).

### 5. Evaluation (`src/evaluate.py`)
- Subject-dependent: train/test split within one subject (stratified, e.g. 80/20
  or 5-fold CV). Get this working first — it's the easy win.
- Cross-subject: train on N-1 subjects, test on 1 held-out subject, repeat
  (leave-one-subject-out). This is harder and accuracy will likely drop —
  that's expected and worth discussing, not a bug to "fix."
- Metrics: accuracy, confusion matrix (`sklearn.metrics.confusion_matrix`),
  and report per-class precision/recall since classes should be balanced but
  verify.
- Generate: accuracy bar chart (band power vs CSP+LDA vs CSP+SVM vs EEGNet),
  confusion matrices, and an ERD/ERS topomap (`mne.viz.plot_topomap`) showing
  spatial power distribution during left vs right imagery — this figure is
  the one that visually reads as "real neuroscience" rather than generic ML.

## Coding conventions
- Python, type hints on function signatures.
- Docstrings explaining the *neuroscience/signal-processing rationale*, not just
  what the code does mechanically (e.g. "bandpass 8-30Hz to isolate mu/beta
  rhythms associated with motor imagery" not just "apply filter").
- Keep preprocessing, features, models, and evaluation in separate modules
  (see repo structure below) — don't put everything in one script.
- Prefer explicit, readable code over clever one-liners; this needs to be
  explainable in an interview.

## Repo structure
```
eeg-motor-imagery/
├── data/                  # downloaded EDF files (gitignored)
├── src/
│   ├── preprocess.py      # loading, filtering, referencing, epoching
│   ├── features.py        # band power, CSP, wavelet extraction
│   ├── models.py          # LDA/SVM + EEGNet definitions
│   ├── train.py           # training loop, CV logic
│   └── evaluate.py        # metrics, confusion matrices, plots
├── notebooks/
│   └── exploration.ipynb  # visualize raw signals, sanity checks, ERD/ERS plots
├── results/
│   └── figures/           # saved output figures
└── README.md
```

## Milestones (build in this order, don't skip ahead)
1. Load + epoch one subject's data, plot raw signal, confirm event markers make sense.
2. Band power features + LDA -> first working accuracy number.
3. CSP + LDA -> compare against band power baseline.
4. CSP + SVM -> compare against LDA.
5. EEGNet -> compare classical vs deep learning.
6. Cross-subject evaluation.
7. Final figures + README writeup.

## What "done" looks like
A README with: problem statement, dataset description, pipeline diagram/summary,
results table (method x accuracy, subject-dependent and cross-subject), and 2-3
figures (accuracy comparison, confusion matrix, ERD/ERS topomap). Code should
run end-to-end from `python src/train.py` with a clear entry point.
