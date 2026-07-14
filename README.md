# EEG motor imagery classifier

Classifies left-hand vs right-hand motor imagery from EEG signals using the
PhysioNet EEG Motor Movement/Imagery Dataset. Implements both a classical
signal-processing pipeline (CSP + LDA/SVM) and a deep learning pipeline
(EEGNet), and compares them under subject-dependent and cross-subject
evaluation.

Author: Pranav Ganji

## Why this project
Complements CerebroDX (a medical imaging classifier) by covering the
signal-processing / brain-computer interface side of neurotech: decoding
real-time neural signals rather than static images.

## Status
🚧 In progress — see `CLAUDE.md` for the full build spec and milestone plan.

## Setup
```bash
pip install -r requirements.txt
```
Data downloads automatically via MNE on first run (no manual download needed).

## Pipeline
Raw EDF -> filter (8-30Hz) -> re-reference -> epoch -> extract features
(band power / CSP / wavelets) -> classify (LDA / SVM / EEGNet) -> evaluate
(subject-dependent + cross-subject).

## Results
_Filled in as milestones complete. Subject-dependent numbers below are for
subject 1 across its three imagery runs (4, 8, 12), 45 trials, 5-fold
stratified cross-validation. Cross-subject columns are still to come._

| Method | Subject-dependent acc | Cross-subject acc |
|---|---|---|
| Band power + LDA | 0.73¹ | — |
| CSP + LDA | 0.60¹ | — |
| CSP + SVM (RBF) | 0.64¹ | — |
| EEGNet | — | — |

¹ Subject 1, runs 4+8+12 (45 trials, majority-class chance ≈ 0.51), 5-fold
CV, mean ± 0.09 across folds. Note band power currently edges out CSP here:
concatenating three separately-recorded runs introduces session-to-session
covariance shifts that CSP's global spatial filters are sensitive to, whereas
per-channel band power is more robust to them. Whether CSP recovers its usual
edge with per-run or multi-subject training is worth checking as more data is
added.
