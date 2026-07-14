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
_(to be filled in as milestones complete)_

| Method | Subject-dependent acc | Cross-subject acc |
|---|---|---|
| Band power + LDA | — | — |
| CSP + LDA | — | — |
| CSP + SVM | — | — |
| EEGNet | — | — |
