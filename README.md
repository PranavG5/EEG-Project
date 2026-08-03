# EEG motor imagery classifier

Classifies left-hand vs right-hand motor imagery from EEG using the PhysioNet
EEG Motor Movement/Imagery Dataset (EEGMMIDB). It implements both a classical
signal-processing pipeline (band power / CSP with LDA and SVM) and a deep
learning pipeline (EEGNet), and compares them under **subject-dependent** and
**cross-subject** evaluation.

Author: Pranav Ganji

## Why this project
Complements CerebroDX (a medical imaging classifier) by covering the
signal-processing / brain-computer interface side of neurotech: decoding
real-time neural signals rather than static images.

## Problem & dataset
- **Task:** binary decode of imagined left- vs right-fist movement from scalp EEG.
- **Data:** PhysioNet EEGMMIDB — 64-channel EEG at 160 Hz, BCI2000 system.
  Imagery runs 4, 8, 12 carry the left/right-fist trials (annotations T1 = left
  fist, T2 = right fist). This build uses the first **10 subjects**, giving
  **450 trials** (230 left / 220 right; majority-class chance ≈ 0.51).
- Data downloads automatically via MNE on first run (cached under `~/mne_data`);
  no manual download needed.

## Setup
```bash
pip install -r requirements.txt
python src/train.py      # runs the full pipeline end-to-end, writes figures
```

## The app 🧠
**NeuroDecode** is a local web app for the decoder — one command, one URL, clean
UI. Pick a demo subject or **upload your own EDF recording**, choose a decoder,
and it shows held-out accuracy, a trial-by-trial decode with confidence bars, a
confusion matrix, per-class accuracy, and the ERD/ERS brain map.

```bash
pip install -r requirements.txt
python run_app.py            # opens http://127.0.0.1:8000 in your browser
```

A single FastAPI server serves both the REST API and the frontend, so there is
no build step and nothing to deploy. The API is documented and browsable at
`/docs`, which also makes the decoder scriptable:

```bash
curl -X POST localhost:8000/api/decode -F decoder=eegnet -F files=@my_session.edf
```

Architecture and a guide to adding decoders, panels and data sources:
[`backend/README.md`](backend/README.md). The design is deliberately modular —
adding a decoder is one registry entry and it appears in the UI automatically —
so the app is ready for the planned physical-EEG phase.

<details>
<summary>Alternative: Streamlit version</summary>

`src/app.py` is an earlier, simpler Streamlit UI with the same core features
plus a trained-EEGNet download button: `streamlit run src/app.py`.
</details>

## Live demo (terminal)
`src/demo.py` replays one subject's trials as if the BCI were decoding in real
time — for each trial it prints the imagined hand, the decoder's guess, a
confidence bar, and a running accuracy. It uses honest held-out (cross-validated)
predictions and runs in seconds, so it is safe to run live.
```bash
python src/demo.py                              # subject 1, CSP + LDA
python src/demo.py --subject 7 --method csp_lda  # a cleanly-decoding subject (~0.96)
python src/demo.py --subject 9                   # a hard subject (~0.38) — decoding varies by person
python src/demo.py --method eegnet --delay 0.3   # replay with the deep model, paced for effect
```
Per-subject accuracy varies a lot (subjects 2 and 7 decode well, 5 and 9 poorly)
— that between-person spread is itself a real, presentable finding.

## Pipeline
```
Raw EDF (64 ch, 160 Hz)
  -> band-pass 8-30 Hz (isolate mu/beta sensorimotor rhythms)
  -> common average reference (sharpen focal contralateral ERD)
  -> epoch T1/T2 cues, 0.5-3.5 s post-cue
  -> features:  band power (per-channel log 8-30 Hz power)
                CSP        (supervised spatial filters, log-variance)
                — or raw epoch tensor fed straight to EEGNet
  -> classify:  LDA (shrinkage) / SVM (RBF) / EEGNet (compact CNN)
  -> evaluate:  subject-dependent (within-subject 5-fold CV)
                cross-subject     (leave-one-subject-out)
```

Modules: `src/preprocess.py` (load/filter/reference/epoch, multi-subject
assembly), `src/features.py` (band power, CSP), `src/models.py` (LDA, SVM,
EEGNet + sklearn wrapper + save/load), `src/evaluate.py` (CV metrics, confusion
matrices, topomap), `src/train.py` (full-study entry point), `src/demo.py`
(terminal live-decode demo), `backend/` + `run_app.py` (the NeuroDecode app).

### EEGNet
A compact CNN (~2.7K parameters) that learns the pipeline end-to-end, with each
block mapping onto a signal-processing idea:
1. **Temporal convolution** — a bank of learned band-pass filters over time.
2. **Depthwise spatial convolution** — learned, CSP-like spatial filters across
   channels (with a max-norm constraint, as in the original paper).
3. **Separable convolution** — a low-parameter summary of the temporal dynamics.
4. **Dense classifier** on the average-pooled features.

Trained with Adam + cross-entropy, early stopping on a held-out validation
split, dropout and max-norm regularisation, and per-channel z-scoring fit on the
training fold only. Epochs are decimated 160 → 80 Hz inside the wrapper (the
signal is already band-limited to 30 Hz, so this is loss-free and halves compute).

## Results
10 subjects, 450 trials, majority-class chance ≈ 0.51. Subject-dependent =
within-subject 5-fold CV averaged over the 10 subjects; cross-subject =
leave-one-subject-out. Values are mean ± std across subjects.

| Method | Subject-dependent acc | Cross-subject acc |
|---|---|---|
| Band power + LDA | 0.653 ± 0.124 | 0.496 ± 0.097 |
| CSP + LDA | 0.602 ± 0.167 | 0.520 ± 0.067 |
| CSP + SVM (RBF) | 0.598 ± 0.177 | 0.507 ± 0.041 |
| EEGNet | 0.607 ± 0.138 | 0.527 ± 0.065 |

### Reading the results
- **Subject-dependent (0.60–0.65):** all four methods clear chance by a clear
  margin once calibrated to an individual. The simple per-channel band-power
  baseline is competitive with — even slightly ahead of — CSP and EEGNet here,
  which is expected on this little data per subject (~45 trials): CSP's global
  spatial filters and EEGNet's ~2.7K weights have little to fit, while band power
  is low-variance and hard to beat in that regime. The large ± reflects genuine
  between-subject spread in how cleanly people modulate their mu/beta rhythm.
- **Cross-subject (near chance):** leaving a whole subject out is much harder —
  accuracy collapses toward the 0.51 chance line for every method. Anatomy,
  electrode placement and imagery strategy differ enough between people that a
  decoder trained on others transfers poorly. **This gap is the expected,
  interesting result, not a bug**: it is exactly why real BCIs include a short
  per-user calibration. With ~400 training trials in this regime, EEGNet
  generalises marginally best of the four (0.527 — the highest cross-subject
  score), edging out CSP + LDA (0.520), but the margin is small and transfer
  still sits only just above chance. A decisive deep-learning advantage would
  need far more subjects and trials than 10 × ~45; the value of EEGNet here is
  that it reaches parity with the hand-built pipelines while learning its
  filters end-to-end.

### Figures (`results/figures/`)
- `accuracy_comparison_combined.png` — subject-dependent vs cross-subject, all
  methods side by side (the generalisation gap at a glance).
- `accuracy_comparison_subject_dependent.png`, `accuracy_comparison_cross_subject.png`.
- `cross_subject_confusion_*.png`, `subject_dependent_confusion_*.png` — per-method
  confusion matrices.
- `erd_ers_topomap.png` — grand-average scalp maps of 8-30 Hz power during left
  vs right imagery and their lateralisation contrast: the direct spatial
  signature of contralateral ERD around C3/C4 that the spatial filters exploit.
- `milestone1_raw_signal.png`, `milestone1_events.png` — early sanity checks.

## Milestone status
1. ✅ Load + epoch one subject, plot raw signal, confirm event markers.
2. ✅ Band power + LDA baseline.
3. ✅ CSP + LDA reference pipeline.
4. ✅ CSP + SVM.
5. ✅ EEGNet — classical vs deep learning.
6. ✅ Cross-subject (leave-one-subject-out) evaluation.
7. ✅ Final figures + writeup.
