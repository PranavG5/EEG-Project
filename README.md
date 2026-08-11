# EEG motor imagery decoder — from dataset to headband

Decodes **which hand you are imagining moving** from EEG. Two halves:

1. A research pipeline on the PhysioNet EEG Motor Movement/Imagery Dataset —
   CSP + LDA/SVM and EEGNet, benchmarked subject-dependent and cross-subject.
2. A **hardware layer** that runs the same decoder live on a consumer EEG
   headband: connect, check signal quality, record your own calibration data,
   train on it, and decode continuously.

Author: Pranav Ganji

> **Buying a headband?** Read [`docs/HARDWARE.md`](docs/HARDWARE.md) first. Which
> device you pick determines whether this works at all — most consumer headbands
> physically cannot see this signal.
>
> **Already have an OpenBCI Ganglion?** Go straight to
> [`docs/GANGLION_QUICKSTART.md`](docs/GANGLION_QUICKSTART.md) — wiring, electrode
> placement, and every command in order.

---

## The 60-second version of why device choice matters

Imagining a left-hand movement suppresses the mu (8–13 Hz) rhythm over the
**right** motor cortex, near electrode position **C4**; imagining the right hand
does the mirror image at **C3**. Decoding "which arm" means measuring that
left–right difference across the top of the head.

A headband whose electrodes sit on the forehead and behind the ears — which
describes most of them, including Muse — is not near either site. It will still
produce above-chance accuracy, from eye and jaw movement, and that is a trap
rather than a result.

The project encodes this judgement in code:

```bash
$ python -m src.cli devices
```

```
Neurosity Crown  [crown]
  electrodes  : CP3, C3, F5, PO3, PO4, F6, C4, CP4  (fixed)
  Montage assessment: GOOD
    core motor (C3/C4/Cz)   : C3, C4
    lateral mirror pairs    : C3/C4, CP3/CP4

Muse 2  [muse2]
  electrodes  : TP9, AF7, AF8, TP10  (fixed)
  Montage assessment: UNSUITABLE
    No electrodes over the sensorimotor strip. Left-vs-right hand imagery
    cannot be decoded from this montage; any accuracy above chance is almost
    certainly eye or muscle artefact, not motor cortex.
```

---

## Opening the app

```bash
pip install -r requirements.txt
python -m src.app
```

Opens `http://localhost:8000`. Runs entirely on your machine on Python's
standard library — no account, no server, nothing leaves the laptop. Five tabs,
in the order you use them: **Hardware** (compare headsets, check signal quality)
→ **Record** (full-screen cued session) → **Data** (upload your friends'
recordings) → **Train** → **Live**.

Everything the app does is also a CLI command; they share the same code. See
[`docs/COLLECTING_DATA.md`](docs/COLLECTING_DATA.md) for the group workflow.

PhysioNet data downloads automatically via MNE on first run. `brainflow` and
`torch` are only needed for live hardware and EEGNet respectively; the classical
pipeline runs without either.

---

## Usage

### Benchmark on the research dataset (no hardware needed)

```bash
python -m src.train --source physionet --subjects 1 2 3 --cross-subject
```

### Estimate a headband's ceiling *before* buying it

Restricts the 64-channel research data to the electrodes a given device
physically has, so the resulting accuracy is an honest upper bound for that
hardware:

```bash
python -m src.train --source physionet --subjects 1 --device crown
python -m src.train --source physionet --subjects 1 --device muse2   # for contrast
```

### The live workflow, once hardware arrives

```bash
python -m src.cli check  --device crown                  # signal quality per electrode
python -m src.cli record --device crown --task executed  # prove the chain works
python -m src.cli record --device crown --task imagery   # ~20 min, 60 cued trials
python -m src.cli train  --source recordings --save models/me.joblib
python -m src.cli live   --device crown --model models/me.joblib
```

### Collecting data as a group

You need **one device for the whole group, not one each** — calibration is ~20
minutes per person, so one headband serves five friends in an afternoon.

```bash
python -m src.cli export --out my_recordings.zip   # on your friend's laptop
python -m src.cli import from_bob.zip              # on yours
python -m src.cli list                             # who has contributed what
```

Recordings made with other software import too — OpenBCI GUI CSV, BrainFlow
CSV, EDF — given a cue-times file:

```bash
python -m src.cli import raw.csv --subject dave --channels C3,C4,CP3,CP4 --labels cues.csv
```

`--channels` is the **position on the head in the file's column order**. The
file says "EXG Channel 0"; only the person who placed the electrodes knows it
was C3, and getting the order wrong trains on permuted electrodes while still
reporting a believable accuracy.

```
LEFT [-----------------#------] RIGHT   RIGHT   0.78
```

Supported out of the box via BrainFlow: OpenBCI (Cyton / Cyton+Daisy /
Ganglion), Neurosity Crown, g.tec Unicorn, Muse 2/S, FreeEEG32, plus a
synthetic board so the entire pipeline can be developed with no hardware at all.

---

## Pipeline

```
                 ┌─ PhysioNet EDF (64ch, 160 Hz) ─┐
                 │                                 │
                 └─ headband stream (4-16ch, 250/256 Hz) ─┐
                                                          │
                              ┌───────────────────────────┘
                              ▼
   channel harmonisation  →  notch 60 Hz  →  bandpass 8-30 Hz
        (10-05 names)          (mains)        (mu + beta rhythms)
                              ▼
   common average reference  →  resample to 128 Hz  →  epoch 0.5-3.5 s
                              ▼
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
   band power            CSP (n=6)              raw epochs
   (Welch, mu/beta)   (spatial filters)              │
        │                     │                      ▼
        └──────► LDA / SVM ◄──┘                   EEGNet
                              ▼
              accuracy · kappa · confusion · ERD topomap
```

Both data sources converge on the *same* preprocessing function. That is
deliberate: any difference between how training data and live data are
processed shows up as an unexplained accuracy collapse, because both halves look
correct in isolation.

### Design decisions worth explaining

- **Canonical 128 Hz.** Research data is 160 Hz, headbands are 250/256 Hz. Every
  filter and every learned temporal kernel is defined in *samples*, so a model
  trained at one rate is meaningless at another. 128 Hz clears the 30 Hz band
  comfortably and divides 256 exactly.
- **Channel selection before referencing.** An average reference over 64
  electrodes is a different signal from one over 8. Subsetting after referencing
  would make models untransferable between montages.
- **Sensorimotor channels by default.** On subject 1, restricting from all 64
  channels to the 21 sensorimotor electrodes moves CSP+LDA from 0.60 to 0.73 —
  CSP fitted on 64 channels with 45 trials is estimating a 64×64 covariance from
  far too little data.
- **Everything fits inside CV folds.** CSP is *supervised*; fitting it before
  splitting leaks test labels and inflates accuracy by 10–20 points. This is the
  most common bug in student BCI projects.
- **No baseline correction on epochs.** Correct for evoked potentials, wrong for
  band-power decoding — CSP is variance-based, so a per-trial DC shift adds
  nothing and can inject baseline artefacts into the whole trial.

---

## Results

Subject-dependent = stratified 5-fold within subject. Cross-subject = leave-one-
subject-out. Chance is **50%**.

### Device-montage comparison (PhysioNet subject 1, 5-fold)

Restricting the research data to each device's actual electrodes — an upper
bound on what that hardware can do, with research-grade signal quality:

| Montage | Channels | Band power + LDA | CSP + LDA | CSP + SVM |
|---|---|---|---|---|
| Full sensorimotor strip | 21 | 64.4% | **73.3%** | 66.7% |
| Neurosity Crown | 8 | 71.1% | **71.1%** | 71.1% |
| OpenBCI Ganglion @ C3/C4/CP3/CP4 | 4 | 66.7% | **71.1%** | 68.9% |
| g.tec Unicorn | 8 | 64.4% | **68.9%** | 62.2% |
| All 64 channels | 64 | 67.4% | 65.1% | 67.4% |
| C3 / Cz / C4 only | 3 | 55.6% | 57.8% | 33.3% |
| Muse 2 † | 2 | 46.7% | 37.8% | 40.0% |

† Muse's TP9/TP10 have no counterpart in the PhysioNet montage, so only AF7/AF8
survive the intersection — the simulation runs on 2 channels rather than 4. The
point stands regardless: none of Muse's electrodes are over motor cortex, and
the result is at or below chance.

Two things this table says that are worth reading twice. **More channels is not
better** — 64 channels underperforms 21, because CSP is estimating a 64×64
covariance from 45 trials. And **four well-placed electrodes beat sixty-four
badly-chosen ones**: a $625 Ganglion with its electrodes at C3/C4/CP3/CP4 scores
the same as the $1,499 Crown and better than the full cap. Placement dominates.

### Method comparison (10 subjects, sensorimotor montage, 438 trials)

`python -m src.train --subjects 1 2 3 4 5 6 7 8 9 10 --cross-subject`

| Method | Subject-dependent | Cross-subject (LOSO) |
|---|---|---|
| Band power + LDA | 65.3% | **61.6%** |
| **CSP + LDA** | **66.7%** | 53.2% |
| CSP + SVM | 62.8% | 50.7% |
| EEGNet | 58.7% | 52.7% |

Four things in this table are worth stating plainly:

- **CSP + LDA wins subject-dependent, and loses badly cross-subject** (66.7% →
  53.2%, kappa 0.33 → 0.07). That is CSP working as designed: it fits spatial
  filters to one person's anatomy, which is exactly what does not transfer.
  Band power, being a cruder and more generic feature, degrades far less and
  becomes the best cross-subject method.
- **EEGNet underperforms the classical pipeline here**, at 58.7%. With ~44
  trials per subject it has nothing to learn from; CSP's hand-built inductive
  bias is worth more than learned features at this data scale. Reproducing the
  paper's numbers needs hundreds of trials per subject.
- **The RBF SVM does not beat LDA.** The class boundary in CSP log-variance
  space is already linear; the extra capacity only overfits.
- **Between-subject spread is enormous** — per-subject accuracy ranges from
  42% to 98% for CSP+LDA. That range is the single most important number on
  this page for anyone planning to test friends: some people decode almost
  perfectly and some sit at chance, and that is a property of the subject, not
  of the code.

### Figures

Generated into `results/figures/` by any training run:

- `accuracy_comparison.png` — every method against the chance line
- `confusion_<model>.png` — per-class precision/recall
- `erd_topomap.png` — **the one that reads as neuroscience.** Mu-band power
  change for left vs right imagery. Left-hand imagery should desynchronise the
  *right* hemisphere and vice versa; if the blobs are symmetric or frontal, the
  classifier is reading eyes or jaw, not motor cortex.

---

## Expectations, stated up front

| Setting | Typical accuracy |
|---|---|
| Research gel cap, trained subject | 80–95% |
| Consumer dry headband, after calibration | **60–75%** |
| Cross-subject, no calibration | 50–65% |

Two things that are results rather than bugs:

- **Cross-subject accuracy drops a lot.** Sulcal anatomy, skull thickness and
  imagery strategy differ between people, so spatial filters optimal for one are
  wrong for another. A PhysioNet-trained model will not decode your headband —
  hence the calibration step in the live workflow.
- **15–30% of healthy people cannot produce a decodable motor-imagery ERD.**
  Well-replicated, not a personal failing. Test several people.

---

## Repo structure

```
src/
├── app.py             # the local web app — `python -m src.app`
├── config.py          # sampling rates, bands, epoch windows, montages
├── preprocess.py      # loading, filtering, referencing, epoching (both sources)
├── features.py        # band power, CSP, wavelet energy, ERD
├── models.py          # LDA / SVM pipelines + EEGNet
├── train.py           # training & evaluation entry point
├── evaluate.py        # metrics, CV schemes, figures
├── acquire.py         # cued calibration recorder (Graz paradigm)
├── ingest.py          # import friends' recordings; CSV/EDF/zip → native format
├── realtime.py        # sliding-window live decoder + model persistence
├── cli.py             # app / devices / check / record / import / train / live
└── devices/
    ├── base.py             # hardware-agnostic EEGSource interface
    ├── channels.py         # 10-05 name harmonisation, montage assessment
    ├── registry.py         # headset catalogue + suitability ratings
    └── brainflow_device.py # BrainFlow-backed live source
docs/
├── HARDWARE.md        # buying guide (incl. budget/DIY), setup, self-deception checks
└── COLLECTING_DATA.md # opening the app, group workflow, privacy
tests/                 # 49 tests, focused on the hardware boundary
results/figures/
```

## Further reading

- [`docs/HARDWARE.md`](docs/HARDWARE.md) — device selection, electrode contact,
  session protocol, artifact controls, safety
- Lawhern et al. (2018), *EEGNet* · Ramoser et al. (2000), *CSP*
- [PhysioNet EEGMMIDB](https://physionet.org/content/eegmmidb/1.0.0/) ·
  [BrainFlow](https://brainflow.readthedocs.io/)
