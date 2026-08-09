# Buying and running a headband for arm-imagery decoding

Everything in this document exists to answer one question: **what has to be
true for a physical headband to tell which arm you were thinking of moving?**

The short version: the decision that matters is not the classifier, not the
sampling rate, and not the channel count. It is **where the electrodes sit on
your head.** Get that wrong and no amount of modelling recovers it. Get it
right and a fairly simple pipeline works.

---

## 1. The physics you are buying into

When you imagine squeezing your left fist, the hand area of your **right**
primary motor cortex partially activates — the motor system is contralateral.
Active cortex stops idling, and its idling rhythm (the **mu rhythm**, 8–13 Hz,
plus **beta**, 13–30 Hz) drops in power. That power drop is called
**event-related desynchronisation (ERD)**.

So the measurable signature of "I am thinking about my left hand" is:

> *less* mu/beta power over the **right** sensorimotor cortex than over the left.

The hand area sits near electrode positions **C3** (left hemisphere → right
hand) and **C4** (right hemisphere → left hand) in the 10–20 system — roughly
on the line between your ears, about 6 cm either side of the crown of your
head. **Under hair.**

Three consequences fall straight out of this, and they drive every
recommendation below:

1. **You need electrodes near C3 and C4.** The signal is a left-right
   difference across the top of the head. Electrodes on the forehead or behind
   the ears measure it only as heavily smeared volume conduction, if at all.
2. **You need to get through hair.** The best position for this task is the
   worst position for dry-electrode contact. This is the single biggest
   practical difficulty and the reason most "easy" consumer headbands measure
   the forehead instead.
3. **It is a *difference*, so you need both sides.** One hemisphere's worth of
   electrodes cannot distinguish "left hand" from "generally concentrating".

---

## 2. Device recommendations

Run `python -m src.cli devices` for the live version of this table — it derives
the ratings from each device's actual electrode list rather than from prose.

| Device | Channels over motor strip | Rating | Price | Verdict |
|---|---|---|---|---|
| **Neurosity Crown** | C3, C4, CP3, CP4 | **good** | ~$1,499 | **Best headband-shaped option.** Laid out for exactly this task. |
| **g.tec Unicorn Hybrid Black** | C3, Cz, C4 | **good** | ~$1,000–1,300 | Textbook BCI montage, research-grade, well documented. |
| **OpenBCI Cyton (8ch)** | you choose | **good** | ~$1,249 + $350–500 headwear | Highest ceiling — put all 8 electrodes on the motor strip. Most work. |
| **OpenBCI Ganglion (4ch)** | you choose | **good** | ~$625 + headwear | Cheapest way to get electrodes at C3/C4. Bare minimum channel count. |
| **Muse 2 / Muse S** | **none** | **unsuitable** | ~$250–400 | Cannot do this task. See below. |
| **EMOTIV EPOC X** | none (no central sites) | marginal | ~$850 | Also not supported by BrainFlow; proprietary SDK. Avoid for this. |

### Why not Muse — the trap worth naming explicitly

Muse is the headband most people buy first, it is a genuinely good product, and
it is the **wrong** product here. Its four electrodes are TP9, AF7, AF8, TP10:
two on the forehead, two behind the ears. Nothing within ~7 cm of the hand area
of motor cortex.

The failure mode is not "lower accuracy". It is worse than that: **you will
probably get above-chance accuracy anyway, and it will not be brain-derived.**
Forehead electrodes see eye movements beautifully. If you glance left when
cued left, or tense your jaw slightly differently for the two conditions, a
classifier will happily learn that at 80%+ and you will believe you have a
motor-imagery BCI. This is the most common self-deception in amateur BCI work.

The project guards against it: `python -m src.cli record --device muse2`
refuses to run without `--force`.

### If you want my single recommendation

**Neurosity Crown**, if the budget tolerates it — it is the only device that is
both an actual headband (easy to put on, no gel, friends can try it in turn)
and correctly positioned (C3, C4, CP3, CP4). It is the shortest path from box
to working decoder.

⚠️ **Check stock before planning around it.** As of this writing Neurosity's own
store shows limited Crown inventory ("few Crowns left"). Prices and availability
in this document were checked in August 2026 and will drift — confirm both
before ordering, and treat OpenBCI as the fallback, since its boards are
produced continuously and are not a single-product company's remaining stock.

**OpenBCI Ganglion or Cyton** if you would rather spend time than money, if
Crown stock has run out, or if learning the hardware side is part of the point. You place the electrodes
yourself, which means you can build the ideal montage — and also means you can
get it wrong, so tell the software where you actually put them:

```bash
python -m src.cli check --device cyton --serial-port /dev/ttyUSB0 \
  --channels FC3,FC4,C5,C3,C4,C6,CP3,CP4
```

---

## 3. What accuracy to actually expect

Published two-class motor-imagery numbers, so you can calibrate expectations
before you are disappointed by a real one:

| Setting | Typical accuracy |
|---|---|
| Research gel cap, trained subject, within-session | 80–95% |
| Research cap, untrained subject, within-session | 65–80% |
| **Consumer dry headband, untrained, after calibration** | **60–75%** |
| Cross-subject, no calibration | 50–65% |
| Any setup, first session, first ten minutes | ~50% (chance) |

Measured in this repo on PhysioNet subject 1, restricted to the Neurosity
Crown's 8 electrodes: **71.1%** (CSP+LDA, 5-fold). That is a realistic ceiling
estimate for that device on a average-ish subject with clean research-grade
signal — real dry electrodes on your own head will land at or below it.

**Chance is 50%, and two-class accuracy near chance is very easy to over-read.**
Always report Cohen's kappa alongside accuracy (the pipeline does), and always
run the label-shuffled control described in §6.

### BCI illiteracy is real

Roughly **15–30% of healthy people cannot produce a decodable motor-imagery
ERD**, at least not without training. This is a well-replicated finding, not a
personal failing and not a bug in your code. If you and your friends test five
people, expect one to be near chance no matter what you do.

Plan for it: test several people, and do not spend a week debugging a pipeline
that is working fine on a subject who simply does not modulate mu strongly.

---

## 4. The thing that surprises everyone: you must calibrate

**A model trained on PhysioNet will not decode your headband.** This is not a
code problem and cannot be fixed by a better architecture.

The mapping from cortical current to recorded voltage differs between a
64-electrode gel cap on a stranger in 2009 and dry electrodes on your head
today: different electrode positions, skull thickness, sulcal anatomy,
reference scheme, amplifier, impedance, and mental strategy for "imagine your
left hand". Cross-subject accuracy is already only 50–65% *within* one dataset;
across hardware it degrades further.

The fix is 20 minutes of your own labelled data:

```bash
python -m src.cli record --device crown --task imagery --trials-per-class 30
python -m src.cli train  --source recordings --save models/me.joblib
python -m src.cli live   --device crown --model models/me.joblib
```

The PhysioNet path in this repo is for **benchmarking and pipeline
development** — including estimating a device's ceiling before you buy it. It
is not the deployment path.

---

## 5. Setup, step by step

### First session, in order. Do not skip steps.

1. **Signal check before anything else.**
   ```bash
   python -m src.cli check --device crown --seconds 20
   ```
   Every channel should read 2–15 µV RMS in the 8–30 Hz band with a 60 Hz ratio
   under ~2. A flat channel is not touching skin; a screaming one is floating or
   picking up muscle. Fix it now — no classifier recovers from an electrode that
   was not in contact.

2. **Eyes-open / eyes-closed test.** Close your eyes for 20 s. Occipital alpha
   (8–12 Hz) should roughly double. If you cannot see that, you are not
   recording EEG at all, and nothing further will work. This is the cheapest
   possible proof of life.

3. **Executed movement before imagined movement.**
   ```bash
   python -m src.cli record --device crown --task executed
   ```
   Actually squeeze the fist. Real movement produces ERD several times stronger
   than imagery. If you cannot decode *executed* left vs right, the problem is
   hardware or placement, not your mental technique — and you have separated
   those two failure modes cleanly.

4. **Then imagery.** Only once executed movement decodes above ~75%.

### Electrode contact

- **Hair is the enemy.** Part it and wiggle the electrode down to scalp. For
  gel/wet setups, a blunt syringe of conductive paste under each electrode.
- **Dry comb electrodes** (Crown, OpenBCI's) work through hair but need firm,
  even pressure. A slightly-too-loose headband is the most common cause of a
  session that "just didn't work".
- **Impedance** below ~20 kΩ for dry, ~5 kΩ for gel. If your device reports it,
  use it; if not, the 60 Hz ratio in `cli check` is a good proxy.
- **Reference and ground matter.** Usually mastoid or earlobe. A bad reference
  corrupts *every* channel simultaneously — if all channels look equally awful,
  suspect the reference before the electrodes.

### The recording environment

- Sit still. Relax the jaw, do not clench. Neck and jaw EMG lands squarely in
  the beta band.
- Feet flat, arms supported, hands still and symmetric.
- Blink freely during rest, try not to during the imagery window.
- Laptop on **battery**, not charger, if 60 Hz noise is a problem.
- Phone away from the amplifier; Bluetooth headsets and Wi-Fi hotspots nearby
  can inject dropouts.

### Mental technique — tell your subjects this

**Kinaesthetic, not visual.** Do not picture a hand from the outside. *Feel*
the movement from the inside: the tension in your forearm, the fingers closing,
the weight. Visual imagery activates visual cortex and decodes poorly;
kinaesthetic imagery is what produces sensorimotor ERD.

Pick one consistent movement — squeezing a fist works better than waving —
and use the same one every trial.

---

## 6. How not to fool yourself

The failure mode of this project is not low accuracy. It is **high accuracy for
the wrong reason.** Four controls, in increasing order of how much they will
save you:

1. **Label-shuffle control.** Randomly permute the labels and retrain. You must
   get ~50%. If shuffled labels still score 65%, you have a leak — almost always
   CSP or a scaler fitted before the train/test split. (This pipeline fits
   everything inside folds via `cross_val_predict`, specifically to avoid it.)

2. **Look at the ERD topomap.** `results/figures/erd_topomap.png`. Left-hand
   imagery should show a blue (negative) patch over the **right** hemisphere and
   vice versa. If your blobs are frontal, symmetric, or midline, you are decoding
   eyes, jaw, or attention — not motor cortex.

3. **Check the CSP patterns.** They should look like lateralised blobs near
   C3/C4. If the top filter loads on a frontal electrode, suspect EOG.

4. **Record a "do nothing differently" session.** Present the cues but have the
   subject ignore them completely. Accuracy must collapse to chance. If it does
   not, something in your setup is cue-locked — a screen flash, a sound, a
   timing artefact — and the classifier found it.

Beyond that, the usual traps:

- **Muscle (EMG) leakage.** If the subject subtly tenses the actual hand, you are
  decoding muscle, not imagery. Watch them. The 8–30 Hz bandpass helps but does
  not eliminate it.
- **Eye movement (EOG).** Looking toward the cued side is the classic
  confound. Use a central fixation cross and cue with a symbol, not a position.
- **Session drift.** Electrode impedance changes over 30+ minutes. A model
  trained on the first half may not work on the second — worth measuring, and a
  good reason to keep sessions under ~20 minutes.
- **Class imbalance from artifact rejection.** If rejection kills more of one
  class, accuracy is inflated. The trainer prints per-class counts; check them.

---

## 7. Live decoding

```bash
python -m src.cli live --device crown --model models/me.joblib
```

What happens under the hood, and the constraints that shaped it:

- A **3-second sliding window**, stepped every **250 ms** → 4 predictions/sec.
- The window is preprocessed by *exactly* the same code as training data. Any
  mismatch between training and live preprocessing silently destroys accuracy.
- An extra **1 second of warm-up** is kept in the buffer and then discarded,
  because an FIR bandpass needs history; without it the first half-second of
  every window is filter transient, not signal.
- Probabilities are **averaged over the last 4 windows** and the decoder
  **abstains below 65% confidence**, reporting `uncertain`. A BCI that says "I
  don't know" is far more usable than one that flickers left/right at 4 Hz.

**Expect ~1–2 seconds of latency.** ERD builds over roughly a second after
imagery starts, and the smoothing adds another. This is inherent to
non-invasive motor imagery — it is not a tuning problem. If you want something
that feels instant, motor imagery is the wrong paradigm (SSVEP or P300 are
faster but require visual stimulation).

---

## 8. Budget

| Item | Cost |
|---|---|
| Neurosity Crown (everything included) | ~$1,499 |
| **or** g.tec Unicorn Hybrid Black | ~$1,000–1,300 |
| **or** OpenBCI Cyton + Ultracortex Mark IV | ~$1,750 |
| **or** OpenBCI Ganglion + headband kit + dry combs | ~$1,025 |
| Conductive paste / gel (wet setups) | ~$25 |
| Spare dry comb electrodes | ~$50 |

Everything on the software side is free and already in this repo.

---

## 9. Safety and scope

- These are **research and hobbyist devices, not medical equipment.** Nothing
  here diagnoses anything, and no output should inform a health decision.
- **Never connect an electrode-wearing person to mains-powered equipment that
  is not medically isolated.** Use battery power for the amplifier; run the
  laptop on battery during recording. Commercial EEG boards are isolated by
  design — do not improvise around that.
- Stop if anyone reports skin irritation from electrodes or paste. Clean
  electrodes between people.
- If you record friends, tell them plainly what you are storing and let them say
  no. Raw EEG is personal data. Recordings land in `data/recordings/`, which is
  gitignored — keep it that way.

---

## 10. Sources

- [PhysioNet EEG Motor Movement/Imagery Dataset](https://physionet.org/content/eegmmidb/1.0.0/)
- [BrainFlow supported boards](https://brainflow.readthedocs.io/en/stable/SupportedBoards.html) — the abstraction layer this project uses
- [Neurosity Crown tech specs](https://neurosity.co/tech-specs)
- [OpenBCI shop](https://shop.openbci.com/collections/openbci-products) and [motor imagery docs](https://docs.openbci.com/Deprecated/MotorImagery/)
- [g.tec Unicorn Hybrid Black](https://www.gtec.at/product/unicorn-hybrid-black/)
- [Analysis of Minimal Channel EEG for Wearable BCI](https://www.mdpi.com/2079-9292/13/3/565)
- Lawhern et al. (2018), *EEGNet: a compact convolutional network for EEG-based BCIs*
- Ramoser et al. (2000), *Optimal spatial filtering of single trial EEG* — the CSP paper
