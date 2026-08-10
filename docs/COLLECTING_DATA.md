# Running the app, and collecting data with friends

Two things this document covers: how to open the app, and how a group of people
pools recordings into one dataset.

---

## 1. Opening the app

```bash
pip install -r requirements.txt
python -m src.app
```

That's it — it opens `http://localhost:8000` in your browser. No account, no
server, no internet. It runs on Python's standard library, so there is nothing
extra to install beyond the requirements file.

Equivalent: `python -m src.cli app`. Everything the app does is also available
as CLI commands if you prefer them, and the two share the same code.

The app has five tabs, in the order you should use them:

| Tab | What it does |
|---|---|
| **1 · Hardware** | Compares headsets, and runs a per-electrode signal check on the one you have |
| **2 · Record** | Runs a cued session with full-screen arrows |
| **3 · Data** | Upload recordings from friends; see what the group has collected |
| **4 · Train** | Train on everything collected, per person and across people |
| **5 · Live** | Real-time left/right decoding |

### Why the cue display is in a browser

The full-screen arrow is not decoration. Cue presentation quality directly
determines data quality — a subject reading a scrolling terminal has their
attention on parsing text rather than on their hand, and the resulting ERD is
weaker. A large arrow on a dark background against a fixation cross is the
standard Graz-paradigm stimulus, and it is what the app draws.

Timing stays in Python. The browser polls for the current cue phase and renders
it; it never decides *when* a cue happens. Cue onsets are stamped against the
device's own sample clock, because a JavaScript timer drifts against the
amplifier's crystal — over a 15-minute session, a 0.5% difference is 4.5
seconds, longer than a whole trial, and every label would be wrong.

### Sharing the app on your Wi-Fi

By default the app binds to loopback, so only your machine can reach it. You
*can* expose it:

```bash
python -m src.app --host 0.0.0.0
```

Understand what that does before you run it: anyone on the same network gets
unauthenticated access to your EEG recordings and to the connected hardware,
for as long as it runs. On university Wi-Fi, don't. Passing a laptop around, or
having each person run the app locally and send you a zip, is both easier and
safer.

---

## 2. Collecting data from several people

### The important framing: buy one device, share it

You do **not** need one headband each. EEG calibration is ~20 minutes per
person, so one device serves a group of five in an afternoon. This single fact
changes the budget more than any hardware choice: a $625 Ganglion shared by
five people is $125 each, and buying five cheap unsuitable headbands is worse
than sharing one good one in every respect.

Practical notes for passing a headband around:

- **Wipe electrodes with an alcohol wipe between people.** Non-negotiable.
- **Re-run the signal check for every person.** Head sizes differ; a band that
  fit the last person may not be contacting this one. This is the single most
  common cause of one person's data being unusable.
- **Log who is who.** The app's "Whose head is this?" field becomes the subject
  ID that cross-subject evaluation groups on. `alice` and `Alice` are two
  people as far as the code is concerned.
- **Budget 30 minutes per person**, not 20 — fitting and checking takes longer
  than recording.

### Three ways to get data in

**A. Everyone records on their own laptop, sends you a zip.** The best option
if people have their own time with the device.

```bash
# on their machine, after recording
python -m src.cli export --out my_recordings.zip

# on yours
python -m src.cli import my_recordings.zip
```

Or drag the zip onto the app's **Data** tab. Same thing.

**B. Everyone records on one machine.** Simplest. Just change the subject name
between people; sessions accumulate in `data/recordings/` automatically.

**C. Somebody recorded with other software.** OpenBCI's own GUI, a DIY rig, a
lab system — anything that produced a CSV or EDF.

```bash
python -m src.cli import their_recording.csv \
  --subject dave \
  --channels C3,C4,CP3,CP4 \
  --labels their_cues.csv
```

Two things are required here and both are easy to get wrong:

- **`--channels` is the position on the head, in the file's column order.** The
  file says "EXG Channel 0"; only the person who placed the electrodes knows
  that was C3. Getting the order wrong trains a model on permuted electrodes,
  which still produces a plausible accuracy number and is undetectable
  afterwards.
- **A cue-times file is mandatory.** A signal file alone has no idea which
  trials were left and which were right. Format is a two-column CSV:

  ```csv
  onset_s,label
  5.0,left
  11.0,right
  17.0,left
  ```

  `onset_s` is seconds from the start of the recording — not clock time, which
  never matches once you account for the gap between starting the software and
  starting the stream. `left`/`right`, `L`/`R`, and `0`/`1` are all accepted.

If your friend's software can't export cue times, they have to be recorded
some other way, and the honest answer is that it is easier to re-record using
this app than to reconstruct them.

### Checking what you have

```bash
python -m src.cli list
```

```
    person      task  trials     montage  electrodes
------------------------------------------------------------------
     alice   imagery  60 (30L/30R)   good  C3, C4, CP3, CP4
       bob   imagery  58 (30L/28R)   good  C3, C4, CP3, CP4
     carol  executed  60 (30L/30R)   good  C3, C4, CP3, CP4

3 people, 178 trials: alice, bob, carol
```

Watch two columns. **Montage** must not say `unsuitable` — if it does, that
person's electrodes were nowhere near motor cortex and their data cannot
contribute. **Trials** should be near-balanced; a big L/R imbalance means
artifact rejection hit one class harder, which inflates accuracy.

### Then train

```bash
python -m src.cli train --source recordings --cross-subject --save models/group.joblib
```

With two or more people this also runs leave-one-person-out: train on everyone
else, test on the held-out person. **Expect that to be much worse** — on
research data the gap is 66.7% → 53.2%. That gap is a result worth writing up,
not a bug to fix. It is the reason each person needs their own calibration.

---

## 3. Privacy — worth deciding before you collect, not after

Raw EEG from your friends is personal biometric data, and they are giving it to
you on trust.

- Everything stays on your machine. The app has no network calls, no telemetry,
  no cloud. `data/` is gitignored — **keep it that way**; do not commit
  recordings to GitHub to "share them with the group", use the zip export.
- Tell people plainly what you are storing and what you'll do with it, and let
  them say no or withdraw later. Deleting a person's data is deleting their
  `.npz`/`.json` pair.
- Use first names or initials as subject IDs, not student IDs.
- EEG is not diagnostic, and this project cannot tell anyone anything about
  their health. If someone asks, that is the answer.

---

## 4. Quick reference

```bash
python -m src.app                       # open the app (everything below has a tab)

python -m src.cli devices               # which headset works for this task
python -m src.cli check   --device crown        # per-electrode signal quality
python -m src.cli record  --device crown --task executed --subject alice
python -m src.cli list                          # what the group has collected
python -m src.cli export  --out my_recordings.zip
python -m src.cli import  from_bob.zip
python -m src.cli import  raw.csv --subject dave --channels C3,C4,CP3,CP4
python -m src.cli train   --source recordings --cross-subject --save models/group.joblib
python -m src.cli live    --device crown --model models/group.joblib
```

See [`HARDWARE.md`](HARDWARE.md) for which device to buy and how to set it up.
