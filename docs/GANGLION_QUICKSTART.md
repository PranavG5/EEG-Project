# Ganglion quickstart — from box to working decoder

Follow these in order. Don't skip step 5; it is the one that saves you a wasted
afternoon.

---

## Step 0 — What to actually order

The $624.99 board is **not** everything you need. The box contains only the
board, a case, and a battery. Real shopping list:

| Item | Price | Why |
|---|---|---|
| Ganglion Board (4ch) | $624.99 | The amplifier |
| **Ganglion Dongle** | $19.99 | **Mandatory** unless you use `ganglion_native` (see below) |
| Gold Cup Electrodes (pack of 10) | $44.99 | You need 6. The pack covers spares |
| Ten20 Conductive Paste | $24.99 | Gets you through hair. Not optional |
| **Total** | **~$715** | ~$143 each split five ways |

Optional: a cheap cotton headband (~$5) to hold electrodes down, and medical
tape.

⚠️ **The Ganglion is currently sold out at OpenBCI.** Set the "notify me" alert
on the product page. If you don't want to wait, the PiEEG (~$350 + a Pi) or a
used Ganglion on eBay are the alternatives — check the dongle is included with
any used board, it's board-specific and annoying to source alone.

**Can you skip the $20 dongle?** Maybe. Your laptop's own Bluetooth can talk to
the Ganglion directly — use `--device ganglion_native` instead of
`--device ganglion` everywhere below, plus `--mac-address` (printed on the
board). It's flakier on some Bluetooth stacks. Try it; buy the dongle if it
misbehaves.

---

## Step 1 — Install the software (do this before the hardware arrives)

```bash
pip install -r requirements.txt
python -m src.app
```

Your browser opens at `http://localhost:8000`. Click through the tabs. On the
**Hardware** tab, pick "BrainFlow synthetic board" and run a signal check — it
works with no hardware attached and proves your install is fine.

---

## Step 2 — Wire the electrodes to the board

Six gold cup electrodes. The Ganglion's pin header is labelled `+1-`, `+2-`,
`+3-`, `+4-`, `REF`, `D_G`.

| Electrode → | Board pin | Goes on the head at |
|---|---|---|
| 1 | `+1` (top row) | **C3** |
| 2 | `+2` | **C4** |
| 3 | `+3` | **CP3** |
| 4 | `+4` | **CP4** |
| 5 | `REF` | Left earlobe |
| 6 | `D_G` | Right earlobe |

**Flip switches SW1–SW4 to the DOWN position.** That ties each channel's `−`
input to `REF`, which is the standard EEG configuration. If you skip this, you
are recording the difference between two scalp electrodes instead of each
electrode against a common reference, and nothing downstream will make sense.

**The order matters.** The software assumes channel 1 = C3, 2 = C4, 3 = CP3,
4 = CP4. Wire it that way and you never have to type `--channels` at all. If you
wire it differently, you must say so every time:
`--channels CP3,C3,C4,CP4` (or whatever order you actually used).

---

## Step 3 — Find C3, C4, CP3, CP4 on the head

You need a tape measure. Two minutes, once you've done it twice.

1. **Find Cz.** Measure ear-to-ear, from the little flap in front of one ear,
   over the top of the head, to the same point on the other ear. Halfway is
   **Cz** — the top of the head. Mark it with a skin-safe pencil.
2. **C3 and C4.** On that same ear-to-ear line, C3 is **20% of the total
   distance to the left** of Cz, and C4 is 20% to the right. For a typical
   36 cm ear-to-ear measurement that's about **7 cm either side of Cz**.
3. **CP3 and CP4.** About **2 cm behind** C3 and C4 respectively (toward the
   back of the head).

That's it. C3 and C4 are the important ones — they sit over the hand area of
the motor cortex, on opposite hemispheres. Everything this project decodes is
the difference between them.

**Attaching through hair:** part the hair with the wooden end of a cotton bud
so you can see scalp. Fill the gold cup with Ten20 paste until it domes
slightly. Press it into the parting and hold for a couple of seconds. The paste
does the work — the cup does not need to touch skin, the paste needs to bridge
to it. A cotton headband over the top keeps everything from drifting.

---

## Step 4 — Plug in and check the signal

Plug the dongle into USB. Turn the board on.

You need the port name for the dongle:
- **Mac:** run `ls /dev/cu.usb*` → something like `/dev/cu.usbserial-DM01234`
- **Linux:** `ls /dev/ttyUSB*` → usually `/dev/ttyUSB0`
- **Windows:** Device Manager → Ports → something like `COM3`

In the app, **Hardware** tab: pick "OpenBCI Ganglion (4ch)", paste the port
into the serial port box, click **Run check**.

Or from the terminal:

```bash
python -m src.cli check --device ganglion --serial-port /dev/ttyUSB0
```

You want all four channels saying **ok**:

```
 channel   RMS (uV)   60Hz ratio   status
--------------------------------------------------
      C3       8.12         0.31   ok
      C4       9.44         0.28   ok
     CP3       7.03         0.44   ok
     CP4      11.20         0.52   ok
```

| If you see | It means | Do this |
|---|---|---|
| `FLAT` | Electrode isn't touching scalp | More paste, part the hair again, press it in |
| `LINE NOISE` | Poor contact acting as an antenna | Re-seat that electrode; unplug the laptop charger |
| `NOISY` | Movement, or a floating electrode | Sit still; check the wire isn't dangling |

**Do not continue until all four say ok.** No classifier recovers from an
electrode that wasn't touching skin.

---

## Step 5 — Prove it works: record REAL movement first

This is the step people skip and regret. Actually squeezing your fist produces
a much stronger brain signal than imagining it. If you can't decode real
movement, the problem is your hardware — not your imagination — and you've
found that out in 10 minutes instead of three days.

App → **Record** tab. Name: your name. Task: **executed**. Trials per side: 20.
Start.

Or:

```bash
python -m src.cli record --device ganglion --serial-port /dev/ttyUSB0 \
  --task executed --subject pranav --trials-per-class 20
```

The screen goes black with a `+`, then a big **←** or **→**. When you see the
arrow, **actually squeeze that fist** for the four seconds it's on screen. Then
rest. ~12 minutes.

Then train on it:

```bash
python -m src.cli train --source recordings --save models/me_executed.joblib
```

**You want 75%+ here.** If you get it, your hardware and electrode placement are
good and you can move on. If you're near 50%, go back to step 4 — something is
wrong with contact or placement, and imagined movement will only be harder.

---

## Step 6 — Now record imagined movement

Same thing, task **imagery**, 30 trials per side (~20 minutes).

```bash
python -m src.cli record --device ganglion --serial-port /dev/ttyUSB0 \
  --task imagery --subject pranav --trials-per-class 30
```

**Tell the person wearing it, in these words:** *feel* the movement from the
inside — the tension in your forearm, your fingers closing, the weight of it.
Don't picture a hand from outside; that uses the wrong part of the brain and
decodes badly. Pick one movement (squeezing a fist works well) and use the same
one every single trial. Keep still, loose jaw, try not to blink while the arrow
is up.

---

## Step 7 — Train

```bash
python -m src.cli train --source recordings --save models/me.joblib
```

```
csp_lda   within-subject 5-fold acc=0.712  kappa=0.42  n=58 trials x 4 ch
```

**60–75% is a good result** for imagined movement on a dry consumer setup. 50%
is chance. Don't be disappointed by 68% — that is roughly what the literature
reports for untrained subjects, and your figure is honest because the code
refits everything inside each cross-validation fold.

Check `results/figures/erd_topomap.png`. Left-hand imagery should show a blue
patch over the **right** side of the head. If it does, you're reading motor
cortex.

---

## Step 8 — Live decoding

```bash
python -m src.cli live --device ganglion --serial-port /dev/ttyUSB0 --model models/me.joblib
```

```
LEFT [·············●··············] RIGHT   RIGHT  0.78
```

Or the app's **Live** tab. Expect 1–2 seconds of lag — that's inherent to motor
imagery, not something to tune away. It says `uncertain` rather than guessing
when it isn't confident.

---

## Step 9 — Add your friends

One board serves everyone; you don't need more hardware. ~30 minutes per person
including fitting.

- **Wipe the electrodes with an alcohol wipe between people.**
- **Re-run step 4 for every person.** Head sizes differ. This is the single most
  common reason one person's data is unusable.
- Use a different `--subject` name for each person. `alice` and `Alice` count as
  two different people.

If a friend records on their own laptop:

```bash
python -m src.cli export --out my_recordings.zip     # them
python -m src.cli import from_bob.zip                # you
```

Or drag the zip onto the app's **Data** tab. Then:

```bash
python -m src.cli list                                        # who contributed what
python -m src.cli train --source recordings --cross-subject   # train on everyone
```

`--cross-subject` also tests training on everyone *except* one person and
decoding that person. It will score much worse — around 53% versus 67%. That
gap is a real finding worth writing up, not a bug. It's why each person needs
their own calibration.

**Expect one person in five to sit near chance no matter what.** 15–30% of
people don't produce a decodable motor-imagery signal. Not your code's fault,
not theirs.

---

## The whole thing, condensed

```bash
pip install -r requirements.txt
python -m src.app                                             # or use the commands below

python -m src.cli check  --device ganglion --serial-port /dev/ttyUSB0
python -m src.cli record --device ganglion --serial-port /dev/ttyUSB0 --task executed --subject me
python -m src.cli train  --source recordings --save models/me_executed.joblib   # want 75%+
python -m src.cli record --device ganglion --serial-port /dev/ttyUSB0 --task imagery --subject me
python -m src.cli train  --source recordings --save models/me.joblib            # want 60-75%
python -m src.cli live   --device ganglion --serial-port /dev/ttyUSB0 --model models/me.joblib
```

Notice there's no `--channels` anywhere: wire channel 1→C3, 2→C4, 3→CP3, 4→CP4
and the software already knows.

See [`HARDWARE.md`](HARDWARE.md) for the reasoning, and
[`COLLECTING_DATA.md`](COLLECTING_DATA.md) for the group workflow in detail.
