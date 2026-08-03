# NeuroDecode app — architecture & extension guide

The app is one local Python server that serves both the decoder API and the web
UI, so there is no build step and no second process to manage.

```bash
python run_app.py          # from the repo root -> http://127.0.0.1:8000
```

## Layout

```
run_app.py              one-command launcher (opens your browser)
backend/
  main.py               FastAPI HTTP layer: routes, schemas, static mounting
  core.py               decoder registry + decode logic (the extensible core)
  static/
    index.html          UI markup / panels
    styles.css          design tokens (CSS variables) + component styles
    app.js              api / state / render modules
src/                    the actual science: preprocessing, features, models
```

The rule of thumb: **`src/` owns the science, `core.py` owns what the app can do,
`main.py` owns HTTP, `static/` owns presentation.** Most new features touch only
one of these.

## API

| Method | Route | Purpose |
|---|---|---|
| GET | `/api/health` | liveness + registered decoders |
| GET | `/api/decoders` | decoder metadata (drives the UI's model picker) |
| GET | `/api/subjects` | demo subject ids |
| POST | `/api/decode` | calibrate a decoder, decode trials |

Interactive docs (auto-generated, try requests in the browser): `/docs`.

`POST /api/decode` takes multipart form data:
- `decoder` — a key from `/api/decoders` (default `csp_lda`)
- **either** `subject` (demo subject id) **or** `files` (one or more `.edf` uploads)

```bash
# demo subject
curl -X POST localhost:8000/api/decode -F decoder=csp_lda -F subject=7

# your own recording
curl -X POST localhost:8000/api/decode -F decoder=eegnet -F files=@my_session.edf
```

## How to extend it

**Add a decoder** — one entry in `DECODERS` in `core.py`:
```python
"my_model": Decoder(
    "my_model", "My model", "What it does.",
    needs_epochs=True, factory=lambda sfreq: make_my_model(),
),
```
It appears in the UI automatically — no frontend edits. Any scikit-learn-style
estimator with `fit`/`predict_proba` works (EEGNet is wrapped this way already).

**Add a result panel** — return a new field from `core.decode()`, then add a
render function in `app.js`:
```js
render.myPanel = (res) => { /* build DOM from res.my_field */ };
// then call it inside renderResults()
```

**Add an endpoint** — a new function in `main.py` delegating to `core.py`
(keep computation out of the HTTP layer so it stays testable).

**Restyle** — edit the CSS variables at the top of `styles.css`; colours,
radius and shadow flow from there. Light mode is handled by the
`prefers-color-scheme` block.

## Planned extension points (for the hardware phase)

The pieces below are where physical-EEG support will slot in:

- **New data source.** `core.epochs_for_uploaded_files()` is the template: any
  function that returns an `mne.Epochs` can back a new source. A live-stream
  source (BrainFlow/OpenBCI) would add `epochs_from_stream()` alongside it.
- **Non-EEGMMIDB montages.** `src/preprocess.py` currently assumes the PhysioNet
  channel naming and T1/T2 annotations. Recordings from your own headset will
  need a small adapter mapping your channel names to the 10-20 montage.
- **Inference-only mode.** `EEGNetClassifier.save()/load()` already exist, so a
  `POST /api/predict` that loads a saved `.pt` and labels *unlabelled* trials is
  a small addition — that is the endpoint a real-time BCI demo would call.

## Notes

- The server binds to `127.0.0.1` — it is not exposed to your network.
- First run of a demo subject downloads data via MNE (cached in `~/mne_data`).
- EEGNet trains a CNN per cross-validation fold; expect ~1 minute on CPU. The
  classical decoders return in a few seconds.
