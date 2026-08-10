"""Local web app: the thing you actually open.

    python -m src.app          →  http://localhost:8000

Runs entirely on your own machine, on Python's standard library — no Flask, no
Node, no account, nothing leaves the laptop. That last point is not incidental:
raw EEG from your friends is personal biometric data, and the right default for
a student project collecting it is that it never touches a server.

Why a browser rather than the terminal, given the CLI already does all of this:

* **The cue display.** A full-screen arrow on a dark background is a far better
  stimulus than scrolling terminal text, and cue presentation quality directly
  determines data quality — a subject squinting at a log line is a subject
  whose attention is on the screen instead of on their hand.
* **Passing the headband around.** Your friends should not need to learn a
  command line to contribute half an hour of data.
* **Uploads.** Drag a file onto a page, versus explaining file paths over text.

Timing authority stays in Python. The browser polls for the current cue phase
and renders it; it never decides when a cue happens, because cue onsets are
stamped against the device's own sample clock and a JavaScript timer would
drift against it.
"""

from __future__ import annotations

import json
import threading
import traceback
import webbrowser
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from .config import MODELS_DIR, RECORDINGS_DIR
from .devices.channels import assess_montage
from .devices.registry import DEVICE_PROFILES


# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------


@dataclass
class RecorderState:
    """What the browser polls to know what to draw.

    Deliberately tiny and lock-protected: it is written by the recording thread
    and read by every HTTP request, several times a second.
    """

    running: bool = False
    phase: str = "idle"       # idle | fixation | cue | imagery | rest | done | error
    label: str = ""           # left | right | ""
    remaining: float = 0.0
    trial: int = 0
    total: int = 0
    message: str = ""
    saved_path: str = ""
    lock: threading.Lock = field(default_factory=threading.Lock)

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "running": self.running,
                "phase": self.phase,
                "label": self.label,
                "remaining": round(self.remaining, 2),
                "trial": self.trial,
                "total": self.total,
                "message": self.message,
                "saved_path": self.saved_path,
            }

    def update(self, **kwargs: object) -> None:
        with self.lock:
            for key, value in kwargs.items():
                setattr(self, key, value)


STATE = RecorderState()
LIVE_STATE: dict = {"running": False, "label": "", "confidence": 0.0, "probabilities": {}}
LIVE_LOCK = threading.Lock()


# --------------------------------------------------------------------------
# Actions
# --------------------------------------------------------------------------


def _open_source(payload: dict):
    from .devices import open_source

    channels = payload.get("channels") or ""
    return open_source(
        payload["device"],
        serial_port=payload.get("serial_port", ""),
        mac_address=payload.get("mac_address", ""),
        ip_address=payload.get("ip_address", ""),
        channel_override=[c.strip() for c in channels.split(",")] if channels else None,
    )


def action_devices(_: dict) -> dict:
    """Catalogue with suitability ratings, for the device picker."""
    out = []
    for key, profile in DEVICE_PROFILES.items():
        assessment = (
            assess_montage(list(profile.default_channels))
            if profile.default_channels
            else None
        )
        out.append(
            {
                "key": key,
                "name": profile.display_name,
                "price": profile.approx_price_usd,
                "connection": profile.connection,
                "sfreq": profile.sfreq,
                "channels": list(profile.default_channels),
                "repositionable": profile.electrodes_repositionable,
                "rating": profile.montage_rating if profile.default_channels else "unknown",
                "note": profile.notes,
                "core_motor": list(assessment.core_motor) if assessment else [],
            }
        )
    order = {"good": 0, "marginal": 1, "unknown": 2, "unsuitable": 3}
    out.sort(key=lambda d: (order.get(d["rating"], 4), d["name"]))
    return {"devices": out}


def action_check(payload: dict) -> dict:
    """Stream briefly and report per-electrode signal health."""
    import time

    from scipy.signal import welch

    seconds = float(payload.get("seconds", 12))
    source = _open_source(payload)
    chunks = []
    with source:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            time.sleep(0.2)
            chunk = source.poll()
            if chunk.size:
                chunks.append(chunk)

    if not chunks:
        raise RuntimeError(
            "No data received. Check that the device is switched on, paired, and "
            "that the dongle is plugged into this machine."
        )

    data = np.concatenate(chunks, axis=1)
    sfreq = source.info.sfreq
    nperseg = min(data.shape[1], int(sfreq * 2))
    freqs, psd = welch(data, fs=sfreq, nperseg=nperseg, axis=-1)
    band = (freqs >= 8) & (freqs <= 30)
    line = (freqs >= 58) & (freqs <= 62)
    df = freqs[1] - freqs[0]

    rows = []
    for i, ch in enumerate(source.info.ch_names):
        rms = float(np.sqrt(psd[i, band].sum() * df)) * 1e6
        ratio = float(psd[i, line].mean()) / (float(psd[i, band].mean()) + 1e-30)
        if rms < 0.5:
            status = "flat"
        elif rms > 50:
            status = "noisy"
        elif ratio > 2.0:
            status = "line"
        else:
            status = "ok"
        rows.append(
            {"channel": ch, "rms_uv": round(rms, 2), "line_ratio": round(ratio, 2),
             "status": status}
        )

    return {
        "channels": rows,
        "seconds": round(data.shape[1] / sfreq, 1),
        "montage": assess_montage(source.info.ch_names).rating,
        "all_ok": all(r["status"] == "ok" for r in rows),
    }


def _record_worker(payload: dict) -> None:
    """Run a calibration session on a background thread, publishing cue state."""
    from .acquire import CalibrationRecorder, CueProtocol, save_session

    try:
        protocol = CueProtocol(
            n_trials_per_class=int(payload.get("trials_per_class", 30)),
            imagery_s=float(payload.get("imagery_seconds", 4.0)),
            seed=int(payload.get("seed", 0)),
        )
        STATE.update(
            running=True, phase="fixation", label="", trial=0,
            total=protocol.n_trials, message="", saved_path="",
        )

        def on_cue(phase: str, label: str, remaining: float) -> None:
            if phase == "progress":
                done = int(str(label).split("/")[0]) if "/" in str(label) else 0
                STATE.update(trial=done)
                return
            STATE.update(phase=phase, label=label, remaining=remaining)

        source = _open_source(payload)
        with source:
            recorder = CalibrationRecorder(
                source,
                protocol,
                subject=payload.get("subject", "self").strip() or "self",
                device_key=payload["device"],
                task=payload.get("task", "imagery"),
                cue_callback=on_cue,
            )
            data, meta = recorder.run()

        path = save_session(data, meta, out_dir=RECORDINGS_DIR)
        STATE.update(
            running=False, phase="done", label="", remaining=0.0,
            saved_path=str(path),
            message=f"Saved {len(meta.trials)} trials to {path.name}",
        )
    except Exception as exc:  # surfaced in the browser, not swallowed
        STATE.update(
            running=False, phase="error", label="", remaining=0.0,
            message=f"{type(exc).__name__}: {exc}",
        )
        traceback.print_exc()


def action_record_start(payload: dict) -> dict:
    if STATE.snapshot()["running"]:
        raise RuntimeError("A recording is already running.")

    from .devices.registry import get_profile

    profile = get_profile(payload["device"])
    if profile.montage_rating == "unsuitable" and not payload.get("force"):
        raise RuntimeError(
            f"{profile.display_name} has no electrodes over the sensorimotor strip, "
            "so left-vs-right hand imagery is not observable with it. Recording "
            "would produce data that cannot work. Tick 'record anyway' to override."
        )

    threading.Thread(target=_record_worker, args=(payload,), daemon=True).start()
    return {"started": True}


def action_record_status(_: dict) -> dict:
    return STATE.snapshot()


def action_sessions(_: dict) -> dict:
    from .ingest import inventory

    sessions = inventory(RECORDINGS_DIR)
    subjects = sorted({s.get("subject", "?") for s in sessions if "error" not in s})
    total_trials = sum(s.get("n_trials", 0) for s in sessions if "error" not in s)
    return {"sessions": sessions, "subjects": subjects, "total_trials": total_trials}


def action_train(payload: dict) -> dict:
    """Train on everything in the recordings directory."""
    import io
    from contextlib import redirect_stdout

    from .train import build_parser, run_training

    models = payload.get("models") or ["bandpower_lda", "csp_lda", "csp_svm"]
    argv = ["--source", "recordings", "--folds", "5", "--models", *models]
    if payload.get("no_figures"):
        argv.append("--no-figures")
    if payload.get("cross_subject"):
        argv.append("--cross-subject")
    save_path = payload.get("save") or str(MODELS_DIR / "group.joblib")
    argv += ["--save", save_path]

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        run_training(build_parser().parse_args(argv))
    return {"log": buffer.getvalue(), "model": save_path}


def _live_worker(payload: dict) -> None:
    from .realtime import LiveDecoder, load_decoder

    try:
        model, ch_names, _ = load_decoder(Path(payload["model"]))
        source = _open_source(payload)
        decoder = LiveDecoder(
            source, model, ch_names,
            confidence_threshold=float(payload.get("threshold", 0.65)),
        )

        def publish(prediction) -> None:
            with LIVE_LOCK:
                LIVE_STATE.update(
                    running=True,
                    label=prediction.label,
                    confidence=round(prediction.confidence, 3),
                    probabilities={k: round(v, 3) for k, v in prediction.probabilities.items()},
                )

        decoder.run(duration_s=float(payload.get("seconds", 120)), callback=publish)
    except Exception as exc:
        with LIVE_LOCK:
            LIVE_STATE.update(running=False, label="error", error=str(exc))
        traceback.print_exc()
    finally:
        with LIVE_LOCK:
            LIVE_STATE["running"] = False


def action_live_start(payload: dict) -> dict:
    with LIVE_LOCK:
        if LIVE_STATE.get("running"):
            raise RuntimeError("Live decoding is already running.")
        LIVE_STATE.update(running=True, label="", confidence=0.0, probabilities={})
    threading.Thread(target=_live_worker, args=(payload,), daemon=True).start()
    return {"started": True}


def action_live_status(_: dict) -> dict:
    with LIVE_LOCK:
        return dict(LIVE_STATE)


def action_models(_: dict) -> dict:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return {"models": sorted(p.name for p in MODELS_DIR.glob("*.joblib"))}


ACTIONS = {
    "devices": action_devices,
    "check": action_check,
    "record/start": action_record_start,
    "record/status": action_record_status,
    "sessions": action_sessions,
    "train": action_train,
    "live/start": action_live_start,
    "live/status": action_live_status,
    "models": action_models,
}


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------


class Handler(BaseHTTPRequestHandler):
    server_version = "EEGMotorImagery/0.3"

    def log_message(self, fmt: str, *args: object) -> None:
        # The default logger prints every poll, and the browser polls at 10 Hz.
        if "/api/record/status" in str(args) or "/api/live/status" in str(args):
            return
        super().log_message(fmt, *args)

    # -- helpers ---------------------------------------------------------

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str) -> None:
        body = html.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # -- routes ----------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802
        route = urlparse(self.path).path
        if route in ("/", "/index.html"):
            self._send_html(PAGE)
            return
        if route.startswith("/api/"):
            name = route[len("/api/"):].strip("/")
            query = {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}
            self._dispatch(name, query)
            return
        self._send_json({"error": "not found"}, 404)

    def do_POST(self) -> None:  # noqa: N802
        route = urlparse(self.path).path
        if route == "/api/upload":
            self._handle_upload()
            return
        if not route.startswith("/api/"):
            self._send_json({"error": "not found"}, 404)
            return

        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            self._send_json({"error": "malformed JSON body"}, 400)
            return
        self._dispatch(route[len("/api/"):].strip("/"), payload)

    def _dispatch(self, name: str, payload: dict) -> None:
        action = ACTIONS.get(name)
        if action is None:
            self._send_json({"error": f"unknown action '{name}'"}, 404)
            return
        try:
            self._send_json(action(payload))
        except Exception as exc:
            traceback.print_exc()
            self._send_json({"error": f"{type(exc).__name__}: {exc}"}, 400)

    def _handle_upload(self) -> None:
        """Accept an uploaded recording and run it through the ingest funnel.

        Uses a minimal multipart parser rather than ``cgi.FieldStorage``, which
        was removed in Python 3.13. Only the filename and bytes are needed.
        """
        from .ingest import import_any

        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type or "boundary=" not in content_type:
            self._send_json({"error": "expected a multipart file upload"}, 400)
            return

        boundary = content_type.split("boundary=")[1].split(";")[0].strip().strip('"')
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            self._send_json({"error": "empty upload"}, 400)
            return
        body = self.rfile.read(length)

        delimiter = b"--" + boundary.encode()
        staging = RECORDINGS_DIR / "_incoming"
        staging.mkdir(parents=True, exist_ok=True)

        # Two passes. Everything is written to staging first so that a CSV and
        # its `*.labels.csv` sidecar uploaded together end up side by side —
        # importing each file as it arrives would fail on whichever came first.
        staged: list[Path] = []
        fields: dict[str, str] = {}

        for part in body.split(delimiter):
            if b"\r\n\r\n" not in part:
                continue
            headers_blob, content = part.split(b"\r\n\r\n", 1)
            headers_text = headers_blob.decode("utf-8", "replace")
            content = content.rstrip(b"\r\n-")

            if "filename=" in headers_text:
                filename = headers_text.split("filename=")[1].split("\r\n")[0].strip().strip('"')
                if not filename:
                    continue
                # Strip path components a browser or crafted request supplied.
                path = staging / Path(filename).name
                path.write_bytes(content)
                staged.append(path)
            elif 'name="' in headers_text:
                key = headers_text.split('name="')[1].split('"')[0]
                fields[key] = content.decode("utf-8", "replace").strip()

        if not staged:
            self._send_json({"error": "no files found in the upload"}, 400)
            return

        kwargs: dict = {}
        if fields.get("subject"):
            kwargs["subject"] = fields["subject"]
        if fields.get("channels"):
            kwargs["channels"] = [c.strip() for c in fields["channels"].split(",") if c.strip()]
        if fields.get("task"):
            kwargs["task"] = fields["task"]

        # Label sidecars are inputs to another file's import, not imports of
        # their own; `_find_sidecar` discovers them by name in the staging dir.
        signals = [p for p in staged if not p.name.lower().endswith(("labels.csv", "_labels.csv"))]
        results, errors = [], []
        for path in signals:
            try:
                imported = import_any(path, out_dir=RECORDINGS_DIR, **kwargs)
                results.extend(p.name for p in imported)
            except Exception as exc:
                errors.append(f"{path.name}: {exc}")

        for path in staged:
            path.unlink(missing_ok=True)
        self._send_json({"imported": results, "errors": errors})


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------

PAGE = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>EEG motor imagery</title>
<style>
  :root{--bg:#0e1116;--panel:#171c24;--line:#28303c;--text:#e6edf3;--dim:#8b98a8;
        --good:#3fb950;--warn:#d29922;--bad:#f85149;--accent:#58a6ff}
  *{box-sizing:border-box}
  body{margin:0;font:15px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;
       background:var(--bg);color:var(--text)}
  header{padding:20px 24px;border-bottom:1px solid var(--line)}
  h1{margin:0;font-size:18px;letter-spacing:.2px}
  header p{margin:6px 0 0;color:var(--dim);font-size:13px}
  nav{display:flex;gap:4px;padding:0 24px;border-bottom:1px solid var(--line);flex-wrap:wrap}
  nav button{background:none;border:0;border-bottom:2px solid transparent;color:var(--dim);
             padding:12px 14px;font:inherit;cursor:pointer}
  nav button.on{color:var(--text);border-bottom-color:var(--accent)}
  main{padding:24px;max-width:1000px}
  section{display:none} section.on{display:block}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:10px;
        padding:18px;margin-bottom:16px}
  .card h2{margin:0 0 4px;font-size:15px}
  .card p.hint{margin:0 0 14px;color:var(--dim);font-size:13px}
  label{display:block;margin:10px 0 4px;font-size:12px;color:var(--dim);
        text-transform:uppercase;letter-spacing:.04em}
  input,select{background:#0d1117;border:1px solid var(--line);color:var(--text);
               border-radius:6px;padding:8px 10px;font:inherit;width:100%}
  .row{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}
  button.go{background:var(--accent);color:#04101f;border:0;border-radius:6px;
            padding:10px 18px;font:inherit;font-weight:600;cursor:pointer;margin-top:16px}
  button.go:disabled{opacity:.45;cursor:not-allowed}
  button.ghost{background:none;border:1px solid var(--line);color:var(--text)}
  table{width:100%;border-collapse:collapse;font-size:13px;margin-top:10px}
  th,td{text-align:left;padding:7px 8px;border-bottom:1px solid var(--line)}
  th{color:var(--dim);font-weight:500;font-size:12px;text-transform:uppercase}
  .pill{display:inline-block;padding:2px 8px;border-radius:99px;font-size:11px;font-weight:600}
  .good{background:rgba(63,185,80,.15);color:var(--good)}
  .marginal{background:rgba(210,153,34,.15);color:var(--warn)}
  .unsuitable,.bad{background:rgba(248,81,73,.15);color:var(--bad)}
  pre{background:#0d1117;border:1px solid var(--line);border-radius:8px;padding:12px;
      overflow:auto;font-size:12px;max-height:420px;white-space:pre-wrap}
  #stage{position:fixed;inset:0;background:#05070a;display:none;place-items:center;z-index:50}
  #stage.on{display:grid}
  #arrow{font-size:22vw;line-height:1;font-weight:700}
  #word{font-size:4vw;color:var(--dim);margin-top:2vh;text-align:center}
  #progress{position:fixed;bottom:28px;left:0;right:0;text-align:center;color:var(--dim)}
  .drop{border:2px dashed var(--line);border-radius:10px;padding:34px;text-align:center;
        color:var(--dim);cursor:pointer}
  .drop.hot{border-color:var(--accent);color:var(--text)}
  .muted{color:var(--dim);font-size:13px}
  .err{color:var(--bad);font-size:13px;margin-top:10px;white-space:pre-wrap}
</style></head><body>

<header>
  <h1>EEG motor imagery — left vs right hand</h1>
  <p>Runs locally. Nothing is uploaded anywhere; recordings stay in <code>data/recordings/</code>.</p>
</header>

<nav id="tabs">
  <button data-t="hw" class="on">1 · Hardware</button>
  <button data-t="rec">2 · Record</button>
  <button data-t="data">3 · Data</button>
  <button data-t="train">4 · Train</button>
  <button data-t="live">5 · Live</button>
</nav>

<main>
  <section id="hw" class="on">
    <div class="card">
      <h2>Which headset works for this?</h2>
      <p class="hint">Ratings describe whether the electrode layout can physically see
        left-vs-right hand imagery — not overall device quality.</p>
      <div id="devlist" class="muted">loading…</div>
    </div>
    <div class="card">
      <h2>Signal check</h2>
      <p class="hint">Do this every session before recording. Resting EEG is 2–15 µV in
        the 8–30 Hz band. A flat channel is not touching skin; high 60 Hz pickup means
        poor contact.</p>
      <div class="row">
        <div><label>Device</label><select id="c_device"></select></div>
        <div><label>Serial port <span class="muted">(OpenBCI)</span></label><input id="c_serial" placeholder="/dev/ttyUSB0"></div>
        <div><label>Electrode positions <span class="muted">(if you moved them)</span></label><input id="c_channels" placeholder="C3,C4,CP3,CP4"></div>
        <div><label>Seconds</label><input id="c_seconds" type="number" value="12"></div>
      </div>
      <button class="go" id="c_run">Run check</button>
      <div id="c_out"></div>
    </div>
  </section>

  <section id="rec">
    <div class="card">
      <h2>Record a calibration session</h2>
      <p class="hint">Do an <b>executed</b> session first — actually squeeze the fist. It
        produces a much stronger signal and proves the hardware works before you blame
        your imagery. Then switch to imagery.</p>
      <div class="row">
        <div><label>Whose head is this?</label><input id="r_subject" placeholder="pranav"></div>
        <div><label>Device</label><select id="r_device"></select></div>
        <div><label>Task</label><select id="r_task">
          <option value="executed">executed (do this first)</option>
          <option value="imagery">imagery</option></select></div>
        <div><label>Trials per side</label><input id="r_trials" type="number" value="30"></div>
        <div><label>Serial port</label><input id="r_serial" placeholder="/dev/ttyUSB0"></div>
        <div><label>Electrode positions</label><input id="r_channels" placeholder="C3,C4,CP3,CP4"></div>
      </div>
      <p class="muted" style="margin-top:14px">
        <b>Tell the subject:</b> feel the movement from the inside — the tension in the
        forearm, the fingers closing. Do not picture a hand from the outside. Keep still,
        loose jaw, try not to blink during the arrow.</p>
      <button class="go" id="r_run">Start — goes full screen</button>
      <div id="r_out"></div>
    </div>
  </section>

  <section id="data">
    <div class="card">
      <h2>Upload recordings</h2>
      <p class="hint">Drop in sessions from your friends' laptops. Accepts this app's
        <code>.npz</code>+<code>.json</code> pairs, a <code>.zip</code> of them, OpenBCI
        GUI CSV, or EDF. Third-party files need a <code>onset_s,label</code> cue CSV
        alongside — without cue times a recording cannot train anything.</p>
      <div class="row">
        <div><label>Whose data is this? <span class="muted">(raw CSV/EDF only)</span></label>
          <input id="u_subject" placeholder="dave"></div>
        <div><label>Electrode positions, in file column order</label>
          <input id="u_channels" placeholder="C3,C4,CP3,CP4"></div>
      </div>
      <p class="muted" style="margin:10px 0 14px">Sessions recorded by this app carry
        their own electrode names — leave those two boxes empty for
        <code>.npz</code>/<code>.zip</code>. For a raw CSV, select the recording
        <b>and</b> its <code>…labels.csv</code> together.</p>
      <div class="drop" id="drop">Drop files here, or click to choose
        <input type="file" id="file" multiple hidden></div>
      <div id="u_out"></div>
    </div>
    <div class="card">
      <h2>What we have so far</h2>
      <div id="sessions" class="muted">loading…</div>
    </div>
  </section>

  <section id="train">
    <div class="card">
      <h2>Train on everything collected</h2>
      <p class="hint">Trains per person and, with two or more people, also tests
        leave-one-person-out — how well a model built on your friends decodes someone
        it has never seen. Expect that to be much worse. That gap is the result, not a bug.</p>
      <div class="row">
        <div><label>Save model as</label><input id="t_save" value="models/group.joblib"></div>
        <div><label>Cross-subject test</label><select id="t_cross">
          <option value="yes">yes (needs 2+ people)</option>
          <option value="no">no</option></select></div>
      </div>
      <button class="go" id="t_run">Train</button>
      <div id="t_out"></div>
    </div>
  </section>

  <section id="live">
    <div class="card">
      <h2>Live decoding</h2>
      <p class="hint">Sliding 3-second window, 4 predictions/second, smoothed over the
        last second. Says "uncertain" below 65% confidence rather than guessing.
        Expect 1–2 seconds of lag — that is inherent to motor imagery.</p>
      <div class="row">
        <div><label>Model</label><select id="l_model"></select></div>
        <div><label>Device</label><select id="l_device"></select></div>
        <div><label>Serial port</label><input id="l_serial" placeholder="/dev/ttyUSB0"></div>
        <div><label>Electrode positions</label><input id="l_channels" placeholder="C3,C4,CP3,CP4"></div>
      </div>
      <button class="go" id="l_run">Start decoding</button>
      <div id="l_bar" style="margin-top:22px;font-size:28px;font-family:ui-monospace,monospace"></div>
      <div id="l_out"></div>
    </div>
  </section>
</main>

<div id="stage"><div><div id="arrow"></div><div id="word"></div></div>
  <div id="progress"></div></div>

<script>
const $ = s => document.querySelector(s);
const api = async (name, body) => {
  const r = await fetch('/api/' + name, body === undefined
    ? {} : {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
  const j = await r.json();
  if (j.error) throw new Error(j.error);
  return j;
};
const esc = s => String(s).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));

document.querySelectorAll('#tabs button').forEach(b => b.onclick = () => {
  document.querySelectorAll('#tabs button').forEach(x => x.classList.remove('on'));
  document.querySelectorAll('main section').forEach(x => x.classList.remove('on'));
  b.classList.add('on'); $('#' + b.dataset.t).classList.add('on');
  if (b.dataset.t === 'data') loadSessions();
  if (b.dataset.t === 'live') loadModels();
});

// ---- devices -------------------------------------------------------------
let DEVICES = [];
async function loadDevices(){
  const {devices} = await api('devices');
  DEVICES = devices;
  const rows = devices.map(d => `<tr>
      <td><b>${esc(d.name)}</b><br><span class="muted">${esc(d.price)} · ${esc(d.connection)}</span></td>
      <td><span class="pill ${d.rating}">${d.rating}</span></td>
      <td class="muted">${d.channels.length ? esc(d.channels.join(', ')) : 'you place them'}
        ${d.repositionable ? '<br><i>repositionable</i>' : ''}</td>
      <td class="muted">${esc(d.note)}</td></tr>`).join('');
  $('#devlist').innerHTML = `<table><tr><th>Device</th><th>For this task</th>
      <th>Electrodes</th><th>Notes</th></tr>${rows}</table>`;
  for (const sel of ['#c_device','#r_device','#l_device']) {
    $(sel).innerHTML = devices.map(d =>
      `<option value="${d.key}">${esc(d.name)} — ${d.rating}</option>`).join('');
  }
}

// ---- signal check --------------------------------------------------------
$('#c_run').onclick = async () => {
  const btn = $('#c_run'); btn.disabled = true; $('#c_out').innerHTML = '<p class="muted">streaming…</p>';
  try {
    const r = await api('check', {device:$('#c_device').value, serial_port:$('#c_serial').value,
      channels:$('#c_channels').value, seconds:+$('#c_seconds').value});
    const rows = r.channels.map(c => `<tr><td>${esc(c.channel)}</td><td>${c.rms_uv}</td>
      <td>${c.line_ratio}</td><td><span class="pill ${c.status==='ok'?'good':'bad'}">${c.status}</span></td></tr>`).join('');
    $('#c_out').innerHTML = `<table><tr><th>Electrode</th><th>RMS µV</th><th>60 Hz ratio</th>
      <th></th></tr>${rows}</table>` + (r.all_ok
        ? '<p class="muted" style="margin-top:12px">All channels usable — go record.</p>'
        : '<p class="err">Fix the flagged electrodes before recording. Re-seat them, part the hair, add gel if your setup uses it.</p>');
  } catch(e){ $('#c_out').innerHTML = `<div class="err">${esc(e.message)}</div>`; }
  btn.disabled = false;
};

// ---- recording -----------------------------------------------------------
const ARROWS = {left:'←', right:'→'};
$('#r_run').onclick = async () => {
  $('#r_out').innerHTML = '';
  try {
    await api('record/start', {device:$('#r_device').value, subject:$('#r_subject').value,
      task:$('#r_task').value, trials_per_class:+$('#r_trials').value,
      serial_port:$('#r_serial').value, channels:$('#r_channels').value});
    $('#stage').classList.add('on'); pollRecord();
  } catch(e){ $('#r_out').innerHTML = `<div class="err">${esc(e.message)}</div>`; }
};
async function pollRecord(){
  let s;
  try { s = await api('record/status'); } catch(e){ return; }
  if (s.phase === 'cue' || s.phase === 'imagery') {
    $('#arrow').textContent = ARROWS[s.label] || '';
    $('#word').textContent = s.label ? 'imagine ' + s.label + ' hand' : '';
  } else if (s.phase === 'fixation') {
    $('#arrow').textContent = '+'; $('#word').textContent = '';
  } else if (s.phase === 'rest') {
    $('#arrow').textContent = '·'; $('#word').textContent = 'rest';
  }
  $('#progress').textContent = s.total ? `trial ${s.trial} / ${s.total}` : '';
  if (!s.running && (s.phase === 'done' || s.phase === 'error')) {
    $('#stage').classList.remove('on');
    $('#r_out').innerHTML = s.phase === 'done'
      ? `<p class="muted" style="margin-top:12px">${esc(s.message)}</p>`
      : `<div class="err">${esc(s.message)}</div>`;
    loadSessions(); return;
  }
  setTimeout(pollRecord, 100);
}

// ---- data ----------------------------------------------------------------
async function loadSessions(){
  try {
    const {sessions, subjects, total_trials} = await api('sessions');
    if (!sessions.length) { $('#sessions').textContent = 'No recordings yet.'; return; }
    const rows = sessions.map(s => s.error
      ? `<tr><td>${esc(s.file)}</td><td colspan="5" class="err">${esc(s.error)}</td></tr>`
      : `<tr><td>${esc(s.subject)}</td><td>${esc(s.task)}</td>
         <td>${s.n_trials} <span class="muted">(${s.n_left}L/${s.n_right}R)</span></td>
         <td class="muted">${esc(s.channels.join(', '))}</td>
         <td><span class="pill ${s.montage_rating}">${s.montage_rating}</span></td>
         <td class="muted">${s.duration_s}s</td></tr>`).join('');
    $('#sessions').innerHTML = `<p class="muted">${subjects.length} people ·
      ${total_trials} trials total</p><table><tr><th>Person</th><th>Task</th><th>Trials</th>
      <th>Electrodes</th><th>Montage</th><th>Length</th></tr>${rows}</table>`;
  } catch(e){ $('#sessions').innerHTML = `<div class="err">${esc(e.message)}</div>`; }
}
const drop = $('#drop'), fileInput = $('#file');
drop.onclick = () => fileInput.click();
drop.ondragover = e => { e.preventDefault(); drop.classList.add('hot'); };
drop.ondragleave = () => drop.classList.remove('hot');
drop.ondrop = e => { e.preventDefault(); drop.classList.remove('hot'); upload(e.dataTransfer.files); };
fileInput.onchange = () => upload(fileInput.files);
async function upload(files){
  if (!files.length) return;
  const fd = new FormData();
  for (const f of files) fd.append('file', f, f.name);
  if ($('#u_subject').value) fd.append('subject', $('#u_subject').value);
  if ($('#u_channels').value) fd.append('channels', $('#u_channels').value);
  $('#u_out').innerHTML = '<p class="muted">uploading…</p>';
  const r = await fetch('/api/upload', {method:'POST', body:fd});
  const j = await r.json();
  $('#u_out').innerHTML =
    (j.imported && j.imported.length ? `<p class="muted">Imported: ${esc(j.imported.join(', '))}</p>` : '') +
    (j.errors && j.errors.length ? `<div class="err">${esc(j.errors.join('\n'))}</div>` : '') +
    (j.error ? `<div class="err">${esc(j.error)}</div>` : '');
  loadSessions();
}

// ---- train ---------------------------------------------------------------
$('#t_run').onclick = async () => {
  const btn = $('#t_run'); btn.disabled = true;
  $('#t_out').innerHTML = '<p class="muted">training… this can take a few minutes</p>';
  try {
    const r = await api('train', {save:$('#t_save').value, cross_subject:$('#t_cross').value === 'yes'});
    $('#t_out').innerHTML = `<pre>${esc(r.log)}</pre>`;
    loadModels();
  } catch(e){ $('#t_out').innerHTML = `<div class="err">${esc(e.message)}</div>`; }
  btn.disabled = false;
};

// ---- live ----------------------------------------------------------------
async function loadModels(){
  try {
    const {models} = await api('models');
    $('#l_model').innerHTML = models.length
      ? models.map(m => `<option value="models/${m}">${esc(m)}</option>`).join('')
      : '<option value="">train a model first</option>';
  } catch(e){}
}
$('#l_run').onclick = async () => {
  $('#l_out').innerHTML = '';
  try {
    await api('live/start', {model:$('#l_model').value, device:$('#l_device').value,
      serial_port:$('#l_serial').value, channels:$('#l_channels').value, seconds:300});
    pollLive();
  } catch(e){ $('#l_out').innerHTML = `<div class="err">${esc(e.message)}</div>`; }
};
async function pollLive(){
  let s;
  try { s = await api('live/status'); } catch(e){ return; }
  const p = s.probabilities || {}, right = p.right ?? 0.5, w = 28;
  const pos = Math.min(Math.round(right * w), w - 1);
  const track = Array(w).fill('·'); track[pos] = '●';
  $('#l_bar').innerHTML = `LEFT ${track.join('')} RIGHT<br>
    <span style="font-size:20px;color:${s.label==='uncertain'?'#8b98a8':'#58a6ff'}">
    ${esc((s.label||'').toUpperCase())} ${s.confidence ? s.confidence.toFixed(2) : ''}</span>`;
  if (s.error) { $('#l_out').innerHTML = `<div class="err">${esc(s.error)}</div>`; return; }
  if (s.running) setTimeout(pollLive, 200);
}

loadDevices(); loadSessions(); loadModels();
</script></body></html>
"""


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def serve(host: str = "127.0.0.1", port: int = 8000, open_browser: bool = True) -> None:
    """Start the app.

    Binds to loopback by default so the app is not reachable from the rest of
    the network. Pass ``--host 0.0.0.0`` deliberately if a friend on the same
    Wi-Fi needs to reach it — and understand that this serves unauthenticated
    access to your EEG recordings and your hardware while it runs.
    """
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    server = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{'localhost' if host == '127.0.0.1' else host}:{port}"
    print(f"EEG motor imagery app running at {url}")
    print("Recordings are saved to data/recordings/ — nothing leaves this machine.")
    print("Ctrl-C to stop.")
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run the local EEG app.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)
    serve(args.host, args.port, open_browser=not args.no_browser)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
