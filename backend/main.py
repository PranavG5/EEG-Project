"""FastAPI backend for the EEG motor-imagery decoder.

A thin, CORS-enabled HTTP layer over :mod:`core`. It exposes the decoder pipeline
as a REST API so any frontend (the bundled Next.js/Vercel app, a notebook, curl,
or a future hardware client) can drive it.

Run locally:

    uvicorn main:app --reload --app-dir backend
    # interactive docs at http://localhost:8000/docs

Endpoints
---------
GET  /api/health              liveness probe
GET  /api/decoders            available decoders (drives the UI's model picker)
GET  /api/subjects            demo subject ids
POST /api/decode              calibrate a decoder and decode trials

The ``/api/decode`` endpoint accepts either a demo subject or uploaded EDF
file(s), so "upload your own data" and "try a built-in subject" are the same
call. Design note: the set of decoders and the shape of the result come entirely
from :mod:`core`, so new models/outputs appear here with no changes to this file.
"""

from __future__ import annotations

import os

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import core

_STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

app = FastAPI(
    title="EEG Motor-Imagery Decoder API",
    version="1.0.0",
    description="Decode imagined left- vs right-hand movement from EEG.",
)

# CORS: allow the Vercel frontend (and local dev) to call the API. Set
# ALLOWED_ORIGINS to a comma-separated list in production to lock this down.
_origins_env = os.environ.get("ALLOWED_ORIGINS", "*")
_origins = ["*"] if _origins_env.strip() == "*" else [
    o.strip() for o in _origins_env.split(",") if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Response schemas (documentation + validation) --------------------------

class DecoderInfo(BaseModel):
    key: str
    name: str
    description: str


class Trial(BaseModel):
    index: int
    imagined: str
    decoded: str
    confidence: float
    correct: bool


class DecodeResult(BaseModel):
    decoder: str
    decoder_name: str
    source: str
    n_trials: int
    n_channels: int
    sfreq: float
    chance: float
    accuracy: float
    class_names: list[str]
    confusion_matrix: list[list[int]]
    trials: list[Trial]
    topomap_png: str | None


# --- Endpoints --------------------------------------------------------------

@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "service": "eeg-decoder", "decoders": list(core.DECODERS)}


@app.get("/api/decoders", response_model=list[DecoderInfo])
def decoders() -> list[dict]:
    return core.list_decoders()


@app.get("/api/subjects")
def subjects() -> dict:
    return {"subjects": core.DEMO_SUBJECTS}


@app.post("/api/decode", response_model=DecodeResult)
async def decode(
    decoder: str = Form("csp_lda"),
    subject: int | None = Form(None),
    files: list[UploadFile] = File(default=[]),
) -> dict:
    """Calibrate ``decoder`` and decode trials from a demo subject or uploads.

    Provide exactly one data source: either ``subject`` (a demo subject id) or
    one or more uploaded EDF ``files``. Uploaded files must be EEGMMIDB-format
    recordings carrying T1/T2 (left/right fist) annotations.
    """
    has_files = bool(files)
    if has_files == (subject is not None):
        raise HTTPException(
            status_code=400,
            detail="Provide exactly one of: a demo 'subject' or uploaded 'files'.",
        )

    try:
        if has_files:
            payload = [(f.filename or "upload.edf", await f.read()) for f in files]
            epochs = core.epochs_for_uploaded_files(payload)
            source_label = f"{len(payload)} uploaded file(s)"
        else:
            epochs = core.epochs_for_demo_subject(int(subject))
            source_label = f"Demo subject {subject}"

        return core.decode(epochs, decoder, source_label)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500,
                            detail=f"Decoding failed: {exc}")


# --- Web UI -----------------------------------------------------------------
# The same server serves the single-page frontend, so the whole app is one
# process at one URL: no separate build step, no CORS config for local use.

app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """Serve the web UI."""
    return FileResponse(os.path.join(_STATIC_DIR, "index.html"))
