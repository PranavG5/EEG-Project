"""Streamlit web UI for the EEG motor-imagery decoder.

Turns the pipeline into an interactive app: choose a data source (upload EDF
recordings, or pick a built-in PhysioNet demo subject), choose a decoder
(band power + LDA, CSP + LDA, or the EEGNet CNN), and the app calibrates it and
"decodes" the trials — showing overall accuracy, a trial-by-trial view of which
hand the model thinks was imagined and how confident it is, a confusion matrix,
and the ERD/ERS brain topomap. A trained EEGNet can be downloaded as a portable
model file for later inference on freshly recorded data.

Run from the repository root:

    streamlit run src/app.py

Why a Streamlit app (not a static web page): the decoder runs on MNE + PyTorch,
so the model lives in Python. Streamlit wraps that Python directly in a browser
UI, which is exactly what is needed to interact with the real model rather than a
mock-up. The app can also be deployed to Streamlit Community Cloud for a public
URL.
"""

from __future__ import annotations

import os
import tempfile

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from evaluate import plot_erd_ers_topomap
from features import band_power
from models import make_csp_lda, make_eegnet, make_lda
from preprocess import (
    epochs_from_edf_paths,
    epochs_to_labels,
    load_subject_epochs,
)

# key -> (display name, needs 3-D epoch tensor?, estimator factory)
DECODERS = {
    "csp_lda": ("CSP + LDA  (reference BCI method)", True,
                lambda sfreq: make_csp_lda(n_components=6)),
    "bandpower_lda": ("Band power + LDA  (simple baseline)", False,
                      lambda sfreq: make_lda()),
    "eegnet": ("EEGNet  (compact CNN, learned features)", True,
               lambda sfreq: make_eegnet(sfreq=sfreq)),
}

HAND_LABEL = {"left_fist": "LEFT hand", "right_fist": "RIGHT hand"}

st.set_page_config(page_title="EEG Motor-Imagery Decoder", page_icon="🧠",
                   layout="wide")


# --- Data loading (cached) -------------------------------------------------

@st.cache_data(show_spinner=False)
def load_demo_epochs_data(subject: int):
    """Load + preprocess + epoch a demo subject; return arrays and the epochs."""
    epochs = load_subject_epochs(subject, runs=(4, 8, 12))
    return _epochs_bundle(epochs)


@st.cache_data(show_spinner=False)
def load_uploaded_epochs_data(file_bytes: list[bytes], names: list[str]):
    """Load + preprocess + epoch uploaded EDF bytes; return arrays and epochs."""
    tmp_paths = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for data, name in zip(file_bytes, names):
            p = os.path.join(tmpdir, os.path.basename(name))
            with open(p, "wb") as fh:
                fh.write(data)
            tmp_paths.append(p)
        epochs = epochs_from_edf_paths(tmp_paths)
        return _epochs_bundle(epochs)


def _epochs_bundle(epochs):
    """Package everything the UI needs from an Epochs object."""
    y, class_names = epochs_to_labels(epochs)
    return {
        "X_epochs": epochs.get_data(copy=False),
        "X_bandpower": band_power(epochs),
        "y": y,
        "class_names": class_names,
        "sfreq": float(epochs.info["sfreq"]),
        "n_channels": len(epochs.ch_names),
        "epochs": epochs,
    }


# --- Decoding --------------------------------------------------------------

def run_decoding(bundle: dict, decoder_key: str):
    """Calibrate the chosen decoder with 5-fold CV and return per-trial results.

    Cross-validated predictions give an honest read: every trial is decoded by a
    model that never trained on it. Also returns a model fit on all trials (for
    saving/download).
    """
    _, needs_epochs, factory = DECODERS[decoder_key]
    X = bundle["X_epochs"] if needs_epochs else bundle["X_bandpower"]
    y = bundle["y"]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    proba = cross_val_predict(factory(bundle["sfreq"]), X, y, cv=cv,
                              method="predict_proba")
    preds = proba.argmax(axis=1)

    # A model fit on everything, for optional download.
    final_model = factory(bundle["sfreq"]).fit(X, y)
    return preds, proba, final_model


def confusion_fig(y, preds, class_names):
    cm = confusion_matrix(y, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=list(class_names))
    fig, ax = plt.subplots(figsize=(4, 3.6))
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title("Confusion matrix (out-of-fold)")
    fig.tight_layout()
    return fig


# --- UI --------------------------------------------------------------------

st.title("🧠 EEG Motor-Imagery Decoder")
st.caption("Decode imagined **left- vs right-hand** movement from EEG. "
           "Upload a recording or try a demo subject, pick a decoder, and watch "
           "it classify each trial.")

with st.sidebar:
    st.header("1 · Data")
    source = st.radio("Source", ["Demo subject (PhysioNet)", "Upload EDF file(s)"])

    bundle = None
    if source == "Demo subject (PhysioNet)":
        subject = st.selectbox(
            "Subject", list(range(1, 11)), index=6,
            help="Subjects vary a lot: 7 and 2 decode cleanly, 5 and 9 poorly.",
        )
        if st.button("Load subject", use_container_width=True):
            with st.spinner(f"Downloading + preprocessing subject {subject}..."):
                st.session_state["bundle"] = load_demo_epochs_data(subject)
                st.session_state["source_label"] = f"Demo subject {subject}"
    else:
        uploads = st.file_uploader(
            "EEGMMIDB-format .edf file(s) with T1/T2 annotations",
            type=["edf"], accept_multiple_files=True,
        )
        if uploads and st.button("Load recording", use_container_width=True):
            with st.spinner("Preprocessing uploaded recording..."):
                st.session_state["bundle"] = load_uploaded_epochs_data(
                    [u.getvalue() for u in uploads], [u.name for u in uploads]
                )
                st.session_state["source_label"] = (
                    f"{len(uploads)} uploaded file(s)")

    st.header("2 · Decoder")
    decoder_key = st.selectbox(
        "Model", list(DECODERS), format_func=lambda k: DECODERS[k][0],
    )
    run = st.button("▶ Decode", type="primary", use_container_width=True,
                    disabled="bundle" not in st.session_state)

bundle = st.session_state.get("bundle")

if bundle is None:
    st.info("👈 Load a demo subject or upload an EDF recording to begin.")
    st.stop()

# Dataset summary.
chance = max(np.bincount(bundle["y"])) / len(bundle["y"])
c1, c2, c3, c4 = st.columns(4)
c1.metric("Source", st.session_state.get("source_label", "—"))
c2.metric("Trials", len(bundle["y"]))
c3.metric("Channels", bundle["n_channels"])
c4.metric("Sampling rate", f"{bundle['sfreq']:.0f} Hz")

if run:
    name = DECODERS[decoder_key][0].split("  ")[0]
    with st.spinner(f"Calibrating {name} with 5-fold cross-validation..."):
        preds, proba, model = run_decoding(bundle, decoder_key)
    st.session_state["results"] = {
        "preds": preds, "proba": proba, "model": model,
        "decoder_key": decoder_key, "name": name,
    }

results = st.session_state.get("results")
if results is None:
    st.info("Pick a decoder and press **Decode**.")
    st.stop()

y = bundle["y"]
class_names = bundle["class_names"]
preds, proba = results["preds"], results["proba"]
acc = float((preds == y).mean())

st.subheader(f"Results — {results['name']}")
m1, m2, m3 = st.columns(3)
m1.metric("Held-out accuracy", f"{acc:.1%}",
          delta=f"{(acc - chance) * 100:+.0f} pts vs chance")
m2.metric("Chance level", f"{chance:.1%}")
m3.metric("Correct", f"{(preds == y).sum()} / {len(y)}")

left, right = st.columns([3, 2])

with left:
    st.markdown("**Trial-by-trial decode** (out-of-fold, so honest)")
    rows = []
    for i in range(len(y)):
        rows.append({
            "trial": i + 1,
            "imagined": HAND_LABEL[class_names[y[i]]],
            "decoded": HAND_LABEL[class_names[preds[i]]],
            # Stored as a 0-100 percentage so the progress column labels it as
            # e.g. "95%" (ProgressColumn formats the raw value, not a fraction).
            "confidence": float(proba[i, preds[i]]) * 100.0,
            "correct": "✅" if preds[i] == y[i] else "❌",
        })
    df = pd.DataFrame(rows)
    st.dataframe(
        df, use_container_width=True, height=360, hide_index=True,
        column_config={
            "confidence": st.column_config.ProgressColumn(
                "confidence", min_value=0.0, max_value=100.0, format="%.0f%%"),
        },
    )

with right:
    st.pyplot(confusion_fig(y, preds, class_names), use_container_width=True)
    if results["decoder_key"] == "eegnet":
        tmp = os.path.join(tempfile.gettempdir(), "eegnet_trained.pt")
        results["model"].save(tmp)
        with open(tmp, "rb") as fh:
            st.download_button("⬇ Download trained EEGNet (.pt)", fh.read(),
                               file_name="eegnet_trained.pt",
                               use_container_width=True)

st.subheader("Where the signal lives — ERD/ERS brain map")
st.caption("Average mu/beta (8-30 Hz) power for each imagined hand and their "
           "left−right contrast. Motor imagery suppresses this rhythm over the "
           "*opposite* motor cortex, so the contrast is lateralised around C3/C4 "
           "— the physical signature the decoder exploits.")
with st.spinner("Computing topomap..."):
    tmp_png = os.path.join(tempfile.gettempdir(), "app_topomap.png")
    plot_erd_ers_topomap(bundle["epochs"], tmp_png)
st.image(tmp_png, use_container_width=True)
