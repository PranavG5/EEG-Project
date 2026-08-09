"""Classifiers: classical CSP/band-power pipelines and EEGNet.

Every model here is built by a factory returning a fitted-on-demand
scikit-learn-compatible object, so :mod:`.train` and :mod:`.evaluate` treat
"CSP+LDA" and "EEGNet" identically. That uniformity is what makes the
comparison in the README a fair one — same folds, same preprocessing, same
metrics.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .config import CANONICAL_SFREQ, N_CSP_COMPONENTS, RANDOM_SEED
from .features import BandPowerFeatures, WaveletEnergyFeatures, make_csp


# --------------------------------------------------------------------------
# Classical pipelines
# --------------------------------------------------------------------------


def build_bandpower_lda(sfreq: float = CANONICAL_SFREQ) -> Pipeline:
    """Log band power -> LDA. The interpretable baseline.

    LDA with shrinkage ('ledoit_wolf' via ``solver="lsqr"``) rather than plain
    LDA because BCI datasets are wide and short — a few dozen trials against
    tens of features — so the empirical covariance is badly conditioned.
    Shrinkage pulls it toward a scaled identity and is close to free accuracy.
    """
    return Pipeline(
        [
            ("features", BandPowerFeatures(sfreq=sfreq)),
            ("scale", StandardScaler()),
            ("clf", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        ]
    )


def build_csp_lda(
    n_components: int = N_CSP_COMPONENTS, sfreq: float = CANONICAL_SFREQ
) -> Pipeline:
    """CSP -> LDA. The standard reference pipeline in the BCI literature.

    This combination has been the benchmark since Ramoser et al. (2000) and is
    still competitive with deep learning on small per-subject datasets, because
    CSP encodes the right inductive bias (the discriminative information is a
    spatial pattern of band power) instead of having to learn it from 45 trials.
    """
    return Pipeline(
        [
            ("csp", make_csp(n_components)),
            ("scale", StandardScaler()),
            ("clf", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        ]
    )


def build_csp_svm(
    n_components: int = N_CSP_COMPONENTS, sfreq: float = CANONICAL_SFREQ
) -> Pipeline:
    """CSP -> RBF SVM.

    The interesting comparison against CSP+LDA: if the RBF kernel wins, the
    class boundary in CSP-feature space is non-linear, which for log-variance
    features usually means the subject's imagery strategy shifted mid-session.
    If it does not win — the common outcome — that is evidence the linear model
    was already capturing the structure, and worth saying so explicitly.
    """
    # Wrapped in CalibratedClassifierCV so that predict_proba is available for
    # the live decoder's confidence threshold. An SVM's decision function is a
    # signed distance, not a probability; Platt scaling on top of it is what
    # makes "only act when >65% confident" mean anything.
    svm = CalibratedClassifierCV(
        SVC(kernel="rbf", C=1.0, gamma="scale", random_state=RANDOM_SEED),
        method="sigmoid",
        cv=3,
    )
    return Pipeline(
        [
            ("csp", make_csp(n_components)),
            ("scale", StandardScaler()),
            ("clf", svm),
        ]
    )


def build_wavelet_lda(sfreq: float = CANONICAL_SFREQ) -> Pipeline:
    """DWT sub-band energy -> LDA (stretch-goal feature set)."""
    return Pipeline(
        [
            ("features", WaveletEnergyFeatures(sfreq=sfreq)),
            ("scale", StandardScaler()),
            ("clf", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        ]
    )


# --------------------------------------------------------------------------
# EEGNet
# --------------------------------------------------------------------------


def _build_eegnet_module(
    n_channels: int,
    n_times: int,
    n_classes: int = 2,
    f1: int = 8,
    d: int = 2,
    f2: int | None = None,
    kernel_length: int | None = None,
    dropout: float = 0.25,
    sfreq: float = CANONICAL_SFREQ,
):
    """Construct the EEGNet architecture (Lawhern et al., 2018).

    The design is a deliberate translation of classical BCI signal processing
    into learned layers, which is why it works with so few parameters:

    * **Block 1a — temporal convolution** (``F1`` kernels of length ~sfreq/2,
      along time only). Each kernel is a learned FIR bandpass filter. Where the
      classical pipeline hard-codes 8-30 Hz, this learns which sub-bands carry
      discriminative information. Kernel length is half the sampling rate so
      the receptive field spans ~500 ms, i.e. at least 4 cycles of mu.
    * **Block 1b — depthwise spatial convolution** (kernel ``(n_channels, 1)``,
      ``D`` filters per temporal filter, grouped so each operates within one
      frequency band). This is the learned analogue of CSP: a weighted sum
      across electrodes producing a spatially filtered signal, learned per
      frequency band rather than on broadband data. The max-norm constraint on
      these weights is EEGNet's regulariser of choice, preventing any single
      electrode from dominating.
    * **Block 2 — separable convolution** (depthwise along time, then pointwise
      across feature maps). Summarises each feature map's temporal envelope and
      then mixes maps. Decoupling the two is what keeps the parameter count in
      the low thousands.
    * **Average pooling + linear classifier.** Pooling turns the filtered
      signal into an average-power estimate over a time window — the same
      quantity CSP's log-variance step computes, arrived at by a different road.

    ELU activations and batch norm throughout; dropout after each block, since
    with ~45 trials per subject overfitting is the dominant failure mode.
    """
    import torch
    import torch.nn as nn

    f2 = f2 or f1 * d
    kernel_length = kernel_length or max(2, int(sfreq // 2))

    class Conv2dWithMaxNorm(nn.Conv2d):
        """Conv2d whose filter weights are renormalised to a max L2 norm.

        Applied to the depthwise spatial layer as in the paper; it constrains
        each spatial filter to a ball, which empirically prevents the network
        from learning a degenerate single-electrode solution.
        """

        def __init__(self, *args: Any, max_norm: float = 1.0, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.max_norm = max_norm

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            with torch.no_grad():
                norm = self.weight.norm(dim=(2, 3), keepdim=True, p=2).clamp(min=1e-8)
                desired = norm.clamp(max=self.max_norm)
                self.weight *= desired / norm
            return super().forward(x)

    class EEGNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # Block 1: temporal filtering then spatial filtering.
            self.conv_temporal = nn.Conv2d(
                1, f1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False
            )
            self.bn_temporal = nn.BatchNorm2d(f1)
            self.conv_spatial = Conv2dWithMaxNorm(
                f1, f1 * d, (n_channels, 1), groups=f1, bias=False, max_norm=1.0
            )
            self.bn_spatial = nn.BatchNorm2d(f1 * d)
            self.pool1 = nn.AvgPool2d((1, 4))
            self.drop1 = nn.Dropout(dropout)

            # Block 2: separable convolution.
            self.conv_depthwise = nn.Conv2d(
                f1 * d, f1 * d, (1, 16), padding=(0, 8), groups=f1 * d, bias=False
            )
            self.conv_pointwise = nn.Conv2d(f1 * d, f2, (1, 1), bias=False)
            self.bn_separable = nn.BatchNorm2d(f2)
            self.pool2 = nn.AvgPool2d((1, 8))
            self.drop2 = nn.Dropout(dropout)

            self.act = nn.ELU()

            with torch.no_grad():
                dummy = torch.zeros(1, 1, n_channels, n_times)
                n_flat = self._features(dummy).shape[1]
            self.classifier = nn.Linear(n_flat, n_classes)

        def _features(self, x: "torch.Tensor") -> "torch.Tensor":
            x = self.bn_temporal(self.conv_temporal(x))
            x = self.drop1(self.pool1(self.act(self.bn_spatial(self.conv_spatial(x)))))
            x = self.conv_pointwise(self.conv_depthwise(x))
            x = self.drop2(self.pool2(self.act(self.bn_separable(x))))
            return x.flatten(start_dim=1)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.classifier(self._features(x))

    return EEGNet()


class EEGNetClassifier(BaseEstimator, ClassifierMixin):
    """scikit-learn wrapper around EEGNet so it can share the evaluation harness.

    Input is the raw ``(n_trials, n_channels, n_times)`` epoch array — no
    hand-designed features, which is the whole point of the comparison against
    CSP.

    Per-trial z-scoring across time is applied inside ``fit``/``predict``
    because EEG amplitude varies by orders of magnitude between sessions,
    subjects and hardware (a research amplifier and a dry-electrode headband do
    not agree on scale), and batch norm alone does not fix a distribution shift
    that large between train and deployment.
    """

    def __init__(
        self,
        n_epochs: int = 200,
        batch_size: int = 16,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        dropout: float = 0.25,
        sfreq: float = CANONICAL_SFREQ,
        device: str = "cpu",
        patience: int = 40,
        verbose: bool = False,
        random_state: int = RANDOM_SEED,
    ) -> None:
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.sfreq = sfreq
        self.device = device
        self.patience = patience
        self.verbose = verbose
        self.random_state = random_state

    # -- helpers ---------------------------------------------------------

    @staticmethod
    def _standardize(x: np.ndarray) -> np.ndarray:
        mean = x.mean(axis=-1, keepdims=True)
        std = x.std(axis=-1, keepdims=True) + 1e-8
        return (x - mean) / std

    # -- sklearn API -----------------------------------------------------

    def fit(self, x: np.ndarray, y: np.ndarray) -> "EEGNetClassifier":
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        torch.manual_seed(self.random_state)
        rng = np.random.default_rng(self.random_state)

        x = self._standardize(np.asarray(x, dtype=np.float32))
        y = np.asarray(y, dtype=np.int64)
        self.classes_ = np.unique(y)

        n_trials, n_channels, n_times = x.shape
        self.n_channels_, self.n_times_ = n_channels, n_times
        dev = torch.device(self.device)
        self.model_ = _build_eegnet_module(
            n_channels, n_times, n_classes=len(self.classes_),
            dropout=self.dropout, sfreq=self.sfreq,
        ).to(dev)

        # Hold out a slice for early stopping. With datasets this small,
        # training to a fixed epoch count either underfits or memorises.
        idx = rng.permutation(n_trials)
        n_val = max(4, int(0.2 * n_trials))
        val_idx, train_idx = idx[:n_val], idx[n_val:]

        tensors = lambda ids: TensorDataset(  # noqa: E731
            torch.from_numpy(x[ids]).unsqueeze(1), torch.from_numpy(y[ids])
        )
        train_loader = DataLoader(
            tensors(train_idx), batch_size=min(self.batch_size, len(train_idx)),
            shuffle=True, drop_last=False,
        )
        xv = torch.from_numpy(x[val_idx]).unsqueeze(1).to(dev)
        yv = torch.from_numpy(y[val_idx]).to(dev)

        opt = torch.optim.Adam(
            self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        loss_fn = torch.nn.CrossEntropyLoss()

        best_loss, best_state, stale = float("inf"), None, 0
        for epoch in range(self.n_epochs):
            self.model_.train()
            for xb, yb in train_loader:
                opt.zero_grad()
                loss = loss_fn(self.model_(xb.to(dev)), yb.to(dev))
                loss.backward()
                opt.step()

            self.model_.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(self.model_(xv), yv))
            if val_loss < best_loss - 1e-4:
                best_loss, stale = val_loss, 0
                best_state = {k: v.detach().clone() for k, v in self.model_.state_dict().items()}
            else:
                stale += 1
                if stale >= self.patience:
                    break
            if self.verbose and epoch % 20 == 0:
                print(f"  epoch {epoch:3d}  val_loss={val_loss:.4f}")

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.model_.eval()
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        import torch

        x = self._standardize(np.asarray(x, dtype=np.float32))
        with torch.no_grad():
            logits = self.model_(
                torch.from_numpy(x).unsqueeze(1).to(torch.device(self.device))
            )
            return torch.softmax(logits, dim=1).cpu().numpy()

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.classes_[self.predict_proba(x).argmax(axis=1)]


def build_eegnet(sfreq: float = CANONICAL_SFREQ, **kwargs: Any) -> EEGNetClassifier:
    return EEGNetClassifier(sfreq=sfreq, **kwargs)


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

MODEL_BUILDERS = {
    "bandpower_lda": build_bandpower_lda,
    "csp_lda": build_csp_lda,
    "csp_svm": build_csp_svm,
    "wavelet_lda": build_wavelet_lda,
    "eegnet": build_eegnet,
}

#: Models that do not need torch, for machines without it installed.
CLASSICAL_MODELS = ("bandpower_lda", "csp_lda", "csp_svm", "wavelet_lda")


def build_model(name: str, sfreq: float = CANONICAL_SFREQ, **kwargs: Any):
    try:
        builder = MODEL_BUILDERS[name]
    except KeyError:
        raise KeyError(
            f"Unknown model '{name}'. Available: {', '.join(MODEL_BUILDERS)}"
        ) from None
    return builder(sfreq=sfreq, **kwargs)
