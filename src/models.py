"""Classifier definitions for the EEG motor-imagery pipeline.

Two families of models live here:

- **Classical (milestones 2-4).** Linear Discriminant Analysis and an RBF-SVM on
  band-power / CSP features. LDA (with shrinkage) is the standard reference
  classifier in the BCI literature — fast, no hyperparameters, and well matched
  to the near-Gaussian, low-dimensional features that band power and CSP produce.

- **Deep learning (milestone 5).** ``EEGNet``, a compact convolutional network
  designed specifically for EEG. It replaces the hand-built feature pipeline with
  learned filters: a temporal convolution acts as a bank of band-pass filters, a
  depthwise spatial convolution learns CSP-like spatial filters, and a separable
  convolution summarises the temporal dynamics — all end-to-end from the raw
  epoch tensor. :class:`EEGNetClassifier` wraps it in a scikit-learn estimator so
  it slots into the exact same cross-validation machinery as the classical models.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch import nn

from features import make_csp


def make_lda() -> Pipeline:
    """Build a standardise -> shrinkage-LDA pipeline.

    Two deliberate choices for the small-sample EEG setting:

    - **StandardScaler** puts every channel's log-power on a common scale so no
      channel dominates the linear boundary purely due to its units/variance.
    - **Shrinkage LDA** (``solver='lsqr', shrinkage='auto'``) regularises the
      class-covariance estimate toward a scaled identity. With only a few dozen
      trials but 64 channels, the empirical covariance is rank-deficient and
      plain LDA is unstable/singular; Ledoit-Wolf shrinkage ("auto") fixes this
      analytically without a manual tuning parameter. This is the standard,
      robust form of LDA used throughout the BCI literature.

    Returns
    -------
    Pipeline
        An unfitted scikit-learn pipeline.
    """
    return make_pipeline(
        StandardScaler(),
        LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
    )


def make_csp_lda(n_components: int = 6) -> Pipeline:
    """Build the reference BCI pipeline: CSP spatial filtering -> shrinkage LDA.

    This ``CSP + LDA`` combination is *the* standard baseline in motor-imagery
    BCI. CSP learns supervised spatial filters that maximise the left/right
    variance (band-power) contrast and emits compact log-variance features; LDA
    draws the linear boundary between the two classes in that space.

    The whole thing is one estimator so that, under cross-validation, CSP is
    re-fit on each training fold only — fitting CSP on all trials first would
    leak label information from the test fold and inflate accuracy.

    Input is 3-D epoched data of shape (n_trials, n_channels, n_times); CSP
    reduces it to (n_trials, n_components) before the classifier.

    Parameters
    ----------
    n_components
        Number of CSP components (features) to retain.

    Returns
    -------
    Pipeline
        An unfitted scikit-learn pipeline (CSP -> LDA).
    """
    return make_pipeline(
        make_csp(n_components=n_components),
        LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
    )


def make_csp_svm(n_components: int = 6) -> Pipeline:
    """Build a CSP -> RBF-SVM pipeline.

    Same CSP front end as :func:`make_csp_lda`, but the classifier is a
    support-vector machine with a radial-basis-function kernel. The motivation
    for swapping LDA for an RBF-SVM is to allow a *non-linear* decision boundary
    in CSP-feature space: LDA can only draw a hyperplane, whereas the RBF kernel
    can carve out curved boundaries, which can help if the two classes are not
    linearly separable in the log-variance features.

    A ``StandardScaler`` sits between CSP and the SVM because RBF-SVMs are highly
    scale-sensitive: the kernel is a function of Euclidean distance, so features
    on larger numeric scales would dominate. ``gamma='scale'`` and ``C=1.0`` are
    the sensible scikit-learn defaults; these could be cross-validated later, but
    are kept fixed here to avoid tuning on such a small dataset.

    As with CSP+LDA, CSP is inside the pipeline so it is re-fit per training fold
    under cross-validation.

    Parameters
    ----------
    n_components
        Number of CSP components (features) to retain.

    Returns
    -------
    Pipeline
        An unfitted scikit-learn pipeline (CSP -> StandardScaler -> RBF SVM).
    """
    return make_pipeline(
        make_csp(n_components=n_components),
        StandardScaler(),
        SVC(kernel="rbf", C=1.0, gamma="scale"),
    )


# --- EEGNet (deep learning, milestone 5) -----------------------------------

class EEGNet(nn.Module):
    """EEGNet: a compact CNN for EEG, after Lawhern et al. (2018).

    The architecture is deliberately a learned analogue of the classical BCI
    pipeline, which is why each block maps onto a signal-processing idea:

    1. **Temporal convolution** — ``F1`` filters of shape ``(1, kern_length)``
       slide along *time* on the raw channels. Each learns a frequency-selective
       temporal kernel, so this block is a bank of learned band-pass filters
       (compare the fixed 8-30 Hz filter used for the classical pipeline). Using
       ``kern_length ≈ sfreq/2`` lets a kernel span ~0.5 s, enough to resolve
       the mu/beta rhythms that carry motor imagery.
    2. **Depthwise spatial convolution** — a ``(n_channels, 1)`` kernel applied
       independently to each temporal feature map (``groups=F1``) collapses the
       channel axis into ``D`` spatial filters per frequency band. This is a
       learned, supervised version of CSP's spatial filtering: it finds channel
       weightings (e.g. a C3-vs-C4 contrast) that best separate the classes. A
       max-norm constraint on these weights (applied during training) regularises
       the spatial filters, as in the original paper.
    3. **Separable convolution** — a depthwise temporal conv followed by a
       pointwise (1x1) mix summarises how each spatial-filtered band evolves over
       time while keeping the parameter count tiny (this is what makes EEGNet a
       ~few-thousand-parameter model rather than a heavyweight CNN).
    4. **Classifier** — average-pooled features are flattened into a single dense
       layer to the class logits.

    ELU activations, batch-norm, average pooling (which acts like the temporal
    smoothing/variance summary that band-power features compute by hand), and
    dropout complete each block. Input is a 4-D tensor ``(batch, 1, n_channels,
    n_times)``.
    """

    def __init__(
        self,
        n_channels: int,
        n_samples: int,
        n_classes: int = 2,
        f1: int = 8,
        depth_multiplier: int = 2,
        f2: int | None = None,
        kern_length: int = 80,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        f2 = f2 if f2 is not None else f1 * depth_multiplier

        # Block 1a: temporal convolution = bank of learned band-pass filters.
        # 'same' padding keeps the time axis length unchanged.
        self.temporal = nn.Sequential(
            nn.Conv2d(1, f1, (1, kern_length), padding="same", bias=False),
            nn.BatchNorm2d(f1),
        )

        # Block 1b: depthwise spatial convolution = learned CSP-like spatial
        # filters. groups=f1 keeps each temporal band's spatial filtering
        # independent; the (n_channels, 1) kernel collapses the channel axis.
        self.spatial = nn.Sequential(
            nn.Conv2d(f1, f1 * depth_multiplier, (n_channels, 1),
                      groups=f1, bias=False),
            nn.BatchNorm2d(f1 * depth_multiplier),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),        # ~temporal smoothing / decimation
            nn.Dropout(dropout),
        )

        # Block 2: separable convolution (depthwise temporal + pointwise mix).
        self.separable = nn.Sequential(
            nn.Conv2d(f1 * depth_multiplier, f1 * depth_multiplier, (1, 16),
                      padding="same", groups=f1 * depth_multiplier, bias=False),
            nn.Conv2d(f1 * depth_multiplier, f2, (1, 1), bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),        # aggressive temporal pooling -> variance-like summary
            nn.Dropout(dropout),
        )

        # Determine the flattened feature size with a dummy forward pass, so the
        # classifier layer adapts to n_channels/n_samples/kern_length without
        # hard-coded arithmetic.
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_samples)
            n_flat = self._features(dummy).shape[1]
        self.classifier = nn.Linear(n_flat, n_classes)

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.temporal(x)
        x = self.spatial(x)
        x = self.separable(x)
        return torch.flatten(x, start_dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self._features(x))


class EEGNetClassifier(BaseEstimator, ClassifierMixin):
    """scikit-learn wrapper that trains :class:`EEGNet` on epoch tensors.

    Exposing EEGNet through the sklearn estimator API means the same
    cross-validation utilities used for the classical pipelines (``cross_val_
    predict``, ``LeaveOneGroupOut``) drive the deep model too, so subject-
    dependent and cross-subject comparisons stay strictly apples-to-apples: every
    model sees exactly the same train/test folds.

    Input ``X`` is the raw epoch tensor of shape ``(n_trials, n_channels,
    n_times)`` — the very same array CSP consumes — which the wrapper reshapes to
    ``(n_trials, 1, n_channels, n_times)`` for the network.

    Training details, all standard for EEGNet on small BCI datasets:

    - **Per-channel z-scoring** using statistics estimated on the *training* fold
      only (stored and reused at predict time), so no test-set information leaks.
    - **Adam + cross-entropy**, with a small held-out validation split carved from
      the training data for **early stopping** on validation loss. EEG datasets
      are tiny and EEGNet will happily overfit, so early stopping is the main
      guard against it (alongside dropout and the max-norm weight constraints).
    - **Max-norm constraints** on the depthwise spatial-conv and dense-layer
      weights, re-applied after every optimiser step, exactly as in the paper.

    Parameters mirror the network's plus optimisation settings; all are stored
    unmodified in ``__init__`` (no logic) so ``sklearn.clone`` works and the
    estimator can be re-instantiated fresh for each CV fold.
    """

    def __init__(
        self,
        n_classes: int = 2,
        f1: int = 8,
        depth_multiplier: int = 2,
        kern_length: int = 80,
        dropout: float = 0.25,
        decim: int = 1,
        lr: float = 1e-3,
        batch_size: int = 32,
        max_epochs: int = 300,
        patience: int = 40,
        val_fraction: float = 0.2,
        max_norm_spatial: float = 1.0,
        max_norm_dense: float = 0.25,
        weight_decay: float = 0.0,
        random_state: int = 42,
        device: str | None = None,
        verbose: bool = False,
    ) -> None:
        self.n_classes = n_classes
        self.f1 = f1
        self.depth_multiplier = depth_multiplier
        self.kern_length = kern_length
        self.dropout = dropout
        self.decim = decim
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.val_fraction = val_fraction
        self.max_norm_spatial = max_norm_spatial
        self.max_norm_dense = max_norm_dense
        self.weight_decay = weight_decay
        self.random_state = random_state
        self.device = device
        self.verbose = verbose

    # -- helpers ------------------------------------------------------------

    def _resolve_device(self) -> torch.device:
        if self.device is not None:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _decimate(self, X: np.ndarray) -> np.ndarray:
        # Keep every `decim`-th time sample. The epochs are band-passed to
        # 30 Hz upstream, so decimating 160 Hz -> 80 Hz (decim=2) leaves the
        # new 40 Hz Nyquist safely above the signal band: no aliasing, no
        # information lost, but the time axis (and conv cost) roughly halves.
        if self.decim > 1:
            return X[:, :, ::self.decim]
        return X

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        # Per-channel z-score using training statistics stored at fit time.
        return (X - self.channel_mean_) / self.channel_std_

    def _apply_max_norm(self) -> None:
        """Clamp weight norms in place, as EEGNet's max-norm regularisation."""
        with torch.no_grad():
            # Depthwise spatial conv is the first module in self.spatial.
            spatial_conv = self.model_.spatial[0]
            self._renorm_(spatial_conv.weight, self.max_norm_spatial, dim=0)
            self._renorm_(self.model_.classifier.weight, self.max_norm_dense,
                          dim=0)

    @staticmethod
    def _renorm_(weight: torch.Tensor, max_norm: float, dim: int) -> None:
        # Rescale each slice along `dim` so its L2 norm does not exceed max_norm.
        norms = weight.norm(2, dim=tuple(i for i in range(weight.dim())
                                         if i != dim), keepdim=True)
        desired = norms.clamp(max=max_norm)
        weight.mul_(desired / (norms + 1e-8))

    # -- sklearn API --------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "EEGNetClassifier":
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        device = self._resolve_device()

        X = self._decimate(np.asarray(X, dtype=np.float32))
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        y_idx = np.searchsorted(self.classes_, y)
        n_channels, n_samples = X.shape[1], X.shape[2]

        # Per-channel standardisation stats from the (training) data.
        self.channel_mean_ = X.mean(axis=(0, 2), keepdims=True)
        self.channel_std_ = X.std(axis=(0, 2), keepdims=True) + 1e-7
        Xs = self._standardize(X)

        # Carve a validation split for early stopping (stratified if feasible).
        stratify = y_idx if np.min(np.bincount(y_idx)) >= 2 else None
        X_tr, X_val, y_tr, y_val = train_test_split(
            Xs, y_idx, test_size=self.val_fraction,
            random_state=self.random_state, stratify=stratify,
        )

        def to_tensor(a: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
            return torch.as_tensor(a, dtype=dtype, device=device)

        # (N, 1, C, T) input tensors.
        X_tr_t = to_tensor(X_tr, torch.float32).unsqueeze(1)
        X_val_t = to_tensor(X_val, torch.float32).unsqueeze(1)
        y_tr_t = to_tensor(y_tr, torch.long)
        y_val_t = to_tensor(y_val, torch.long)

        self.model_ = EEGNet(
            n_channels=n_channels,
            n_samples=n_samples,
            n_classes=len(self.classes_),
            f1=self.f1,
            depth_multiplier=self.depth_multiplier,
            kern_length=self.kern_length,
            dropout=self.dropout,
        ).to(device)

        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr,
                                     weight_decay=self.weight_decay)
        criterion = nn.CrossEntropyLoss()

        n_train = X_tr_t.shape[0]
        rng = np.random.default_rng(self.random_state)
        best_val = float("inf")
        best_state = None
        epochs_no_improve = 0

        for epoch in range(self.max_epochs):
            self.model_.train()
            perm = rng.permutation(n_train)
            for start in range(0, n_train, self.batch_size):
                idx = perm[start:start + self.batch_size]
                optimizer.zero_grad()
                logits = self.model_(X_tr_t[idx])
                loss = criterion(logits, y_tr_t[idx])
                loss.backward()
                optimizer.step()
                self._apply_max_norm()

            # Validation loss for early stopping.
            self.model_.eval()
            with torch.no_grad():
                val_loss = criterion(self.model_(X_val_t), y_val_t).item()

            if val_loss < best_val - 1e-4:
                best_val = val_loss
                best_state = {k: v.detach().clone()
                              for k, v in self.model_.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if self.verbose and (epoch % 20 == 0 or epochs_no_improve == 0):
                print(f"    epoch {epoch:3d}  val_loss {val_loss:.4f} "
                      f"(best {best_val:.4f})")

            if epochs_no_improve >= self.patience:
                if self.verbose:
                    print(f"    early stop at epoch {epoch} "
                          f"(best val_loss {best_val:.4f})")
                break

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        return self

    def _forward_all(self, X: np.ndarray) -> np.ndarray:
        device = self._resolve_device()
        X = self._decimate(np.asarray(X, dtype=np.float32))
        Xs = self._standardize(X)
        X_t = torch.as_tensor(Xs, dtype=torch.float32,
                              device=device).unsqueeze(1)
        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(X_t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._forward_all(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self._forward_all(X)
        return self.classes_[probs.argmax(axis=1)]


def make_eegnet(sfreq: float = 160.0, decim: int = 2, **kwargs) -> EEGNetClassifier:
    """Build an :class:`EEGNetClassifier` with a sensible temporal-kernel length.

    The temporal-convolution kernel is set to span ~0.5 s of the *effective*
    (post-decimation) sampling rate, long enough to capture the mu/beta
    oscillations that motor imagery modulates. Decimation defaults to 2: the
    epochs are already band-passed to 30 Hz, so dropping 160 Hz to an effective
    80 Hz loses no signal (40 Hz Nyquist) while roughly halving training cost.

    Parameters
    ----------
    sfreq
        Sampling rate of the epochs in Hz (160 Hz for EEGMMIDB).
    decim
        Temporal decimation factor applied inside the classifier.
    **kwargs
        Overrides forwarded to :class:`EEGNetClassifier`.

    Returns
    -------
    EEGNetClassifier
        An unfitted, sklearn-compatible EEGNet estimator.
    """
    effective_sfreq = sfreq / decim
    half = int(effective_sfreq // 2)
    # Prefer an odd kernel length: with 'same' padding an even kernel forces
    # PyTorch to make a zero-padded copy of the input each forward pass, so an
    # odd length is both cleaner and slightly faster with no signal-side cost.
    kwargs.setdefault("kern_length", half if half % 2 else half + 1)
    kwargs.setdefault("decim", decim)
    return EEGNetClassifier(**kwargs)
