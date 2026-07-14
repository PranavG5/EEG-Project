"""Classifier definitions for the EEG motor-imagery pipeline.

Milestone 2 provides the classical linear baseline: Linear Discriminant
Analysis on band-power features. LDA (with shrinkage) is the standard reference
classifier in the BCI literature — it is fast, has no hyperparameters to tune,
and is well matched to the near-Gaussian, low-dimensional feature vectors that
band power and CSP produce. Later milestones add an SVM and the EEGNet CNN here.
"""

from __future__ import annotations

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler


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
