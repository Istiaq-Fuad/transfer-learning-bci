"""Feature extraction modules (CSP, Riemannian geometry)."""

from bci.features.csp import (
    CSPFeatureExtractor,
    EnsembleCSPClassifier,
    FBCSPFeatureExtractor,
)
from bci.features.riemannian import (
    FBRiemannianFeatureExtractor,
    RiemannianFeatureExtractor,
    riemannian_recenter,
)

__all__ = [
    "CSPFeatureExtractor",
    "EnsembleCSPClassifier",
    "FBCSPFeatureExtractor",
    "FBRiemannianFeatureExtractor",
    "RiemannianFeatureExtractor",
    "riemannian_recenter",
]
