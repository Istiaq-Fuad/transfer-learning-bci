"""Feature extraction modules (CSP, FBCSP, Riemannian geometry)."""

from bci.features.csp import CSPFeatureExtractor
from bci.features.fbcsp import FBCSPFeatureExtractor
from bci.features.riemannian import RiemannianFeatureExtractor

__all__ = [
    "CSPFeatureExtractor",
    "FBCSPFeatureExtractor",
    "RiemannianFeatureExtractor",
]
