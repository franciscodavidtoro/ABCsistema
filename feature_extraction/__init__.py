"""
Módulo de extracción de características para el sistema de reidentificación.
"""

# from .facial_features import FacialFeatureExtractor
from .body_features import BodyFeatureExtractor
from .feature_vector import FeatureVector
from .hog import HOGExtractor

__all__ = [
    # 'FacialFeatureExtractor',
    'BodyFeatureExtractor',
    'FeatureVector',
    'HOGExtractor'
]
