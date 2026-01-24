"""
Módulo de preprocesadores de imágenes para extracción de características.

Incluye tres tipos de preprocesadores:
- BLP: Binary Local Patterns
- HSH: Histogram of Spatial Hue
- LBP: Local Binary Patterns
"""

from .base_preprocessor import BasePreprocessor
from .blp_preprocessor import BLPPreprocessor
from .hsh_preprocessor import HSHPreprocessor
from .lbp_preprocessor import LBPPreprocessor

__all__ = [
    'BasePreprocessor',
    'BLPPreprocessor',
    'HSHPreprocessor', 
    'LBPPreprocessor'
]
