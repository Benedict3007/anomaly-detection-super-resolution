"""
Anomaly Detection using Super-Resolution Models

This package provides implementations of DRN-L and DRCT models for
industrial anomaly detection through super-resolution reconstruction error.
"""

__version__ = "0.1.0"
__author__ = "Benedict Druegh"
__description__ = "Industrial anomaly detection using Transformer-based super-resolution models"

# Import main components
try:
    from .main import main, parse_args
    __all__ = ["main", "parse_args"]
except ImportError:
    # During development, some modules might not be ready
    __all__ = []
