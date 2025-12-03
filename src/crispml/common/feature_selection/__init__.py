"""
Public API for CRISP-ML feature selection utilities.
"""

from .feature_selection_utils import (
    select_features_auto,
    select_features_include,
    select_features_exclude,
)

__all__ = [
    "select_features_auto",
    "select_features_include",
    "select_features_exclude",
]
