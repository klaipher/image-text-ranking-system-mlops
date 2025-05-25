"""
Inference module for image-text ranking model.
"""

from .predictor import ImageTextRankingPredictor, create_predictor

__all__ = [
    'ImageTextRankingPredictor',
    'create_predictor'
] 