"""Model training and inference components."""

from src.models.baseline_trainer import train
from src.models.simplifier import Simplifier

__all__ = ["train", "Simplifier"]
