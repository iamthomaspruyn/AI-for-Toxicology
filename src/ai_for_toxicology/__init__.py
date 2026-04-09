"""Minimal package for AI-for-Toxicology submission scripts."""

from .config import TOX21_TASKS
from .model import VAEWithPredictor

__all__ = ["TOX21_TASKS", "VAEWithPredictor"]
