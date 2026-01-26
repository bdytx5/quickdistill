"""
QuickDistill - A fast and easy toolkit for distilling AI models.

This package provides tools to:
- Capture and view Weave traces
- Run weak models on strong model outputs
- Evaluate similarity using LLM judges
- Export datasets for model evaluation
"""

__version__ = "0.1.0"
__author__ = "Brett Young"
__email__ = "bdytx5@umsystem.edu"

from quickdistill.cli import main

__all__ = ["main"]
