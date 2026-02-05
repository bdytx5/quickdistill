"""
QuickDistill - A fast and easy toolkit for distilling AI models.

This package provides tools to:
- Capture and view Weave traces
- Run weak models on strong model outputs
- Evaluate similarity using LLM judges
- Export datasets for model evaluation
"""

# Monkey patch for aiohttp/litellm compatibility
# litellm expects aiohttp.ConnectionTimeoutError but it doesn't exist in some versions
try:
    import aiohttp
    if not hasattr(aiohttp, 'ConnectionTimeoutError'):
        aiohttp.ConnectionTimeoutError = aiohttp.ServerTimeoutError
    if not hasattr(aiohttp, 'SocketTimeoutError'):
        aiohttp.SocketTimeoutError = aiohttp.ServerTimeoutError
except Exception:
    pass

__version__ = "0.1.8"
__author__ = "Brett Young"
__email__ = "bdytx5@umsystem.edu"

from quickdistill.cli import main

__all__ = ["main"]
