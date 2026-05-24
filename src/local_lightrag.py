"""Backward-compatible imports for the old local-only config module."""
from lightrag_config import build_lightrag, describe_config, get_config

__all__ = ["build_lightrag", "describe_config", "get_config"]
