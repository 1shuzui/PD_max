"""Compatibility module alias for moved core semantic helpers."""
import sys
from app.ai_detection.core import semantic_checker as _target
sys.modules[__name__] = _target
