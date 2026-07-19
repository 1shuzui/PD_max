"""Compatibility module alias for moved core source-matching helpers."""
import sys
from app.ai_detection.core import known_source_matcher as _target
sys.modules[__name__] = _target
