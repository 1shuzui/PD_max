"""Compatibility module alias for moved core timestamp helpers."""
import sys
from app.ai_detection.core import timestamp_checker as _target
sys.modules[__name__] = _target
