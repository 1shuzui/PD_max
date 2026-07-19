"""Compatibility module alias for moved core bbox-overlap helpers."""
import sys
from app.ai_detection.core import bbox_overlap_checker as _target
sys.modules[__name__] = _target
