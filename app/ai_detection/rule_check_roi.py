"""Compatibility module alias for moved core ROI helpers."""
import sys
from app.ai_detection.core import rule_check_roi as _target
sys.modules[__name__] = _target
