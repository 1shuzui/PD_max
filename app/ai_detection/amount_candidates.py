"""Compatibility module alias for moved core amount-candidate helpers."""
import sys
from app.ai_detection.core import amount_candidates as _target
sys.modules[__name__] = _target
