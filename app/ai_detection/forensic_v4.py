"""Compatibility module alias for the offline v4 workflow."""
import sys
from app.ai_detection.workflows import forensic_v4 as _target
sys.modules[__name__] = _target
