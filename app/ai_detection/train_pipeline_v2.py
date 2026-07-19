"""Compatibility module alias for the v3 training workflow."""
import sys
from app.ai_detection.workflows import training_v3 as _target
sys.modules[__name__] = _target
