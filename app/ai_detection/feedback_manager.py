"""Compatibility module alias for the feedback service."""
import sys
from app.ai_detection.services import feedback_manager as _target
sys.modules[__name__] = _target
