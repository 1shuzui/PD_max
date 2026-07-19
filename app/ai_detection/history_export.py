"""Compatibility module alias for the history-export service."""
import sys
from app.ai_detection.services import history_export as _target
sys.modules[__name__] = _target
