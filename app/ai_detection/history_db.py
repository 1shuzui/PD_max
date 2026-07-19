"""Compatibility module alias for the history persistence service."""
import sys
from app.ai_detection.services import history_db as _target
sys.modules[__name__] = _target
