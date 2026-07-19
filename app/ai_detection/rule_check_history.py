"""Compatibility module alias for the rule-check history service."""
import sys
from app.ai_detection.services import rule_check_history as _target
sys.modules[__name__] = _target
