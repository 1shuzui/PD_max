"""Compatibility module alias for the rule-check display service."""
import sys
from app.ai_detection.services import rule_check_display as _target
sys.modules[__name__] = _target
