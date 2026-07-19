"""Compatibility module alias for the rule-check service."""
import sys
from app.ai_detection.services import rule_check_service as _target
sys.modules[__name__] = _target
