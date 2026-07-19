"""Compatibility module alias for the review-audit service."""
import sys
from app.ai_detection.services import review_audit as _target
sys.modules[__name__] = _target
