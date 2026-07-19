"""Compatibility module alias for resource-limit runtime helpers."""
import sys
from app.ai_detection.runtime import resource_limits as _target
sys.modules[__name__] = _target
