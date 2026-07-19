"""Compatibility module alias for runtime asset helpers."""
import sys
from app.ai_detection.runtime import assets as _target
sys.modules[__name__] = _target
