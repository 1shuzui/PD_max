"""Compatibility module alias for the v3 inference workflow."""
import sys
from app.ai_detection.workflows import inference_v3 as _target
sys.modules[__name__] = _target
