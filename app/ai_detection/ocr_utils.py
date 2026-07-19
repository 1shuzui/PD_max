"""Compatibility module alias for moved core OCR helpers."""
import sys
from app.ai_detection.core import ocr_utils as _target
sys.modules[__name__] = _target
