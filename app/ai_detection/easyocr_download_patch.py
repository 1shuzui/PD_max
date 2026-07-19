"""Compatibility module alias for the EasyOCR runtime helper."""
import sys
from app.ai_detection.runtime import easyocr_download_patch as _target
sys.modules[__name__] = _target
