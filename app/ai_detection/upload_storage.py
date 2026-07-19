"""Compatibility module alias for the upload-storage service."""
import sys
from app.ai_detection.services import upload_storage as _target
sys.modules[__name__] = _target
