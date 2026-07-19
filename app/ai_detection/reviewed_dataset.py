"""Compatibility module alias for the reviewed-dataset service."""
import sys
from app.ai_detection.services import reviewed_dataset as _target
sys.modules[__name__] = _target
