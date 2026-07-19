"""Compatibility module alias for the dataset service."""
import sys
from app.ai_detection.services import dataset_manager as _target
sys.modules[__name__] = _target
