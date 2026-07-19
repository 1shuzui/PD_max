"""Compatibility module alias for the model-registry service."""
import sys
from app.ai_detection.services import model_registry as _target
sys.modules[__name__] = _target
