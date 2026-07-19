"""Compatibility module alias for the training-job service."""
import sys
from app.ai_detection.services import training_jobs as _target
sys.modules[__name__] = _target
