"""Compatibility module alias for the candidate-evaluation workflow."""
import sys
from app.ai_detection.workflows import candidate_evaluation as _target
sys.modules[__name__] = _target
