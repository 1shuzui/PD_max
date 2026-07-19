"""Compatibility module alias for the ForgeGuard client service."""
import sys
from app.ai_detection.services import forgeguard_client as _target
sys.modules[__name__] = _target
