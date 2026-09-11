"""FLAME face map module."""

from .flame import FLAME
from .flame_face_map_gen import (
    FlameConfig,
    FLAME_face_map_gen,
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
    _flameConfig,
)

__all__ = [
    "FLAME",
    "FlameConfig",
    "FLAME_face_map_gen",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "_flameConfig",
]
