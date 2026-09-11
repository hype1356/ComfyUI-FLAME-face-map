"""ComfyUI FLAME Face Map custom node."""

try:
    from .src.flame_face_map.flame_face_map_gen import (
        FLAME_face_map_gen,
        NODE_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS,
    )
except ImportError:
    from src.flame_face_map.flame_face_map_gen import (
        FLAME_face_map_gen,
        NODE_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS,
    )


__all__ = [
    "FLAME_face_map_gen",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]

__author__ = "Swayem Kandangwa"
__email__ = "sk5g22@soton.ac.uk"
__version__ = "0.0.1"
