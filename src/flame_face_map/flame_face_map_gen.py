from dataclasses import dataclass
from inspect import cleandoc
from pathlib import Path
import pickle
from typing import Optional, Tuple

import numpy as np
import torch
from pytorch3d.ops import interpolate_face_attributes
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    MeshRasterizer,
    RasterizationSettings,
    look_at_view_transform,
)
from pytorch3d.structures import Meshes

from .flame import FLAME


@dataclass
class FlameConfig:
    flame_model_path: str = str(Path(__file__).parent / "flame.pkl")
    batch_size: int = 1
    num_worker: int = 4
    ring_margin: float = 0.5
    ring_loss_weight: float = 1.0
    shape_params: int = 300
    expression_params: int = 100
    pose_params: int = 6
    optimize_eyeballpose: bool = True
    optimize_neckpose: bool = True
    use_3D_translation: bool = True


# Backwards compatibility alias
_flameConfig = FlameConfig

RANDOM_SHAPE_VARIANCE = 1.0
RANDOM_EXPRESSION_VARIANCE = 1.0


def _parse_params(param_str: str, expected_length: int, param_name: str) -> torch.Tensor:
    """Parse a comma-separated float string into a float32 tensor of shape (1, expected_length).

    If param_str is empty or whitespace, defaults to zeros (neutral).
    """
    cleaned = param_str.strip() if param_str else ""
    if not cleaned:
        return torch.zeros(1, expected_length, dtype=torch.float32)

    try:
        values = [float(x.strip()) for x in cleaned.split(",") if x.strip()]
    except ValueError as e:
        raise ValueError(f"Failed to parse {param_name} parameters: all values must be valid floats. {e}") from e

    if len(values) != expected_length:
        raise ValueError(f"{param_name} parameter must have length {expected_length}, but got {len(values)} values.")

    return torch.tensor([values], dtype=torch.float32)


def _get_device() -> torch.device:
    """Get the optimal device for mesh rasterization.

    Prefers CUDA if available. Falls back to CPU if only MPS is available
    because PyTorch3D rasterizers do not support Metal/MPS.
    """
    try:
        import comfy.model_management as mm
        device = mm.get_torch_device()
    except (ImportError, AttributeError):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "mps":
        device = torch.device("cpu")
    return device


class FLAME_face_map_gen:
    """
    Generate FLAME face normal map, depth map, and mask.
    """
    config = FlameConfig()
    _flamelayer: Optional[FLAME] = None
    _cached_device: Optional[torch.device] = None
    _face_indices_np: Optional[np.ndarray] = None
    _reindexed_faces: Optional[np.ndarray] = None

    def __init__(self):
        pass

    @classmethod
    def get_flamelayer(cls, device: torch.device) -> FLAME:
        """Lazy-load and cache the FLAME model on the target device."""
        if cls._flamelayer is None or cls._cached_device != device:
            flamelayer = FLAME(cls.config).to(device)
            flamelayer.eval()
            cls._flamelayer = flamelayer
            cls._cached_device = device
        return cls._flamelayer

    @classmethod
    def get_topology(cls, flamelayer: FLAME) -> Tuple[np.ndarray, np.ndarray]:
        """Lazy-load and cache the facial vertex mask and reindexed faces."""
        if cls._face_indices_np is None or cls._reindexed_faces is None:
            mask_path = Path(__file__).parent / "FLAME_masks.pkl"
            with open(mask_path, "rb") as mask_file:
                vertex_masks = pickle.load(mask_file, encoding="latin1")
            face_indices = np.asarray(vertex_masks["face"], dtype=np.int64)
            if face_indices.size == 0:
                raise ValueError("Face vertex mask is empty; cannot proceed with masking.")

            full_faces_np = flamelayer.faces.astype(np.int64)
            vertex_count = int(full_faces_np.max()) + 1
            vertex_id_map = -np.ones(vertex_count, dtype=np.int64)
            vertex_id_map[face_indices] = np.arange(face_indices.shape[0], dtype=np.int64)
            reindexed = vertex_id_map[full_faces_np]
            valid_faces_mask = np.all(reindexed >= 0, axis=1)
            reindexed = reindexed[valid_faces_mask]
            if reindexed.size == 0:
                raise ValueError("No faces remain after applying the facial vertex mask.")

            cls._face_indices_np = face_indices
            cls._reindexed_faces = reindexed
        return cls._face_indices_np, cls._reindexed_faces

    @classmethod
    def INPUT_TYPES(cls):
        """
        Return a dictionary configuring all input fields.
        """
        return {
            "optional": {
                "Shape": ("STRING", {
                    "tooltip": f"Shape parameter in FLAME model\n Format: comma-separated floats of length {cls.config.shape_params}\n Example: 0.0,1.2,0.5,-0.3"
                }),
                "Expression": ("STRING", {
                    "tooltip": f"Expression parameter in FLAME model\n Format: comma-separated floats of length {cls.config.expression_params}\n Example: 0.0,1.2,0.5,-0.3"
                }),
                "Pose": ("STRING", {
                    "tooltip": f"Pose parameter in FLAME model\n Format: comma-separated floats of length {cls.config.pose_params}\n Example: 0.0,1.2,0.5,-0.3,..."
                }),
                "Neck_Pose": ("STRING", {
                    "tooltip": "Neck Pose parameter in FLAME model\n Format: comma-separated floats of length 3\n Example: 0.0,1.2,0.5"
                }),
            },
            "required": {
                "Randomize_Shape": (["enable", "disable", "neutral"],),
                "Randomize_Expression": (["enable", "disable", "neutral"],),
                "Randomize_Pose": (["enable", "disable", "look ahead"],),
                "Randomize_Neck_Pose": (["enable", "disable", "straight"],),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = ("Face Normal Map Output", "Face Depth Map Output", "Mask Output")
    DESCRIPTION = cleandoc(__doc__)
    FUNCTION = "generate"
    CATEGORY = "FLAME"

    def generate(
        self,
        Shape="",
        Expression="",
        Pose="",
        Neck_Pose="",
        Randomize_Shape="enable",
        Randomize_Expression="enable",
        Randomize_Pose="enable",
        Randomize_Neck_Pose="enable",
    ) -> tuple:
        device = _get_device()
        flamelayer = self.get_flamelayer(device)
        face_indices_np, reindexed_faces_np = self.get_topology(flamelayer)

        # Shape parameters
        match Randomize_Shape:
            case "disable":
                shape_params = _parse_params(Shape, self.config.shape_params, "Shape")
            case "neutral":
                shape_params = torch.zeros(1, self.config.shape_params, dtype=torch.float32)
            case _:  # enable
                shape_params = torch.randn(1, self.config.shape_params, dtype=torch.float32) * RANDOM_SHAPE_VARIANCE

        # Expression parameters
        match Randomize_Expression:
            case "disable":
                expression_params = _parse_params(Expression, self.config.expression_params, "Expression")
            case "neutral":
                expression_params = torch.zeros(1, self.config.expression_params, dtype=torch.float32)
            case _:  # enable
                expression_params = torch.randn(1, self.config.expression_params, dtype=torch.float32) * RANDOM_EXPRESSION_VARIANCE

        # Pose parameters
        match Randomize_Pose:
            case "disable":
                pose_params = _parse_params(Pose, self.config.pose_params, "Pose")
            case "look ahead":
                pose_params = torch.zeros(1, self.config.pose_params, dtype=torch.float32)
            case _:  # enable
                pose_params = torch.randn(1, self.config.pose_params, dtype=torch.float32)

        # Neck pose parameters
        match Randomize_Neck_Pose:
            case "disable":
                neck_pose = _parse_params(Neck_Pose, 3, "Neck Pose")
            case "straight":
                neck_pose = torch.zeros(1, 3, dtype=torch.float32)
            case _:  # enable
                neck_pose = torch.randn(1, 3, dtype=torch.float32)

        # Move inputs to device
        shape_params = shape_params.to(device)
        expression_params = expression_params.to(device)
        pose_params = pose_params.to(device)
        neck_pose = neck_pose.to(device)
        eye_pose = torch.randn(1, 6, device=device)

        with torch.no_grad():
            # Forward pass of FLAME
            vertices = flamelayer(shape_params, expression_params, pose_params, neck_pose, eye_pose)

            # Mask vertices to keep only the facial region
            face_indices_tensor = torch.as_tensor(face_indices_np, dtype=torch.long, device=device)
            vertices = vertices.index_select(1, face_indices_tensor)

            # Faces tensor referencing only facial vertices
            faces_tensor = (
                torch.as_tensor(reindexed_faces_np, dtype=torch.long, device=device)
                .unsqueeze(0)
                .contiguous()
            )

            # Generate vertex normals
            meshes = Meshes(verts=vertices, faces=faces_tensor).to(device)
            vertex_normals_tensor = meshes.verts_normals_padded()

            # Camera setup
            R, T = look_at_view_transform(dist=1.0, elev=0.0, azim=0.0)
            R = R.to(device)
            T = T.to(device)
            cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=20.0)

            raster_settings = RasterizationSettings(
                image_size=512,
                blur_radius=0.0,
                faces_per_pixel=1,
            )
            rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)

            # Rasterize the mesh
            fragments = rasterizer(meshes)

            # Visibility mask: [B, H, W] - 1.0 where mesh is visible, 0.0 for background
            # ComfyUI standard MASK shape is [B, H, W]
            mask_tensor = (fragments.pix_to_face[..., 0] >= 0).float()

            N, V, C = vertex_normals_tensor.shape
            N, F, _ = faces_tensor.shape

            faces_idx = faces_tensor.view(N, -1).unsqueeze(2).expand(N, F * 3, C)
            face_normals = vertex_normals_tensor.gather(1, faces_idx)
            face_attributes_tensor = face_normals.view(N, F, 3, C)

            # Normal map
            pixel_normals = interpolate_face_attributes(
                fragments.pix_to_face, fragments.bary_coords, face_attributes_tensor.squeeze(0)
            )
            pixel_normals = torch.nan_to_num(pixel_normals)

            normal_map_tensor = pixel_normals[..., 0, :3]  # [B, H, W, 3]
            normal_map_tensor = normal_map_tensor * mask_tensor.unsqueeze(-1)

            # Convert normals from [-1, 1] to [0, 1] range
            normal_map_image = (normal_map_tensor * 0.5) + 0.5

            # Depth map
            pixel_depths = fragments.zbuf[..., 0]
            pixel_depths = torch.nan_to_num(pixel_depths, nan=0.0, posinf=0.0, neginf=0.0)

            # Normalize depth values to [0, 1] range for visualization
            masked_depths = pixel_depths.clone()
            masked_depths[mask_tensor == 0] = float("nan")

            valid_depths = masked_depths[~torch.isnan(masked_depths)]
            if valid_depths.numel() > 0:
                min_depth = valid_depths.min()
                max_depth = valid_depths.max()
                if max_depth > min_depth:
                    # Invert so closer pixels are brighter (higher values)
                    depth_map_tensor = 1.0 - ((pixel_depths - min_depth) / (max_depth - min_depth))
                    depth_map_tensor = depth_map_tensor * mask_tensor
                else:
                    depth_map_tensor = pixel_depths * mask_tensor
            else:
                depth_map_tensor = pixel_depths * mask_tensor

            # Expand depth map to 3 channels [B, H, W, 3] for ComfyUI IMAGE output
            depth_map_tensor = depth_map_tensor.unsqueeze(-1).expand(-1, -1, -1, 3)

        # ComfyUI standard: return tensors on CPU
        return (normal_map_image.cpu(), depth_map_tensor.cpu(), mask_tensor.cpu())


NODE_CLASS_MAPPINGS = {
    "FLAME_face_map_gen": FLAME_face_map_gen,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FLAME_face_map_gen": "FLAME Face Map Generator",
}

