from inspect import cleandoc
from pathlib import Path

import numpy as np
import torch
import pickle

from .flame import FLAME

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRasterizer
)
from pytorch3d.ops import interpolate_face_attributes

class _flameConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    flame_model_path = str(Path(__file__).parent / "flame.pkl")
    batch_size = 1
    num_worker = 4
    ring_margin = 0.5
    ring_loss_weight = 1.0
    shape_params = 300
    expression_params = 100
    pose_params = 6
    optimize_eyeballpose = True
    optimize_neckpose = True
    use_3D_translation = True

RANDOM_SHAPE_VARIANCE = 1.5
RANDOM_EXPRESSION_VARIANCE = 1.5

class FLAME_face_map_gen:
    """
    Generate FLAME face normal map.
    """
    config = _flameConfig()

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        """
            Return a dictionary which contains config for all input fields.
            Some types (string): "MODEL", "VAE", "CLIP", "CONDITIONING", "LATENT", "IMAGE", "INT", "STRING", "FLOAT".
            Input types "INT", "STRING" or "FLOAT" are special values for fields on the node.
            The type can be a list for selection.

            Returns: `dict`:
                - Key input_fields_group (`string`): Can be either required, hidden or optional. A node class must have property `required`
                - Value input_fields (`dict`): Contains input fields config:
                    * Key field_name (`string`): Name of a entry-point method's argument
                    * Value field_config (`tuple`):
                        + First value is a string indicate the type of field or a list for selection.
                        + Secound value is a config for type "INT", "STRING" or "FLOAT".
        """
        return {
            "optional": {
                "Shape": ("STRING", { "tooltip": f"Shape parameter in FLAME model\n Format: comma-separated floats of length {_flameConfig.shape_params}\n Example: 0.0,1.2,0.5,-0.3"}),
                "Expression": ("STRING", { "tooltip": f"Expression parameter in FLAME model\n Format: comma-separated floats of length {_flameConfig.expression_params}\n Example: 0.0,1.2,0.5,-0.3"}),
                "Pose": ("STRING", {"tooltip": f"Pose parameter in FLAME model\n Format: comma-separated floats of length {_flameConfig.pose_params}\n Example: 0.0,1.2,0.5,-0.3,..." }),
                "Neck_Pose": ("STRING", {"tooltip": f"Neck Pose parameter in FLAME model\n Format: comma-separated floats of length 3\n Example: 0.0,1.2,0.5"}),
            },
            "required": {
                "Randomize_Shape": (["enable", "disable"],),
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

    CATEGORY = "utils"

    def generate(self, Shape="", Expression="", Pose="", Neck_Pose="",
                 Randomize_Shape="enable", Randomize_Expression="enable",
                 Randomize_Pose="enable", Randomize_Neck_Pose="enable") -> tuple:
        flamelayer = FLAME(self.config)

        if Randomize_Shape == "disable" and Shape:
            shape_values = [float(x) for x in Shape.split(",")]
            if len(shape_values) != self.config.shape_params:
                raise ValueError(f"Shape parameter must have length {self.config.shape_params}.")
            shape_params = torch.tensor([shape_values], dtype=torch.float32)
        else:
            shape_params = torch.randn(1, self.config.shape_params) * RANDOM_SHAPE_VARIANCE

        match Randomize_Expression:
            case "disable":
                expression_values = [float(x) for x in Expression.split(",")]
                if len(expression_values) != self.config.expression_params:
                    raise ValueError(f"Expression parameter must have length {self.config.expression_params}.")
                expression_params = torch.tensor([expression_values], dtype=torch.float32)

            case "enable":
                expression_params = torch.randn(1, self.config.expression_params) * RANDOM_EXPRESSION_VARIANCE

            case "neutral":
                expression_params = torch.zeros(1, self.config.expression_params)

        match Randomize_Pose:
            case "disable":
                pose_values = [float(x) for x in Pose.split(",")]
                if len(pose_values) != self.config.pose_params:
                    raise ValueError(f"Pose parameter must have length {self.config.pose_params}.")
                pose_params = torch.tensor([pose_values], dtype=torch.float32)

            case "enable":
                pose_params = torch.randn(1, self.config.pose_params)

            case "look ahead":
                pose_params = torch.zeros(1, self.config.pose_params)

        match Randomize_Neck_Pose:
            case "disable":
                neck_values = [float(x) for x in Neck_Pose.split(",")]
                if len(neck_values) != 3:
                    raise ValueError("Neck Pose parameter must have length 3.")
                neck_pose = torch.tensor([neck_values], dtype=torch.float32)

            case "enable":
                neck_pose = torch.randn(1, 3)

            case "straight":
                neck_pose = torch.zeros(1, 3)

        # Forward Pass of FLAME
        eye_pose = torch.randn(1, 6)
        vertices = flamelayer(
            shape_params, expression_params, pose_params, neck_pose, eye_pose
        )

        # Load vertex masks
        with open(Path(__file__).parent / "FLAME_masks.pkl", "rb") as mask_file:
            vertex_masks = pickle.load(mask_file, encoding="latin1")
        face_indices_np = np.asarray(vertex_masks["face"], dtype=np.int64)
        if face_indices_np.size == 0:
            raise ValueError("Face vertex mask is empty; cannot proceed with masking.")

        # Mask vertices to keep only the facial region
        face_indices_tensor = torch.as_tensor(face_indices_np, dtype=torch.long, device=vertices.device)
        vertices = vertices.index_select(1, face_indices_tensor)

        # Reindex faces so they only reference the retained facial vertices
        full_faces_np = flamelayer.faces.astype(np.int64)
        vertex_count = int(full_faces_np.max()) + 1
        vertex_id_map = -np.ones(vertex_count, dtype=np.int64)
        vertex_id_map[face_indices_np] = np.arange(face_indices_np.shape[0], dtype=np.int64)
        reindexed_faces = vertex_id_map[full_faces_np]
        valid_faces_mask = np.all(reindexed_faces >= 0, axis=1)
        reindexed_faces = reindexed_faces[valid_faces_mask]
        if reindexed_faces.size == 0:
            raise ValueError("No faces remain after applying the facial vertex mask.")

        faces_tensor = (
            torch.as_tensor(reindexed_faces, dtype=torch.long, device=vertices.device)
            .unsqueeze(0)
            .contiguous()
        )

        # Generate vertex normals
        meshes = Meshes(
            verts=vertices,
            faces=faces_tensor
        ).to(vertices.device)
        vertex_normals_tensor = meshes.verts_normals_padded()
        device = meshes.device

        # Set up a camera
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

        # Visibility mask ensures we ignore empty pixels during interpolation
        mask_tensor = (fragments.pix_to_face[..., 0] >= 0).float()

        N, V, C = vertex_normals_tensor.shape
        N, F, _ = faces_tensor.shape

        faces_idx = faces_tensor.view(N, -1)
        faces_idx = faces_idx.unsqueeze(2).expand(N, F * 3, C)

        face_normals = vertex_normals_tensor.gather(1, faces_idx)

        face_attributes_tensor = face_normals.view(N, F, 3, C)

        # Normal map
        pixel_normals = interpolate_face_attributes(
            fragments.pix_to_face, fragments.bary_coords, face_attributes_tensor.squeeze(0)
        )
        pixel_normals = torch.nan_to_num(pixel_normals)

        normal_map_tensor = pixel_normals[..., 0, :3]  # B, H, W, 3
        normal_map_tensor = normal_map_tensor * mask_tensor.unsqueeze(-1)

        # Create a visibility mask where rasterized pixels are white and background is black
        mask_tensor = mask_tensor.unsqueeze(1) if mask_tensor.dim() == 3 else mask_tensor

        # Format and save the normal map
        # Normals are in [-1, 1] range. We need to convert to [0, 1] for saving.
        normal_map_image = (normal_map_tensor * 0.5) + 0.5

        # Depth map
        pixel_depths = fragments.zbuf[..., 0]
        pixel_depths = torch.nan_to_num(pixel_depths, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Normalize depth values to [0, 1] range for visualization
        masked_depths = pixel_depths.clone()
        masked_depths[mask_tensor.squeeze(1) == 0] = float('nan')
        
        valid_depths = masked_depths[~torch.isnan(masked_depths)]
        if valid_depths.numel() > 0:
            min_depth = valid_depths.min()
            max_depth = valid_depths.max()
            
            if max_depth > min_depth:
                # Invert so closer pixels are brighter (higher values)
                depth_map_tensor = 1.0 - ((pixel_depths - min_depth) / (max_depth - min_depth))
                depth_map_tensor = depth_map_tensor * mask_tensor.squeeze(1)
            else:
                depth_map_tensor = pixel_depths * mask_tensor.squeeze(1)
        else:
            depth_map_tensor = pixel_depths * mask_tensor.squeeze(1)
        
        # Add batch dimension if needed and convert to 3-channel for IMAGE output
        depth_map_tensor = depth_map_tensor.unsqueeze(-1).expand(-1, -1, -1, 3)

        return (normal_map_image, depth_map_tensor, mask_tensor)


NODE_CLASS_MAPPINGS = {
    "FLAME_face_map_gen": FLAME_face_map_gen
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FLAME_face_map_gen": "FLAME Face Map Generator"
}
