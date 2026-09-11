# ComfyUI FLAME Face Map

A [ComfyUI](https://github.com/comfyanonymous/ComfyUI) custom node that generates 3D facial maps using the [FLAME](https://flame.is.tue.mpg.de/) 3D statistical face model.

This node comes with the FLAME 2023 model.

---

## Quickstart

### Method 1: ComfyUI-Manager (Recommended)
1. Install [ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager).
2. Search for **ComfyUI FLAME Face Map** in the node manager and click **Install**.
3. Restart ComfyUI.

### Method 2: Manual Installation
1. Navigate to your ComfyUI custom nodes directory:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/hype1356/ComfyUI-FLAME-face-map.git
   cd ComfyUI-FLAME-face-map
   ```
2. Install the required dependencies in your ComfyUI Python environment:
   ```bash
   pip install -r requirements.txt
   ```
3. Restart ComfyUI.

> [!NOTE]
> `pytorch3d` is compiled for your specific PyTorch and CUDA version. If installing manually on CUDA systems, ensure PyTorch3D matches your installed PyTorch version. On Apple Silicon (MPS), the rasterizer automatically falls back to CPU to ensure stability.

---

## Node Reference

### `FLAME Face Map Generator`
- **Category**: `FLAME`
- **Return Types**: `(IMAGE, IMAGE, MASK)`
- **Return Names**: `("Face Normal Map Output", "Face Depth Map Output", "Mask Output")`

#### Inputs

| Input | Type | Options / Format | Description |
| :--- | :--- | :--- | :--- |
| **`Randomize_Shape`** | COMBO | `enable`, `disable`, `neutral` | Controls shape randomization or manual input. |
| **`Randomize_Expression`** | COMBO | `enable`, `disable`, `neutral` | Controls expression randomization, manual input, or neutral face. |
| **`Randomize_Pose`** | COMBO | `enable`, `disable`, `look ahead` | Controls pose randomization, manual input, or look-ahead alignment. |
| **`Randomize_Neck_Pose`** | COMBO | `enable`, `disable`, `straight` | Controls neck randomization, manual input, or straight posture. |
| **`Shape`** *(Optional)* | STRING | 300 comma-separated floats (You can enter less than 300 values and the rest will be filled with zeroes) | Custom shape coefficients (used when `Randomize_Shape` is `disable`). |
| **`Expression`** *(Optional)* | STRING | 100 comma-separated floats (You can enter less than 100 values and the rest will be filled with zeroes) | Custom expression coefficients (used when `Randomize_Expression` is `disable`). |
| **`Pose`** *(Optional)* | STRING | 6 comma-separated floats (You can enter less than 6 values and the rest will be filled with zeroes) | Custom pose coefficients (used when `Randomize_Pose` is `disable`). |
| **`Neck_Pose`** *(Optional)* | STRING | 3 comma-separated floats (You can enter less than 3 values and the rest will be filled with zeroes) | Custom neck coefficients (used when `Randomize_Neck_Pose` is `disable`). |

#### Outputs

| Output | Type | Shape | Description |
| :--- | :--- | :--- | :--- |
| **`Face Normal Map Output`** | `IMAGE` | `[1, 512, 512, 3]` | Surface normal map in `[0, 1]` range. |
| **`Face Depth Map Output`** | `IMAGE` | `[1, 512, 512, 3]` | Inverted and normalized depth map. |
| **`Mask Output`** | `MASK` | `[1, 512, 512]` | Standard ComfyUI binary visibility mask. |

---

## Acknowledgments & Citations

This node builds on the [FLAME model](https://flame.is.tue.mpg.de/) and [SMPL-X](https://github.com/vchoutas/smplx).

```bibtex
@article{FLAME,
  title = {Learning a model of facial shape and expression from {4D} scans},
  author = {Li, Tianye and Bolkart, Timo and Black, Michael. J. and Li, Hao and Romero, Javier},
  journal = {ACM Transactions on Graphics, (Proc. SIGGRAPH Asia)},
  volume = {36},
  number = {6},
  year = {2017},
  pages = {194:1--194:17},
  url = {https://doi.org/10.1145/3130800.3130813}
}
```

## License

This custom node is licensed under the [GNU General Public License v3 (GPLv3)](LICENSE).
FLAME model weights are subject to the [FLAME License](https://flame.is.tue.mpg.de/license.html).
