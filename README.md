<div align="center">

<img src="assets/fig/title.png" width="85%" alt="OmniDexGrasp: Generalizable Dexterous Grasping via Foundation Model and Force Feedback">

<h3>Accepted to ICRA 2026</h3>

<h4>
<a href="https://wyl2077.github.io/">Yi-Lin Wei</a><sup>&#42;</sup>,
<a href="https://zhexiluo.github.io/">Zhexi Luo</a><sup>&#42;</sup>,
Yuhao Lin,
<a href="https://frenkielm.github.io/">Mu Lin</a>,
Zhizhao Liang,
<a href="https://github.com/Rain-Shuoyu">Shuoyu Chen</a>,
<a href="https://www.isee-ai.cn/~zhwshi/">Wei-Shi Zheng</a><sup>†</sup>
</h4>

<sup>&#42;</sup>Equal contribution &nbsp; <sup>†</sup>Corresponding author


<a href="https://arxiv.org/abs/2510.23119">
    <img src='https://img.shields.io/badge/Paper-red?style=for-the-badge&labelColor=B31B1B&color=B31B1B' alt='Paper PDF'></a>
<a href="https://isee-laboratory.github.io/OmniDexGrasp/">
    <img src='https://img.shields.io/badge/Project_Page-orange?style=for-the-badge&labelColor=D35400&color=D35400' alt='Project Page'></a>

<img src="assets/fig/teaser.png" width="100%" alt="OmniDexGrasp Teaser">

</div>

## 📢 News

- 🚧 *Coming soon...*

## 📝 TODO

⬜ Support more dexterous hand types

⬜ One-click online demo

## 📑 Table of Contents

- [🛠️ Installation](#installation)
- [📋 Prerequisites](#prerequisites)
- [▶️ Usage](#usage)
- [🧩 Bring Your Own Data](#custom-data)
- [📄 Citation](#citation)
- [📜 License](#license)
- [🙏 Acknowledgement](#acknowledgement)


## 🛠️ Installation

<a id="installation"></a>

<details>
<summary><b>Installation</b></summary>

```bash
# 1. Clone the repository
git clone --recursive https://github.com/ISEE-Laboratory/OmniDexGrasp.git
cd OmniDexGrasp

# 2. Initialize submodules
git submodule update --init --recursive
```

Due to unresolved dependency conflicts between upstream model stacks, multiple `uv` virtual environments are recommended:

| Env | Module | Reference |
|-----|--------|-----------|
| `.venv` | This repository | See instructions below |
| `.venv-hamer` | HaMeR hand estimation | [geopavlakos/hamer](https://github.com/geopavlakos/hamer) |
| `.venv-gsam` | Grounded-SAM-2 segmentation | [IDEA-Research/Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2) |
| `.venv-megapose` | MegaPose6D pose estimation | [megapose6d/megapose6d](https://github.com/megapose6d/megapose6d) |

For `hamer`, `gsam`, and `megapose`, please refer to their official documentation for model-specific installation details.

Optional dependency groups in `pyproject.toml` are split by runtime:

| Group | Purpose |
|-------|---------|
| `recons-server` | Common FastAPI server wrapper dependencies |
| `hamer-server` | HaMeR server wrapper dependencies |
| `gsam-server` | Grounded-SAM-2 server wrapper dependencies |
| `dataset-download` | Optional HuggingFace dataset download helper |
| `image-generation` | Optional Gemini client for `scripts/gen_human_grasp.py` |
| `dev` | Optional interactive development tools |

**Setting up the `omnidexgrasp` environment with `uv`:**

```bash
# 1. Create and activate the local uv environment
uv venv --python 3.10
source .venv/bin/activate

# 2. Install locked base dependencies
uv sync

# 3. Install PyTorch3D
uv pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git"

# 4. Install nvdiffrast
uv pip install --no-build-isolation "git+https://github.com/NVlabs/nvdiffrast.git"

# 5. Patch and install the local CSDF CUDA extension from submodule
git -C omnidexgrasp/thirdparty/CSDF apply ../../../patches/csdf-torch-cuda-check.patch
uv pip install -e omnidexgrasp/thirdparty/CSDF --no-build-isolation

# 6. Required for EasyHOI optimization
uv pip install --no-build-isolation "git+https://github.com/otaheri/chamfer_distance"
```

> **Note:** This project uses the PyTorch CUDA 13.0 wheel index. Source-build packages require a matching CUDA toolkit.
>
> The CSDF patch replaces an upstream `CHECK_EQ` CUDA error check that fails to compile with newer PyTorch headers.
> Apply it once before installing CSDF.
>
> Optional add-on groups use `uv sync --inexact` so uv does not remove manually built packages such as PyTorch3D, nvdiffrast, and CSDF.
>
> EasyHOI itself is not installed with `uv pip install -e` because the upstream repository does not provide `pyproject.toml` or `setup.py`. Stage 2 adds the EasyHOI source tree to `PYTHONPATH` instead.

**Setting up server environments (`hamer`, `gsam`, `megapose`) with `uv`:**

The upstream model stacks still have conflicting dependencies, so keep separate `uv` virtual environments for these modules. Install each upstream project according to its official documentation inside its own environment, then install this repository's lightweight server dependencies:

```bash
# Example: HaMeR server environment
uv venv .venv-hamer --python 3.10
source .venv-hamer/bin/activate
uv sync --active --only-group hamer-server
uv pip install -e omnidexgrasp/thirdparty/hamer --no-build-isolation

# Example: Grounded-SAM-2 server environment
uv venv .venv-gsam --python 3.10
source .venv-gsam/bin/activate
uv sync --active --only-group gsam-server
uv pip install -e omnidexgrasp/thirdparty/Grounded-SAM-2

# Example: MegaPose6D environment
uv venv .venv-megapose --python 3.10
source .venv-megapose/bin/activate
uv pip install -e omnidexgrasp/thirdparty/megapose6d
```

</details>


## 📋 Prerequisites

<a id="prerequisites"></a>

<details>
<summary><b>Download Assets, Checkpoints & Dataset</b></summary>

```
OmniDexGrasp/
├── assets/
│   ├── mano/models/                # MANO hand model
│   │   ├── MANO_RIGHT.pkl
│   │   └── mano_mean_params.npz
│   └── robo/                       ✅ Included in repo
├── datasets/                      # Download HuggingFace dataset here
└── checkpoints/
    ├── hamer/                      # HaMeR + ViTPose + Detectron2
    │   ├── hamer_ckpts/checkpoints/hamer.ckpt
    │   ├── vitpose_ckpts/vitpose+_huge/wholebody.pth
    │   └── detectron2/model_final_f05665.pkl
    └── gsam2/                      # Grounded-SAM-2
        ├── sam2.1_hiera_base_plus.pt
        └── grounding-dino-base/
```

### Download dataset from HuggingFace

The dataset is available at: https://huggingface.co/datasets/wyl2077/OmniDexGrasp

Download it into the local `datasets/` folder before running the code.

Example using `huggingface_hub`:

```bash
uv sync --inexact --group dataset-download
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='wyl2077/OmniDexGrasp',
    repo_type='dataset',
    local_dir='datasets',
    allow_patterns=['*'],
    ignore_patterns=['.git']
)
PY
```

If you prefer the HuggingFace web UI, download the dataset files manually and place them under `datasets/`.

Download checkpoints following the official documentation of each submodule:
[geopavlakos/hamer](https://github.com/geopavlakos/hamer) |
[IDEA-Research/Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2) |
[megapose6d/megapose6d](https://github.com/megapose6d/megapose6d)

MANO hand model requires registration at [mano.is.tue.mpg.de](https://mano.is.tue.mpg.de/).

</details>


## ▶️ Usage

<a id="usage"></a>

### 🍵 Stage 1: Reconstruction

Reconstruct 3D hand and object from input images. All `python -m ...` commands below are run from the `omnidexgrasp/` directory because the code imports `recons`, `optim`, `human2robo`, and `utils` as top-level packages.

Activate the matching `uv` environment once per terminal before running each command group:

| Command group | Environment |
|---------------|-------------|
| HaMeR server | `source ../.venv-hamer/bin/activate` |
| GSAM server | `source ../.venv-gsam/bin/activate` |
| MegaPose pose estimation | `source ../.venv-megapose/bin/activate` |
| Reconstruction client, optimization, retargeting, visualization | `source ../.venv/bin/activate` |

**Phase 1: Hand & Object Reconstruction**

```bash
# Terminal 1 with .venv-hamer active: Start HaMeR server
python -m recons.server.hamer

# Terminal 2 with .venv-gsam active: Start GSAM server
python -m recons.server.gsam

# Main terminal with .venv active: Run reconstruction client
python -m recons.client
```

**Phase 2: Object Pose Estimation**

```bash
# Kill the HaMeR & GSAM servers first to free VRAM, then run with .venv-megapose active:
python -m recons.pose_est
```

> **Note:** Loading all models simultaneously requires >24GB VRAM, exceeding a single RTX 4090. We split reconstruction into two phases — kill the servers before running pose estimation to free VRAM.

### 🤚 Stage 2: Hand Pose Optimization

**Refine the reconstructed hand pose to achieve physically plausible hand-object interaction.**

```bash
export EASYHOI_ROOT="$PWD/thirdparty/EasyHOI"
export PYTHONPATH="$EASYHOI_ROOT:$EASYHOI_ROOT/src:${PYTHONPATH:-}"
python -m optim.main
```

### 🤖 Stage 3: Human-to-Robot Retargeting

**Map human hand pose to dexterous robot hands**
```bash
python -m human2robo.main
# Specify hand types:
python -m human2robo.main hand_types=[inspire,wuji,shadow]
```

### 👁️ Visualization

Visualize the retargeting results interactively.

```bash
python -m scripts.vis_dexgrasp --output ../out --port 8080
```

Expected output: an interactive 3D viewer to browse tasks, switch hand types, and inspect grasp poses.

<img src="assets/fig/vis.png" width="100%" alt="Visualization Example">


## 🧩 Bring Your Own Data

<a id="custom-data"></a>

<details>
<summary><b>Data Preparation Guide</b></summary>

To run OmniDexGrasp on your own objects, prepare the following files for each grasp instance under `datasets/{task_name}/`:

**1. Capture Scene Image & Depth**

Use an RGB-D camera (e.g., Intel RealSense D435) to capture:
- `scene_image.png` — RGB image of the object
- `depth.png` — Aligned depth map
- `camera.yaml` — Camera intrinsics

Our method is agnostic to the specific depth camera model.

**2. Obtain Object Mesh**

We use [Hyper3D](https://hyper3d.ai/) to reconstruct the object mesh, producing:
- `base.obj` — Object mesh
- `material.mtl` — Material file
- `shaded.png` — Texture map

You can also use open-source alternatives such as [TRELLIS.2](https://github.com/microsoft/TRELLIS.2).

**3. Generate Grasp Image**

We recommend using [gpt-image-1](https://platform.openai.com/docs/guides/image-generation) or [gemini-3-pro-image](https://ai.google.dev/gemini-api/docs/image-generation) to generate `generated_human_grasp.png` — a synthetic image depicting a human hand grasping the object.

If you use `scripts/gen_human_grasp.py`, it writes files named `generated_human_grasp_0.png`, `generated_human_grasp_1.png`, and so on. Pick one generated image and copy or rename it to `generated_human_grasp.png`, which is the filename consumed by `recons.client`, `recons.pose_est`, and `optim.main`.

Install its optional Gemini client dependencies only when you use that script:

```bash
uv sync --inexact --group image-generation
```

For best results, ensure the generated image matches the aspect ratio of the original `scene_image.png`.

</details>


## 📄 Citation

<a id="citation"></a>

If you find this work useful, please consider citing:

```bibtex
@inproceedings{wei2026omnidexgrasp,
  title={OmniDexGrasp: Generalizable Dexterous Grasping via Foundation Model and Force Feedback},
  author={Wei, Yi-Lin and Luo, Zhexi and Lin, Yuhao and Lin, Mu and Liang, Zhizhao and Chen, Shuoyu and Zheng, Wei-Shi},
  booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
  year={2026}
}
```


## 📜 License

<a id="license"></a>

This project is released under the [Apache License 2.0](LICENSE).

## ⭐ Star History

<div align="center">

[![Star History Chart](https://api.star-history.com/svg?repos=ISEE-Laboratory/OmniDexGrasp&type=Date)](https://star-history.com/#ISEE-Laboratory/OmniDexGrasp&Date)

</div>

## 🙏 Acknowledgement

<a id="acknowledgement"></a>

We thank the following open-source projects for their valuable contributions to the community:
- [HaMeR](https://github.com/geopavlakos/hamer)
- [Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2)
- [EasyHOI](https://github.com/lym29/EasyHOI)
- [MegaPose6D](https://github.com/megapose6d/megapose6d)
