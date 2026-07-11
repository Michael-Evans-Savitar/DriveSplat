# DriveSplat

This is the official code release for **DriveSplat: Unified Neural Gaussian Reconstruction for Dynamic Driving Scenes**.

DriveSplat reconstructs dynamic driving scenes with neural Gaussian anchors, hash-based level-of-detail modeling, and anchor-conditioned deformation for non-rigid actors.

## Overview

- Training and rendering entry points: `train.py`, `render.py`
- Core model code: `scene/`, `gaussian_renderer/`, `utils/`, `arguments/`
- Waymo and KITTI preparation helpers: `script/waymo/`, `script/kitti/`
- Required CUDA extensions: `submodules/diff-gaussian-rasterization`, `submodules/simple-knn`
- Waymo TFRecord reader: `submodules/simple-waymo-open-dataset-reader`

## Installation

### Requirements

The released environment is tested with:

- Linux
- NVIDIA GPU with CUDA support
- CUDA 11.7
- Python 3.7.12
- PyTorch 1.13.0
- GCC version compatible with your CUDA toolkit
- `conda`
- `gsutil` / Google Cloud SDK for Waymo downloading
- `colmap` executable in `PATH` for the default Waymo COLMAP initialization

### Create the environment

The step-by-step installation is the most reliable way to reproduce the reference environment:

```bash
conda create -n drive -y python=3.7 pip=22.3.1
conda activate drive
conda install -y pytorch=1.13.0 torchvision=0.14.0 torchaudio=0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install -y pytorch-scatter=2.1.1 -c pyg
pip install -r requirements.txt
```

The equivalent environment file is also provided:

```bash
conda env create -f environment.yml
conda activate drive
```

If `torch-scatter` is not resolved by conda on your machine, install the matching wheel manually:

```bash
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
```

### Install local CUDA extensions

DriveSplat depends on the local modified rasterizer. Do not replace it with the upstream `diff-gaussian-rasterization` package unless you also port DriveSplat's renderer calls.

```bash
export CUDA_HOME="${CONDA_PREFIX}"
export PATH="${CUDA_HOME}/bin:${PATH}"
nvcc --version   # should report CUDA 11.7 for the reference environment
pip install -e submodules/diff-gaussian-rasterization
pip install -e submodules/simple-knn
pip install -e submodules/simple-waymo-open-dataset-reader
```

If `${CONDA_PREFIX}/bin/nvcc` does not exist, install CUDA Toolkit 11.7 in the environment or set `CUDA_HOME` to a system CUDA 11.7 installation before building the local extensions. A mismatched system CUDA, for example CUDA 12.x with PyTorch cu117, can compile a different binary or fail during extension build.

The local `diff-gaussian-rasterization` is important: it returns RGB, radii, depth, alpha, and semantic buffers, and includes the corresponding CUDA kernels/backward path used by this codebase.

The rasterizer source vendored here has been checked against the source used in our reference experimental environment; after excluding build artifacts, the trees are byte-identical. For reproducible performance, install and rebuild this repository's local copy inside the target environment instead of using a package from PyPI or another Gaussian Splatting project.

To force a clean rebuild:

```bash
pip uninstall -y diff-gaussian-rasterization simple-knn
rm -rf submodules/diff-gaussian-rasterization/build \
       submodules/diff-gaussian-rasterization/*.egg-info \
       submodules/simple-knn/build \
       submodules/simple-knn/*.egg-info
pip install -e submodules/diff-gaussian-rasterization
pip install -e submodules/simple-knn
```

Quick import check:

```bash
python - <<'PY'
import torch
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from simple_knn._C import distCUDA2
print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda)
print("DriveSplat CUDA extensions: OK")
PY
```

### Install COLMAP

The example Waymo config uses `init_mode='colmap'`. On the first training run for a scene, DriveSplat invokes the `colmap` command-line tool and writes the reconstructed sparse model to:

```text
<model_path>/colmap/triangulated/sparse/model/
  cameras.bin
  images.bin
  points3D.bin
```

Install COLMAP with your system package manager or conda-forge, and verify it is visible before training:

```bash
colmap -h
```

If your server does not provide COLMAP, you can reuse a precomputed COLMAP sparse model by placing `cameras.bin`, `images.bin`, and `points3D.bin` in the path above before launching training. When these files already exist, the training code skips the COLMAP reconstruction step.

## Prepare Data

DriveSplat follows the Waymo/KITTI organization used by [Street Gaussians](https://github.com/zju3dv/street_gaussians) and the Waymo preparation convention used by [DriveStudio / OmniRe](https://github.com/ziyc/drivestudio). Please follow the licenses and terms of the original datasets.

### Waymo Open Dataset

First authenticate Google Cloud access:

```bash
gcloud auth login
```

Create local data roots:

```bash
mkdir -p data/waymo/raw
mkdir -p data/waymo/new_processed
```

Download the TFRecords needed by a split file. For the example scenes in `script/waymo/waymo_example_scenes.txt`, use the training split list:

```bash
python script/waymo/waymo_download.py \
  --target_dir data/waymo/raw \
  --source gs://waymo_open_dataset_scene_flow/train \
  --segment_file script/waymo/waymo_train_list.txt \
  --split_file script/waymo/waymo_example_scenes.txt
```

For validation scenes, switch both the source and segment list:

```bash
python script/waymo/waymo_download.py \
  --target_dir data/waymo/raw \
  --source gs://waymo_open_dataset_scene_flow/valid \
  --segment_file script/waymo/waymo_val_list.txt \
  --scene_ids 23 24 25
```

Convert TFRecords to DriveSplat's processed layout:

```bash
python script/waymo/waymo_convert.py \
  --root_dir data/waymo/raw \
  --save_dir data/waymo/new_processed \
  --split_file script/waymo/waymo_example_scenes.txt \
  --segment_file script/waymo/segment_list_train.txt
```

Expected processed layout for one scene:

```text
data/waymo/new_processed/026/
  images/
  intrinsics/
  extrinsics/
  ego_pose/
  track/track_info.txt
  dynamic_mask/
  pointcloud.npz
  lidar_depth/
  sky_mask/
  depth/          # monocular depth prior, if use_depth=True
  normal/         # monocular normal prior, if use_normal=True
```

Generate LiDAR depth maps:

```bash
python script/waymo/generate_lidar_depth.py --datadir data/waymo/new_processed/026
```

Generate sky masks with GroundingDINO + SAM. This step needs optional segmentation dependencies:

```bash
pip install git+https://github.com/IDEA-Research/GroundingDINO.git
pip install git+https://github.com/facebookresearch/segment-anything.git supervision
```

Then run:

```bash
python script/generate_sky_mask.py \
  --datadir data/waymo/new_processed/026 \
  --sam_checkpoint /path/to/sam_vit_h_4b8939.pth
```

Monocular depth and normal priors are optional only if the corresponding config disables them. The example Waymo config uses depth priors, so prepare `depth/*.npy` or set `use_depth=False` in the config. Normal priors are loaded from `normal/*.jpg` when the directory exists.

If you store monocular depth priors as images and need normal maps, run:

```bash
python script/depth2normal/generate_waymo_normal.py --datadir data/waymo/new_processed/026
```

### KITTI

Prepare KITTI using the same processed layout expected by `script/kitti/`. The training wrapper expects:

```text
<KITTI_ROOT>/2011_09_26_drive_00XX_sync/
```

Generate LiDAR depth for a processed KITTI sequence:

```bash
python script/kitti/generate_lidar_depth.py --datadir /path/to/kitti/2011_09_26_drive_0001_sync
```

## Configuration

The release includes one example config:

- Example Waymo config: `arguments/waymo_default.py`
- Core AGD switch: `use_anchor_deform=True` in `arguments/__init__.py`
- Non-rigid schedule: `non_rigid_start_iter`, `non_rigid_warmup_iter`
- Hash LOD controls: `use_hash_encoding`, `hash_levels`, `anchor_generation_levels`
- Optional priors: `use_depth`, `use_normal`

When adding a new scene, copy `arguments/waymo_default.py`, update `selected_frames`, camera selection, and loss/prior settings, then pass it through `--configs`. 

## Training

Train one Waymo scene:

```bash
GPU=0 bash scripts/run_waymo.sh 026 data/waymo/new_processed outputs/waymo/paper
```

Equivalent explicit command:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --configs arguments/waymo_default.py \
  -s data/waymo/new_processed/026 \
  --resolution -1 \
  -m outputs/waymo/paper/026 \
  --gpu 0 \
  --visible_threshold 0.01 \
  --base_layer -1 \
  --port 6020
```

Train a KITTI sequence:

```bash
GPU=0 bash scripts/run_kitti.sh 01 /path/to/kitti/processed outputs/kitti/paper
```


## Rendering

Use `render.py` to render a trained checkpoint for visualization or downstream analysis.

Equivalent explicit rendering command:

```bash
CUDA_VISIBLE_DEVICES=0 python render.py \
  --configs arguments/waymo_default.py \
  -s data/waymo/new_processed/026 \
  -m outputs/waymo/paper/026 \
  --iteration -1 \
  --resolution -1 \
  --gpu 0 \
  --visible_threshold 0.01 \
  --base_layer -1 \
  --port 6020
```

## Output Structure

Training writes results under the model path passed by `-m`:

```text
outputs/waymo/paper/026/
  cfg_args
  chkpnt/
  input_ply/
  point_cloud/iteration_30000/
  log_images/
```



## Acknowledgements

This codebase builds on ideas and components from 3D Gaussian Splatting, Street Gaussians, OmniRe, Octree-GS, GroundingDINO, SAM, and the Waymo Open Dataset tooling. The differentiable rasterizer includes DriveSplat-specific modifications for depth, alpha, and semantic rendering.

## Citation

If you find this repository useful, please cite our paper:

```bibtex
@misc{wang2026drivesplatunifiedneuralgaussian,
      title={DriveSplat: Unified Neural Gaussian Reconstruction for Dynamic Driving Scenes},
      author={Cong Wang and Ruiqi Song and Wei Tian and Chenming Zhang and Lingxi Li and Long Chen},
      year={2026},
      eprint={2508.15376},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2508.15376},
}
```
