# FDπ: policy-feature Fréchet distance

FDπ measures how close generated camera frames are to real ones **from the
perspective of end-to-end driving policies**. For each policy π_k, we extract the scene-context feature the policy uses to plan on the real
frame and on the generated frame of the same scene, fit a Gaussian to each set,
and compute

```
FD_k   = ||μ_k − μ̃_k||² + Tr(Σ_k + Σ̃_k − 2 (Σ_k Σ̃_k)^½)
FDπ_k  = FD_k / Tr(Σ_k)                 
FDπ    = (1/|Π|) Σ_k FDπ_k              
```

The panel Π currently has five front-camera policies: DrivoR, DiffusionDrive, LTF
(Latent TransFuser), RAP and SparseDriveV2.

FDπ is computed on NAVSIM `navtest` and on nuScenes `val`.

## Installation

Tested with Python 3.10, PyTorch 2.8.0 and CUDA 12.8 on Linux. Building the
mmcv and SparseDriveV2 CUDA ops needs `nvcc` from the CUDA toolkit matching
your PyTorch build.

```bash
conda create -n fdpi python=3.10 -y
conda activate fdpi
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# mmcv (used by RAP) has no prebuilt wheel for this torch version; build it from source.
pip install -U openmim ninja transforms3d
MAX_JOBS=8 mim install "mmcv==2.1.0"

# SparseDriveV2 deformable-aggregation op (builds two .so files into ops/)
cd navsim/agents/sparsedrive/ops && python setup.py build_ext --inplace && cd -

pip install -e .
```

## Data

**NAVSIM**: the OpenScene `test` split (`navsim_logs/test` and
`sensor_blobs/test`), downloaded with the
[NAVSIM instructions](https://github.com/autonomousvision/navsim/blob/main/docs/install.md).
Only the `CAM_F0` images are read.

**nuScenes**: `v1.0-trainval` metadata, the `samples/CAM_FRONT` keyframes and
the [CAN bus expansion](https://www.nuscenes.org/download) (ego velocity and
acceleration).

**Policy checkpoints**:

```bash
hf download UCLA-VAIL/DreamStream --include "policy/*" --include "config.json" --local-dir .
hf download Lanl11/RAP_ckpts RAP_DINO_navsimv1.ckpt --local-dir policy
```

```
policy/
├── drivor_Nav1_f0_25epochs.pth
├── diffusiondrive_front_only_epoch_99_step_33300.ckpt
├── ltf_epoch99_step_11100.ckpt
├── RAP_DINO_navsimv1.ckpt
├── sparsedrivev2_epoch_9_step_6650.ckpt
└── aux/
    ├── kmeans_navsim_traj_20.npy
    └── sparsedrive/kmeans/{path_1024.npy, velocity_256.npy, trajectory_1024_256.npz}
```

The policies' image backbones are from public pretrained weights
(timm ResNet-34 and DINOv2, and
[`facebook/dinov3-vith16plus-pretrain-lvd1689m`](https://huggingface.co/facebook/dinov3-vith16plus-pretrain-lvd1689m)
for RAP).

## Generated frames

Generate the front-camera frame for each evaluated sample and organize the files
as follows. Any image resolution works: frames are resized the same way as real frames.

**NAVSIM**: mirror the `sensor_blobs/test` path of the current front-camera
frame of each `navtest` scene:

```
<render_root>/<log_name>/CAM_F0/<image_token>.jpg
```

Every `navtest` scene whose current frame exists under `<render_root>` is evaluated.

**nuScenes**: one file per keyframe of the 150 `val` scenes, where
`N = 5 × keyframe_index` (the 10 Hz frame index of the 2 Hz keyframe,
`keyframe_index` counting from 0 at the first sample of the scene):

```
<render_root>/scene-XXXX/frame-N/output.png
```

Keyframes with at least three preceding keyframes are evaluated.

## Running

```bash
python -m fdpi.run --dataset navsim --navsim-root /path/to/openscene \
    --render-root /path/to/navsim_render --ckpt-dir /path/to/policy --output-dir out/my_model_navsim

python -m fdpi.run --dataset nuscenes --dataroot /path/to/nuscenes \
    --render-root /path/to/nuscenes_render --ckpt-dir /path/to/policy --output-dir out/my_model_nuscenes
```

For every policy this extracts features on the real and on the generated
frames (`<output-dir>/{real,generated}/<policy>.npz`), then prints the
per-policy FDπ_k and the panel FDπ, and writes them to `<output-dir>/fdpi.json`.
Use one output directory per set of generated frames; existing feature files
in it are reused.

`--agents` restricts the run to a subset of `drivor,dd,ltf,rap,sd`. The two stages can also be
run separately with `python -m fdpi.dump_navsim`, `python -m fdpi.dump_nuscenes`
and `python -m fdpi.frechet`.

## Acknowledgements and licenses

`navsim/` contains the parts of [NAVSIM](https://github.com/autonomousvision/navsim)
(Apache-2.0), and the policy code from
[DrivoR](https://github.com/valeoai/DrivoR) (Apache-2.0),
[DiffusionDrive](https://github.com/hustvl/DiffusionDrive) (MIT),
[RAP](https://github.com/vita-epfl/RAP) (Apache-2.0) and
[SparseDriveV2](https://github.com/swc-17/SparseDriveV2) (Apache-2.0).
