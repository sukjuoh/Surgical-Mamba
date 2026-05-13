# Surgical-Phase-Mamba

Causal, online surgical phase recognition with a dual-path SurgicalMamba block:
a fast per-clip path and a slow cross-clip path with per-chunk Cayley rotation
and λ-modulated time-warp. Evaluated on Cholec80, M2CAI16-workflow, and
AutoLaparo.

## 1. Preparation

### Step 1: Datasets

Download source videos and extract frames at 1 fps:

| Dataset | Source |
| --- | --- |
| Cholec80 | https://camma.unistra.fr/datasets/ |
| M2CAI16-workflow | https://camma.unistra.fr/datasets/ |
| AutoLaparo | https://autolaparo.github.io/ |

Frame extraction with ffmpeg:

```bash
ffmpeg -hide_banner -i <video>.mp4 -r 1 -start_number 0 \
    <out_dir>/<tag>/%08d.jpg
```

Expected per-video directory tags:

| Dataset | Train tag | Test tag |
| --- | --- | --- |
| Cholec80 | `video01`..`video80` | (same) |
| M2CAI16 | `workflow_video_01`..`workflow_video_27` | `test_workflow_video_01`..`test_workflow_video_14` |
| AutoLaparo | `01`..`21` | (same) |

Standard public splits (cuhk4040 / official 27/14 / TeCNO 10/4/7) are
hardcoded in [`train.py`](train.py) (`DATASET_SPLITS`).

Frame extraction helpers are in [`data/`](data/):
`preprocess_videos.py`, `video2frame_cutmargin.py`.

### Step 2: Pretrained backbone

ConvNeXt-Tiny ImageNet-1k weights are downloaded automatically by `timm` on
first use.

### Step 3: Environment

```bash
pip install -r requirements.txt
```

`mamba-ssm` requires a CUDA-capable GPU and matching CUDA toolkit.

## 2. Training

Each dataset has its own shell script:

```bash
bash train_cholec80.sh
bash train_m2cai16.sh
bash train_autolaparo.sh
```

Each script invokes [`train.py`](train.py) with the dataset-specific flags
(data paths, epochs, etc.). Run `python train.py --help` to see all options.

## 3. Model

```
ConvNeXt-Tiny → visual projector → SurgicalMamba × N → MambaHead → per-frame phase logits
```

- [`models/causal_surgical_mamba.py`](models/causal_surgical_mamba.py) — top-level model.
- [`models/surgical_mamba.py`](models/surgical_mamba.py) — dual-path SurgicalMamba block.
- [`models/extractors.py`](models/extractors.py) — frame-level visual extractor (ConvNeXt-Tiny).

For frame-by-frame online inference:

```python
from models import CausalSurgicalMamba, OnlineSession

model = CausalSurgicalMamba(num_phases=7).eval().cuda()
model.load_state_dict(torch.load("cholec80_release.pt")["model"])
session = OnlineSession(model, clip_len=128, device="cuda")
for frame in stream:
    logits = session.step(frame)
```

## 4. Pretrained checkpoints

Released weights (state-dict only):

| Dataset | File |
| --- | --- |
| Cholec80 | `cholec80_release.pt` |
| M2CAI16 | `m2cai16_release.pt` |
| AutoLaparo | `autolaparo_release.pt` |

(Download links: TBD.)

## 5. Citation

```
TBD
```
