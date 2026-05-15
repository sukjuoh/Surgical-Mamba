# Surgical-Phase-Mamba

Causal, online surgical phase recognition with a dual-path SurgicalMamba block:
a fast per-clip path and a slow cross-clip path with per-chunk Cayley rotation
and λ-modulated time-warp. Evaluated on Cholec80, M2CAI16-workflow, and
AutoLaparo with a ConvNeXt-Tiny visual backbone.

## Installation

```bash
pip install -r requirements.txt
```

`mamba-ssm` requires a CUDA-capable GPU and a matching CUDA toolkit.
ConvNeXt-Tiny ImageNet-1k weights are downloaded automatically by `timm` on
first use.

## Datasets

Download the source videos and extract frames at 1 fps:

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

Helpers are in [`data/`](data/): `preprocess_videos.py`,
`video2frame_cutmargin.py`.

Standard public splits are hardcoded in [`train.py`](train.py) under
`DATASET_SPLITS` — Cholec80 (40 train / 40 test), M2CAI16 (27 train / 14 test,
official split with `workflow_video_*` and `test_workflow_video_*` tags), and
AutoLaparo (10 / 4 / 7).

## Training

One shell script per dataset:

```bash
bash train_cholec80.sh
bash train_m2cai16.sh
bash train_autolaparo.sh
```

Each script invokes [`train.py`](train.py) with dataset-specific flags
(data paths, epochs, chunk sizes). Run `python train.py --help` for the
full argparse listing.

## Evaluation

Phase-recognition metrics (per-video accuracy, precision, recall, Jaccard)
follow the MATLAB evaluation protocol from
[TMRNet](https://github.com/YuemingJin/TMRNet/tree/main/code/eval) —
`Main.m` for Cholec80 / AutoLaparo and `Main_m2cai.m` for M2CAI16.

## Model

```
ConvNeXt-Tiny → visual projector → SurgicalMamba × N → MambaHead → per-frame phase logits
```

- [`models/causal_surgical_mamba.py`](models/causal_surgical_mamba.py) — top-level model.
- [`models/surgical_mamba.py`](models/surgical_mamba.py) — dual-path SurgicalMamba block (fast + slow with per-chunk Cayley rotation and λ-modulated time-warp).
- [`models/extractors.py`](models/extractors.py) — ConvNeXt-Tiny visual extractor.

## Pretrained checkpoints

Released weights (state-dict only):

| Dataset | Phases | File |
| --- | --- | --- |
| Cholec80 | 7 | `cholec80_release.pt` |
| M2CAI16 | 8 | `m2cai16_release.pt` |
| AutoLaparo | 7 | `autolaparo_release.pt` |

Download links: TBD.

Load and run online inference:

```python
import torch
from models import CausalSurgicalMamba, OnlineSession

model = CausalSurgicalMamba(num_phases=7).eval().cuda()  # 8 for M2CAI16
model.load_state_dict(torch.load("cholec80_release.pt")["model"])

session = OnlineSession(model, clip_len=128, device="cuda")
for frame in stream:               # frame: (3, H, W) tensor
    logits = session.step(frame)
```

## Citation

```
@misc{oh2026surgicalmambadualpathssdstate,
      title={SurgicalMamba: Dual-Path SSD with State Regramming for Online Surgical Phase Recognition}, 
      author={Sukju Oh and Sukkyu Sun},
      year={2026},
      eprint={2605.14889},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2605.14889}, 
}
```
