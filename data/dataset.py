"""Per-video clip dataset for surgical phase recognition.

Loads one video at a time and yields sequential clips of ``seq_len`` frames.
The last clip is zero-padded when the video length is not a multiple of
``seq_len``; a boolean valid_mask indicates which positions are real frames.

Supports any dataset that follows the layout::

    {data_root}/{tag}/{frame_id:08d}.jpg
    {phase_dir}/{tag}-phase.txt
    {tool_dir}/{tag}-tool.txt        # optional; pass tool_dir="_no_tools" if absent

``tag`` is produced from the integer ``video_id`` via ``tag_format`` (e.g.
``"video{:02d}"`` for Cholec80, ``"workflow_video_{:02d}"`` for M2CAI16,
``"{:02d}"`` for AutoLaparo).

Phase annotation format (tab-separated, header row)::

    Frame  Phase
    0      Preparation

The second column may be a phase name (mapped via ``CHOLEC80_PHASES``) or an
integer phase ID; both are accepted automatically.
"""

import os
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


CHOLEC80_PHASES = [
    "Preparation",
    "CalotTriangleDissection",
    "ClippingCutting",
    "GallbladderDissection",
    "GallbladderPackaging",
    "CleaningCoagulation",
    "GallbladderRetraction",
]

PHASE2IDX = {p: i for i, p in enumerate(CHOLEC80_PHASES)}

TOOL_COLS = ["Grasper", "Bipolar", "Hook", "Scissors", "Clipper", "Irrigator", "SpecimenBag"]


def _load_phase_labels(path: str) -> List[int]:
    labels = []
    with open(path) as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            tok = parts[1]
            if tok.lstrip("-").isdigit():
                labels.append(int(tok))
            else:
                labels.append(PHASE2IDX[tok])
    return labels


def _load_tool_annots(path: str) -> np.ndarray:
    rows = []
    with open(path) as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            rows.append([float(x) for x in parts[1:8]])
    return np.array(rows, dtype=np.float32)


class VideoClipDataset(Dataset):
    """Sequential clips for a single video.

    Args:
        video_id:    integer video number, formatted via ``tag_format``.
        data_root:   directory containing per-video frame subdirectories.
        phase_dir:   directory containing ``{tag}-phase.txt`` files.
        tool_dir:    directory with ``{tag}-tool.txt``; pass ``"_no_tools"`` to
                     fill tool annotations with zeros (for datasets without
                     tool labels).
        seq_len:     frames per clip.
        img_size:    spatial size for resizing frames.
        is_train:    if True, apply train-time augmentation.
        tag_format:  Python format string mapping ``video_id`` → directory tag.
    """

    def __init__(
        self,
        video_id: int,
        data_root: str,
        phase_dir: str,
        tool_dir: str,
        seq_len: int = 60,
        img_size: int = 224,
        is_train: bool = False,
        tag_format: str = "video{:02d}",
    ):
        self.seq_len = seq_len
        self.img_size = img_size
        self.is_train = is_train

        tag = tag_format.format(video_id)
        frame_dir = os.path.join(data_root, tag)
        phase_path = os.path.join(phase_dir, f"{tag}-phase.txt") \
            if os.path.isabs(phase_dir) else os.path.join(data_root, phase_dir, f"{tag}-phase.txt")
        tool_path = os.path.join(tool_dir, f"{tag}-tool.txt") \
            if os.path.isabs(tool_dir) else os.path.join(data_root, tool_dir, f"{tag}-tool.txt")

        self.frame_files = sorted(
            os.path.join(frame_dir, f) for f in os.listdir(frame_dir) if f.endswith(".jpg")
        )
        self.num_frames = len(self.frame_files)

        self.phase_labels = _load_phase_labels(phase_path)
        assert len(self.phase_labels) == self.num_frames, (
            f"{tag}: phase labels {len(self.phase_labels)} != frames {self.num_frames}"
        )

        if tool_dir == "_no_tools":
            self.tool_annots = np.zeros((self.num_frames, len(TOOL_COLS)), dtype=np.float32)
        else:
            self.tool_annots = _load_tool_annots(tool_path)
            assert len(self.tool_annots) == self.num_frames, (
                f"{tag}: tool annots {len(self.tool_annots)} != frames {self.num_frames}"
            )

        self.num_clips = (self.num_frames + seq_len - 1) // seq_len

        self._norm_mean = [0.485, 0.456, 0.406]
        self._norm_std = [0.229, 0.224, 0.225]

        self._eval_transform = A.Compose([
            A.SmallestMaxSize(max_size=img_size),
            A.CenterCrop(height=img_size, width=img_size),
            A.Normalize(mean=self._norm_mean, std=self._norm_std),
            ToTensorV2(),
        ])

        self._train_transform = A.Compose([
            A.SmallestMaxSize(max_size=img_size + 40),
            A.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5,
            ),
            A.RandomCrop(height=img_size, width=img_size),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.Normalize(mean=self._norm_mean, std=self._norm_std),
            ToTensorV2(),
        ])

    def __len__(self) -> int:
        return self.num_clips

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Returns:
            frames:      (T, 3, H, W) float32
            tool_annots: (T, 7)       float32
            labels:      (T,)         int64 phase indices
            valid_mask:  (T,)         bool, False for padded positions
            frame_start: int, absolute frame index of the first frame
        """
        start = idx * self.seq_len
        end = min(start + self.seq_len, self.num_frames)
        real_len = end - start

        tfm = self._train_transform if self.is_train else self._eval_transform

        frame_list = []
        for i in range(start, end):
            img = np.array(Image.open(self.frame_files[i]).convert("RGB"))
            frame_list.append(tfm(image=img)["image"])

        frames = torch.stack(frame_list)
        tools = torch.from_numpy(self.tool_annots[start:end])
        labels = torch.tensor(self.phase_labels[start:end], dtype=torch.long)

        if real_len < self.seq_len:
            pad = self.seq_len - real_len
            frames = torch.cat([frames, frames[-1:].expand(pad, -1, -1, -1)], dim=0)
            tools = torch.cat([tools, tools[-1:].expand(pad, -1)], dim=0)
            labels = torch.cat([labels, labels[-1:].expand(pad)], dim=0)

        valid_mask = torch.zeros(self.seq_len, dtype=torch.bool)
        valid_mask[:real_len] = True

        return frames, tools, labels, valid_mask, start
