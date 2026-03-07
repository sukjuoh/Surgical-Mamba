"""
CholecT80 per-video clip dataset.

Loads one video at a time and yields sequential clips of `seq_len` frames.
The last clip is zero-padded when the video length is not a multiple of seq_len;
a boolean valid_mask (T,) indicates which positions are real frames.

Directory layout expected:
  {data_root}/video{NN:02d}/{frame_id:06d}.jpg   (1-indexed, 1 fps)
  {data_root}/{phase_dir}/video{NN:02d}-phase.txt
  {data_root}/{tool_dir}/video{NN:02d}-tool.txt

Phase annotation format (tab-separated, header row):
  Frame  Phase
  0      Preparation
  ...

Tool annotation format (tab-separated, header row):
  Frame  Grasper  Bipolar  Hook  Scissors  Clipper  Irrigator  SpecimenBag
  0      1        0        0     0         0        0          0
  ...
"""

import os
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from models.surgical_phase_llm import CHOLEC80_PHASES

PHASE2IDX = {p: i for i, p in enumerate(CHOLEC80_PHASES)}

TOOL_COLS = ["Grasper", "Bipolar", "Hook", "Scissors", "Clipper", "Irrigator", "SpecimenBag"]


def _load_phase_labels(path: str) -> List[int]:
    """Returns list of integer phase indices, one per frame (0-indexed)."""
    labels = []
    with open(path) as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split("\t")
            labels.append(PHASE2IDX[parts[1]])
    return labels


def _load_tool_annots(path: str) -> np.ndarray:
    """Returns (N, 7) float32 array of tool presence per frame."""
    rows = []
    with open(path) as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split("\t")
            rows.append([float(x) for x in parts[1:8]])
    return np.array(rows, dtype=np.float32)


class VideoClipDataset(Dataset):
    """
    Sequential clips for a single video.

    Args:
        video_id:    integer video number (1-80)
        data_root:   root of cholec80_preprocessed
        phase_dir:   subdirectory name for phase annotations
        tool_dir:    subdirectory name for tool annotations
        seq_len:     frames per clip (T)
        img_size:    spatial size for resizing frames
        is_train:    if True, apply data augmentation
    """

    def __init__(
        self,
        video_id: int,
        data_root: str,
        phase_dir: str = "phase_annotations_preprocessed",
        tool_dir: str = "tool_annotations_1fps",
        seq_len: int = 60,
        img_size: int = 224,
        is_train: bool = False,
    ):
        self.seq_len = seq_len
        tag = f"video{video_id:02d}"

        frame_dir = os.path.join(data_root, tag)
        phase_path = os.path.join(data_root, phase_dir, f"{tag}-phase.txt")
        tool_path = os.path.join(data_root, tool_dir, f"{tag}-tool.txt")

        # Frame file list (sorted, 1-indexed)
        self.frame_files = sorted(
            os.path.join(frame_dir, f) for f in os.listdir(frame_dir) if f.endswith(".jpg")
        )
        self.num_frames = len(self.frame_files)

        # Annotations
        self.phase_labels = _load_phase_labels(phase_path)  # list[int]
        self.tool_annots = _load_tool_annots(tool_path)      # (N, 7)

        assert len(self.phase_labels) == self.num_frames, (
            f"{tag}: phase labels {len(self.phase_labels)} != frames {self.num_frames}"
        )
        assert len(self.tool_annots) == self.num_frames, (
            f"{tag}: tool annots {len(self.tool_annots)} != frames {self.num_frames}"
        )

        # Number of clips (last clip may be padded)
        self.num_clips = (self.num_frames + seq_len - 1) // seq_len

        _norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        if is_train:
            # Resize slightly larger then random-crop to preserve spatial content
            # while adding position/scale variety.  Color jitter simulates OR
            # lighting changes.  Horizontal flip is anatomically valid for
            # laparoscopic views (mirrored presentation is common).
            self.transform = transforms.Compose([
                transforms.Resize((img_size + 32, img_size + 32)),
                transforms.RandomCrop(img_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05
                ),
                transforms.ToTensor(),
                _norm,
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                _norm,
            ])

    def __len__(self) -> int:
        return self.num_clips

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            frames:     (T, 3, H, W)  float32
            tool_annots:(T, 7)        float32
            labels:     (T,)          int64 phase indices
            valid_mask: (T,)          bool, False for padded positions
        """
        start = idx * self.seq_len
        end   = min(start + self.seq_len, self.num_frames)
        real_len = end - start

        # Load frames
        frame_list = []
        for i in range(start, end):
            img = Image.open(self.frame_files[i]).convert("RGB")
            frame_list.append(self.transform(img))

        frames = torch.stack(frame_list)         # (real_len, 3, H, W)
        tools  = torch.from_numpy(self.tool_annots[start:end])   # (real_len, 7)
        labels = torch.tensor(self.phase_labels[start:end], dtype=torch.long)  # (real_len,)

        # Pad to seq_len if needed
        if real_len < self.seq_len:
            pad = self.seq_len - real_len
            frames = torch.cat([frames, frames[-1:].expand(pad, -1, -1, -1)], dim=0)
            tools  = torch.cat([tools,  tools[-1:].expand(pad, -1)],           dim=0)
            labels = torch.cat([labels, labels[-1:].expand(pad)],              dim=0)

        valid_mask = torch.zeros(self.seq_len, dtype=torch.bool)
        valid_mask[:real_len] = True

        return frames, tools, labels, valid_mask
