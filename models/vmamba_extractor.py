"""
VMamba Tiny feature extractor for surgical phase recognition.
Outputs pooled feature vectors by removing the classification head from VSSM.

Two variants:
  - VMambaTinyExtractor     : VMamba-Tiny v2 (Mamba1-based, forward_type="v05_noz")
  - VMambaTinyM2Extractor   : VMamba-Tiny Mamba2-based (forward_type="m0_noz")

Reference: https://github.com/MzeroMiko/VMamba
"""

import torch
import torch.nn as nn
from .vmamba import VSSM


class _BaseExtractor(nn.Module):
    def _remove_head(self):
        self.backbone.classifier.head = nn.Identity()
        self.num_features = self.backbone.num_features

    def _load_pretrained(self, ckpt_path: str):
        state_dict = torch.load(ckpt_path, map_location="cpu")
        if "model" in state_dict:
            state_dict = state_dict["model"]
        missing, unexpected = self.backbone.load_state_dict(state_dict, strict=False)
        print(f"[{self.__class__.__name__}] Loaded pretrained from {ckpt_path}")
        if missing:
            print(f"  Missing keys: {missing}")
        if unexpected:
            print(f"  Unexpected keys: {unexpected}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W)
        Returns:
            (B, num_features) pooled feature vector
        """
        return self.backbone(x)


class VMambaTinyExtractor(_BaseExtractor):
    """
    VMamba-Tiny (Mamba1) feature extractor.
    depths=[2,2,5,2], dims=96 -> 768-dim output
    forward_type: v05_noz
    """

    def __init__(self, pretrained: str = None, channel_first: bool = True):
        super().__init__()
        self.backbone = VSSM(
            depths=[2, 2, 5, 2],
            dims=96,
            drop_path_rate=0.2,
            patch_size=4, in_chans=3, num_classes=1000,
            ssm_d_state=1, ssm_ratio=2.0, ssm_dt_rank="auto", ssm_act_layer="silu",
            ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0.0,
            ssm_init="v0", forward_type="v05_noz",
            mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False,
            patch_norm=True, norm_layer=("ln2d" if channel_first else "ln"),
            downsample_version="v3", patchembed_version="v2",
            use_checkpoint=False, posembed=False, imgsize=224,
        )
        self._remove_head()
        if pretrained is not None:
            self._load_pretrained(pretrained)


class VMambaTinyM2Extractor(_BaseExtractor):
    """
    VMamba-Tiny Mamba2 feature extractor.
    depths=[2,2,4,2], dims=96 -> 768-dim output
    forward_type: m0_noz, ssm_d_state=64
    """

    def __init__(self, pretrained: str = None):
        super().__init__()
        self.backbone = VSSM(
            depths=[2, 2, 4, 2],
            dims=96,
            drop_path_rate=0.2,
            patch_size=4, in_chans=3, num_classes=1000,
            ssm_d_state=64, ssm_ratio=1.0, ssm_dt_rank="auto", ssm_act_layer="gelu",
            ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0.0,
            ssm_init="v2", forward_type="m0_noz",
            mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False,
            patch_norm=True, norm_layer="ln",
            downsample_version="v3", patchembed_version="v2",
            use_checkpoint=False, posembed=False, imgsize=224,
        )
        self._remove_head()
        if pretrained is not None:
            self._load_pretrained(pretrained)
