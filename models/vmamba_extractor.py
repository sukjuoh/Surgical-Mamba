"""
VMamba Tiny feature extractor for surgical phase recognition.
Outputs pooled feature vectors by removing the classification head from VSSM.

Variants:
  - VMambaTinyExtractor     : VMamba-Tiny v2 (Mamba1-based, forward_type="v05_noz")
  - VMambaTinyM2Extractor   : VMamba-Tiny Mamba2-based (forward_type="m0_noz")
  - CLIPExtractor           : CLIP ViT-L/14 with attention pooling → 768-dim

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


class CLIPExtractor(nn.Module):
    """
    CLIP ViT-L/14 feature extractor with attention pooling.

    CLIP ViT-L/14 internal transformer dim = 1024.
    Patch grid: (224 / 14)^2 = 256 patch tokens per frame.

    Pipeline per frame:
      CLIP vision encoder → hidden_states at mid + final layers
      Mid  layer (depth//2): fine-grained spatial/texture features
      Final layer (depth-1): high-level semantic features

      Each layer: skip CLS → patch tokens (B, 256, 1024)
                  → separate attention pooling (learnable query) → (B, 1024)
      Fuse: LayerNorm(mid_pooled + final_pooled)  → (B, 1024)
      Linear(1024 → 768) → (B, 768)

    Two independent pool_query/pool_attn heads allow mid and final layers to be
    aggregated with different spatial attention patterns — the mid-layer head can
    focus on fine-grained tool regions while the final-layer head captures
    semantic phase-level context.

    Args:
        freeze:           freeze CLIP vision encoder (pooling heads + proj always trainable)
        trainable_layers: unfreeze last N ViT-L/14 transformer layers + final LayerNorm
                          (0 = fully frozen, ViT-L/14 has 24 layers total)
        n_heads:          number of heads in each attention pooling head
    """

    _CLIP_DIM  = 1024    # ViT-L/14 internal hidden dim
    _OUT_DIM   = 768     # target output dim (matches VMamba-Tiny)
    _NUM_LAYERS = 24     # ViT-L/14 transformer depth

    def __init__(self, freeze: bool = True, trainable_layers: int = 0, n_heads: int = 8):
        super().__init__()
        from transformers import CLIPVisionModel
        self.vision_model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
        self.num_features = self._OUT_DIM
        self._mid_layer   = self._NUM_LAYERS // 2  # layer 12

        if freeze:
            for p in self.vision_model.parameters():
                p.requires_grad_(False)
            # Selectively unfreeze last N transformer layers + post_layernorm
            if trainable_layers > 0:
                encoder_layers = self.vision_model.vision_model.encoder.layers
                for layer in encoder_layers[-trainable_layers:]:
                    for p in layer.parameters():
                        p.requires_grad_(True)
                # Also unfreeze final LayerNorm (feeds directly into patch tokens)
                for p in self.vision_model.vision_model.post_layernorm.parameters():
                    p.requires_grad_(True)

        # Mid-layer attention pooling: focuses on fine-grained spatial features
        self.pool_query_mid   = nn.Parameter(torch.randn(1, 1, self._CLIP_DIM) * 0.02)
        self.pool_attn_mid    = nn.MultiheadAttention(self._CLIP_DIM, n_heads, batch_first=True)

        # Final-layer attention pooling: focuses on high-level semantic features
        self.pool_query_final = nn.Parameter(torch.randn(1, 1, self._CLIP_DIM) * 0.02)
        self.pool_attn_final  = nn.MultiheadAttention(self._CLIP_DIM, n_heads, batch_first=True)

        self.fuse_norm = nn.LayerNorm(self._CLIP_DIM)

        # Project fused CLIP dim → VMamba-compatible dim
        self.proj = nn.Linear(self._CLIP_DIM, self._OUT_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W)   — H=W=224 expected
        Returns:
            (B, 768)           — mid+final fused spatially-pooled per-frame feature
        """
        out = self.vision_model(pixel_values=x, output_hidden_states=True)

        # hidden_states[0] = patch embeddings, [1..24] = transformer layer outputs
        # mid layer index in hidden_states = _mid_layer + 1 (offset by embedding layer)
        mid_patches   = out.hidden_states[self._mid_layer + 1][:, 1:, :]  # (B, 256, 1024)
        final_patches = out.last_hidden_state[:, 1:, :]                    # (B, 256, 1024)

        B = x.shape[0]

        # Mid-layer pooling: learns fine-grained spatial aggregation
        q_mid = self.pool_query_mid.expand(B, -1, -1)
        pooled_mid, _   = self.pool_attn_mid(q_mid, mid_patches, mid_patches)
        pooled_mid = pooled_mid.squeeze(1)                                 # (B, 1024)

        # Final-layer pooling: learns semantic aggregation
        q_fin = self.pool_query_final.expand(B, -1, -1)
        pooled_final, _ = self.pool_attn_final(q_fin, final_patches, final_patches)
        pooled_final = pooled_final.squeeze(1)                             # (B, 1024)

        # Fuse: residual sum + LayerNorm
        fused = self.fuse_norm(pooled_mid + pooled_final)                  # (B, 1024)

        return self.proj(fused)                                            # (B, 768)


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
