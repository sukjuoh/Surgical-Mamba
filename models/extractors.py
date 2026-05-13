"""Frame-level feature extractors for surgical phase recognition."""

import torch
import torch.nn as nn


class ConvNeXtTinyExtractor(nn.Module):
    """ConvNeXt-Tiny / ConvNeXtV2-Tiny feature extractor via timm.

    Output: (B, 768) — global-average-pooled stage-4 features.

    Args:
        pretrained: if True, load ImageNet-1k weights from the timm hub.
        model_name: timm model name, e.g. "convnext_tiny" or "convnextv2_tiny".
    """

    def __init__(self, pretrained: bool = True, model_name: str = "convnext_tiny",
                 drop_path_rate: float = 0.0, grad_checkpointing: bool = False):
        super().__init__()
        import timm
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0,
            drop_path_rate=drop_path_rate,
        )
        self.num_features = self.backbone.num_features  # 768

        if grad_checkpointing:
            self.backbone.set_grad_checkpointing(enable=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone.forward_features(x)
        return feat.mean(dim=[-2, -1])
