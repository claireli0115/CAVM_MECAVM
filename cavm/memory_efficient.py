"""ME-CAVM: shared CAVM generation with token-split feature encoding."""

import torch
from .model import CAVM
from .token_swin import SwinTransformerTokenTune


class MECAVM(CAVM):
    """Replace only the FCE Swin encoder, as in the research implementation."""

    def __init__(
        self, img_size=(192, 192, 128), prefix_ratio=0.1, feature_size=24, **kwargs
    ):
        if not 0 < prefix_ratio <= 1:
            raise ValueError("prefix_ratio must lie in (0, 1].")
        super().__init__(img_size=img_size, feature_size=feature_size, **kwargs)
        self.swinViT = SwinTransformerTokenTune(
            in_chans=4,
            embed_dim=feature_size,
            window_size=self.swinViT.window_size,
            patch_size=self.swinViT.patch_size,
            depths=kwargs.get("depths", (2, 2, 2, 2)),
            num_heads=kwargs.get("num_heads", (3, 6, 12, 24)),
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=kwargs.get("drop_rate", 0.0),
            attn_drop_rate=kwargs.get("attn_drop_rate", 0.0),
            drop_path_rate=kwargs.get("dropout_path_rate", 0.0),
            norm_layer=torch.nn.LayerNorm,
            use_checkpoint=False,
            spatial_dims=3,
            downsample="merging",
            prefix_ratio=prefix_ratio,
        )
