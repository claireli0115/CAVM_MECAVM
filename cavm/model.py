"""Core CAVM architecture, extracted from the research implementation."""

import math
from typing import Sequence, Tuple, Union
import torch
from monai.networks import blocks
from monai.networks.nets import swin_unetr
from .transformer import ModelArgs, Transformer


class CAVM(swin_unetr.SwinUNETR):
    """SwinUNETR with cross attention."""

    def __init__(
        self,
        img_size: Union[Sequence[int], int] = (192, 192, 128),
        *args,
        num_heads: Sequence[int] = (3, 6, 12, 24),
        feature_size: int = 24,
        norm_name: Union[Tuple, str] = "instance",
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        spatial_dims: int = 3,
        **kwargs
    ) -> None:
        """ """
        img_size = tuple(img_size)
        if len(img_size) != 3 or any((s < 64 or s % 32 for s in img_size)):
            raise ValueError("Use three spatial dimensions >= 64, divisible by 32.")
        if math.prod(s // 16 for s in img_size) % 4:
            raise ValueError(
                "The stage-3 feature grid must contain a multiple of four voxels."
            )
        if spatial_dims != 3:
            raise ValueError("This core implementation supports only 3D volumes.")
        if kwargs.get("in_channels", 4) != 4 or kwargs.get("out_channels", 4) != 4:
            raise ValueError(
                "This core implementation requires four input and output channels."
            )
        kwargs.setdefault("in_channels", 4)
        kwargs.setdefault("out_channels", 4)
        super().__init__(
            img_size,
            *args,
            num_heads=num_heads,
            feature_size=feature_size,
            norm_name=norm_name,
            spatial_dims=spatial_dims,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            **kwargs
        )
        self.swinViT_0 = swin_unetr.SwinTransformer(
            in_chans=4,
            embed_dim=feature_size,
            window_size=self.swinViT.window_size,
            patch_size=self.swinViT.patch_size,
            depths=(2, 2, 2, 2),
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=0,
            norm_layer=torch.nn.LayerNorm,
            use_checkpoint=False,
            spatial_dims=spatial_dims,
            downsample="merging",
        )
        self.swinViT_0_0 = swin_unetr.SwinTransformer(
            in_chans=4,
            embed_dim=feature_size,
            window_size=self.swinViT.window_size,
            patch_size=self.swinViT.patch_size,
            depths=(2, 2, 2, 2),
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=0,
            norm_layer=torch.nn.LayerNorm,
            use_checkpoint=False,
            spatial_dims=spatial_dims,
            downsample="merging",
        )
        self.encoder1 = blocks.UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=4,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )
        self.encoderT0 = blocks.UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=16 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )
        self.encoderT1 = blocks.UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=8 * feature_size,
            out_channels=8 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )
        self.encoder11 = blocks.UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=8 * feature_size,
            out_channels=8 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )
        self.feature_size = feature_size
        self.img_size = img_size
        model_args4: ModelArgs = ModelArgs(
            dim=16 * feature_size,
            n_layers=4,
            n_heads=4,
            max_seq_len=math.prod((s // 32 for s in img_size)) * 3,
            seq_step=math.prod((s // 32 for s in img_size)),
        )
        self.model_llamalike4 = Transformer(model_args4)
        model_args3: ModelArgs = ModelArgs(
            dim=2 * 16 * feature_size,
            n_layers=8,
            n_heads=8,
            max_seq_len=math.prod((s // 16 for s in img_size)) // 4 * 3,
            seq_step=math.prod((s // 16 for s in img_size)) // 4,
        )
        self.model_llamalike3 = Transformer(model_args3)

    def forward_encoder(self, xs):
        """Encode features. list 3"""
        x_hiddens = self.swinViT_0_0(xs[0], self.normalize)
        xs_dec4 = [self.encoderT0(x_hiddens[4])]
        xs_dec3 = [self.encoderT1(x_hiddens[3])]
        for i in range(1, 3):
            x_hiddens = self.swinViT(xs[i], self.normalize)
            xs_dec4.append(self.encoder10(x_hiddens[4]))
            xs_dec3.append(self.encoder11(x_hiddens[3]))
        return (xs_dec3, xs_dec4)

    def forward_encoder_0(self, x):
        """Encode features."""
        x_hiddens = self.swinViT_0(x, self.normalize)
        x_enc0 = self.encoder1(x)
        x_enc1 = self.encoder2(x_hiddens[0])
        x_enc2 = self.encoder3(x_hiddens[1])
        x_enc3 = self.encoder4(x_hiddens[2])
        return {"enc0": x_enc0, "enc1": x_enc1, "enc2": x_enc2, "enc3": x_enc3}

    def forward_decoder(
        self, xs_encoded3: list, xs_encoded4: list, x0_encoded: torch.Tensor
    ) -> torch.Tensor:
        """Decode features."""
        xs_logits = []
        for i in range(len(xs_encoded3)):
            x_dec3 = self.decoder5(xs_encoded4[i], xs_encoded3[i])
            x_dec2 = self.decoder4(x_dec3, x0_encoded["enc3"])
            x_dec1 = self.decoder3(x_dec2, x0_encoded["enc2"])
            x_dec0 = self.decoder2(x_dec1, x0_encoded["enc1"])
            x_out = self.decoder1(x_dec0, x0_encoded["enc0"])
            xs_logits.append(self.out(x_out))
        return xs_logits

    def trans_encode_4(self, xs_encoded: list):
        xs_flatten = []
        for i in range(3):
            xs_flatten.append(xs_encoded[i].view(1, self.feature_size * 16, -1))
        xs_concatenated = torch.cat(tuple(xs_flatten), dim=2).transpose(1, 2)
        return xs_concatenated

    def trans_encode_3(self, xs_encoded: list):
        xs_flatten = []
        for i in range(3):
            xs_flatten.append(xs_encoded[i].view(1, self.feature_size * 16 * 2, -1))
        xs_concatenated = torch.cat(tuple(xs_flatten), dim=2).transpose(1, 2)
        return xs_concatenated

    def trans_decode_4(self, xs_calculated: torch.Tensor):
        xs_encoded = list(xs_calculated.transpose(1, 2).chunk(3, dim=2))
        for i in range(len(xs_encoded)):
            xs_encoded[i] = xs_encoded[i].reshape(
                1, int(self.feature_size * 16), *(s // 32 for s in self.img_size)
            )
        return xs_encoded

    def trans_decode_3(self, xs_calculated: torch.Tensor):
        xs_encoded = list(xs_calculated.transpose(1, 2).chunk(3, dim=2))
        for i in range(len(xs_encoded)):
            xs_encoded[i] = xs_encoded[i].reshape(
                1, int(self.feature_size * 8), *(s // 16 for s in self.img_size)
            )
        return xs_encoded

    def forward(self, x: list, x0: torch.Tensor) -> Sequence[torch.Tensor]:
        """Two views forward."""
        x_encoded3, x_encoded4 = self.forward_encoder(x)
        x0_encoded = self.forward_encoder_0(x0)
        x_encoded3 = self.trans_encode_3(x_encoded3)
        x_encoded4 = self.trans_encode_4(x_encoded4)
        x_encoded3 = self.model_llamalike3(x_encoded3, 0)
        x_encoded4 = self.model_llamalike4(x_encoded4, 0)
        x_encoded3 = self.trans_decode_3(x_encoded3)
        x_encoded4 = self.trans_decode_4(x_encoded4)
        return (
            x_encoded3,
            x_encoded4,
            self.forward_decoder(x_encoded3, x_encoded4, x0_encoded),
        )

    def forward_label(self, x: list) -> Sequence[torch.Tensor]:
        """Encode features. list 3"""
        with torch.no_grad():
            xs_dec4 = []
            xs_dec3 = []
            for i in range(3):
                x_hiddens = self.swinViT(x[i], self.normalize)
                xs_dec4.append(self.encoder10(x_hiddens[4]))
                xs_dec3.append(self.encoder11(x_hiddens[3]))
        return (xs_dec3, xs_dec4)

    def test(self, x: torch.Tensor, x0: torch.Tensor):
        x0_encoded = self.forward_encoder_0(x0)
        x_hiddens0 = self.swinViT_0_0(x, self.normalize)
        x_encoded0_4 = self.encoderT0(x_hiddens0[4])
        x_encoded0_3 = self.encoderT1(x_hiddens0[3])
        x_encoded_0_3 = [
            x_encoded0_3,
            torch.zeros_like(x_encoded0_3),
            torch.zeros_like(x_encoded0_3),
        ]
        x_encoded_0_3 = self.trans_encode_3(x_encoded_0_3)
        x_encoded_0_4 = [
            x_encoded0_4,
            torch.zeros_like(x_encoded0_4),
            torch.zeros_like(x_encoded0_4),
        ]
        x_encoded_0_4 = self.trans_encode_4(x_encoded_0_4)
        x_encoded_0_3 = self.model_llamalike3(x_encoded_0_3, 0)
        x_encoded_0_4 = self.model_llamalike4(x_encoded_0_4, 0)
        x_encoded1_3 = self.trans_decode_3(x_encoded_0_3)
        x_encoded1_4 = self.trans_decode_4(x_encoded_0_4)
        x_033 = self.forward_decoder(x_encoded1_3, x_encoded1_4, x0_encoded)[0]
        x_hiddens033 = self.swinViT(x_033, self.normalize)
        x_encoded1_4 = self.encoder10(x_hiddens033[4])
        x_encoded1_3 = self.encoder11(x_hiddens033[3])
        x_encoded_1_3 = [x_encoded0_3, x_encoded1_3, torch.zeros_like(x_encoded0_3)]
        x_encoded_1_3 = self.trans_encode_3(x_encoded_1_3)
        x_encoded_1_4 = [x_encoded0_4, x_encoded1_4, torch.zeros_like(x_encoded0_4)]
        x_encoded_1_4 = self.trans_encode_4(x_encoded_1_4)
        x_encoded_1_3 = self.model_llamalike3(x_encoded_1_3, 0)
        x_encoded_1_4 = self.model_llamalike4(x_encoded_1_4, 0)
        x_encoded2_3 = self.trans_decode_3(x_encoded_1_3)
        x_encoded2_4 = self.trans_decode_4(x_encoded_1_4)
        x_066 = self.forward_decoder(x_encoded2_3, x_encoded2_4, x0_encoded)[1]
        x_hiddens066 = self.swinViT(x_066, self.normalize)
        x_encoded2_4 = self.encoder10(x_hiddens066[4])
        x_encoded2_3 = self.encoder11(x_hiddens066[3])
        x_encoded_2_3 = [x_encoded0_3, x_encoded1_3, x_encoded2_3]
        x_encoded_2_3 = self.trans_encode_3(x_encoded_2_3)
        x_encoded_2_4 = [x_encoded0_4, x_encoded1_4, x_encoded2_4]
        x_encoded_2_4 = self.trans_encode_4(x_encoded_2_4)
        x_encoded_2_3 = self.model_llamalike3(x_encoded_2_3, 0)
        x_encoded_2_4 = self.model_llamalike4(x_encoded_2_4, 0)
        x_encoded_3 = self.trans_decode_3(x_encoded_2_3)
        x_encoded_4 = self.trans_decode_4(x_encoded_2_4)
        return (
            x_encoded_3,
            x_encoded_4,
            self.forward_decoder(x_encoded_3, x_encoded_4, x0_encoded),
        )
