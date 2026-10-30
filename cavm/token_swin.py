# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified: respect the caller's gradient mode in active branches.
import itertools
from typing import Optional, Sequence, Tuple, Type, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn import LayerNorm
from monai.networks.blocks import MLPBlock as Mlp
from monai.networks.blocks import (
    PatchEmbed,
    UnetOutBlock,
    UnetrBasicBlock,
    UnetrUpBlock,
)
from monai.networks.layers import DropPath, trunc_normal_
from monai.utils import ensure_tuple_rep, look_up_option, optional_import

rearrange, _ = optional_import("einops", name="rearrange")
__all__ = [
    "SwinUNETRTokenTune",
    "window_partition",
    "window_reverse",
    "WindowAttentionTokenTune",
    "SwinTransformerBlockTokenTune",
    "PatchMerging",
    "PatchMergingV2",
    "MERGING_MODE",
    "BasicLayerTokenTune",
    "SwinTransformerTokenTune",
]


class SwinUNETRTokenTune(nn.Module):
    """
    基于Swin UNETR的TokenTune版本，应用TOKENTUNE方法减少内存占用。
    原论文：Hatamizadeh et al., Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images
    TOKENTUNE论文：Simoulin et al., Memory-Efficient Fine-Tuning of Transformers via Token Selection
    """

    def __init__(
        self,
        img_size: Union[Sequence[int], int],
        in_channels: int,
        out_channels: int,
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        feature_size: int = 24,
        norm_name: Union[Tuple, str] = "instance",
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        normalize: bool = True,
        use_checkpoint: bool = False,
        spatial_dims: int = 3,
        prefix_ratio: float = 0.1,
        downsample="merging",
    ) -> None:
        """
        Args:
            img_size: 输入图像的尺寸。
            in_channels: 输入通道数。
            out_channels: 输出通道数。
            feature_size: 网络特征尺寸。
            depths: 每个阶段的层数。
            num_heads: 注意力头数。
            norm_name: 特征归一化类型。
            drop_rate: dropout率。
            attn_drop_rate: 注意力dropout率。
            dropout_path_rate: drop path率。
            normalize: 是否归一化输出特征。
            use_checkpoint: 是否使用梯度检查点以减少内存。
            spatial_dims: 空间维度（2或3）。
            downsample: 下采样模块类型。
            prefix_ratio: 选择prefix patches的比例（用于TOKENTUNE）。
        """
        super().__init__()
        img_size = ensure_tuple_rep(img_size, spatial_dims)
        patch_size = ensure_tuple_rep(2, spatial_dims)
        window_size = ensure_tuple_rep(7, spatial_dims)
        if spatial_dims not in (2, 3):
            raise ValueError("空间维度应为2或3。")
        for m, p in zip(img_size, patch_size):
            for i in range(5):
                if m % np.power(p, i + 1) != 0:
                    raise ValueError("输入图像尺寸应能被阶段分辨率整除。")
        if not 0 <= drop_rate <= 1:
            raise ValueError("dropout率应在0到1之间。")
        if not 0 <= attn_drop_rate <= 1:
            raise ValueError("注意力dropout率应在0到1之间。")
        if not 0 <= dropout_path_rate <= 1:
            raise ValueError("drop path率应在0到1之间。")
        if feature_size % 12 != 0:
            raise ValueError("特征尺寸应能被12整除。")
        self.normalize = normalize
        self.swinViT = SwinTransformerTokenTune(
            in_chans=in_channels,
            embed_dim=feature_size,
            window_size=window_size,
            patch_size=patch_size,
            depths=depths,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=dropout_path_rate,
            norm_layer=nn.LayerNorm,
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims,
            downsample=(
                look_up_option(downsample, MERGING_MODE)
                if isinstance(downsample, str)
                else downsample
            ),
            prefix_ratio=prefix_ratio,
        )
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )
        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )
        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=2 * feature_size,
            out_channels=2 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )
        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=4 * feature_size,
            out_channels=4 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )
        self.encoder10 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=16 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=8 * feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder1 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.out = UnetOutBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=out_channels,
        )

    def load_from(self, weights):
        with torch.no_grad():
            self.swinViT.patch_embed.proj.weight.copy_(
                weights["state_dict"]["module.patch_embed.proj.weight"]
            )
            self.swinViT.patch_embed.proj.bias.copy_(
                weights["state_dict"]["module.patch_embed.proj.bias"]
            )
            for bname, block in self.swinViT.layers1[0].blocks.named_children():
                block.load_from(weights, n_block=bname, layer="layers1")
            self.swinViT.layers1[0].downsample.reduction.weight.copy_(
                weights["state_dict"]["module.layers1.0.downsample.reduction.weight"]
            )
            self.swinViT.layers1[0].downsample.norm.weight.copy_(
                weights["state_dict"]["module.layers1.0.downsample.norm.weight"]
            )
            self.swinViT.layers1[0].downsample.norm.bias.copy_(
                weights["state_dict"]["module.layers1.0.downsample.norm.bias"]
            )
            for bname, block in self.swinViT.layers2[0].blocks.named_children():
                block.load_from(weights, n_block=bname, layer="layers2")
            self.swinViT.layers2[0].downsample.reduction.weight.copy_(
                weights["state_dict"]["module.layers2.0.downsample.reduction.weight"]
            )
            self.swinViT.layers2[0].downsample.norm.weight.copy_(
                weights["state_dict"]["module.layers2.0.downsample.norm.weight"]
            )
            self.swinViT.layers2[0].downsample.norm.bias.copy_(
                weights["state_dict"]["module.layers2.0.downsample.norm.bias"]
            )
            for bname, block in self.swinViT.layers3[0].blocks.named_children():
                block.load_from(weights, n_block=bname, layer="layers3")
            self.swinViT.layers3[0].downsample.reduction.weight.copy_(
                weights["state_dict"]["module.layers3.0.downsample.reduction.weight"]
            )
            self.swinViT.layers3[0].downsample.norm.weight.copy_(
                weights["state_dict"]["module.layers3.0.downsample.norm.weight"]
            )
            self.swinViT.layers3[0].downsample.norm.bias.copy_(
                weights["state_dict"]["module.layers3.0.downsample.norm.bias"]
            )
            for bname, block in self.swinViT.layers4[0].blocks.named_children():
                block.load_from(weights, n_block=bname, layer="layers4")
            self.swinViT.layers4[0].downsample.reduction.weight.copy_(
                weights["state_dict"]["module.layers4.0.downsample.reduction.weight"]
            )
            self.swinViT.layers4[0].downsample.norm.weight.copy_(
                weights["state_dict"]["module.layers4.0.downsample.norm.weight"]
            )
            self.swinViT.layers4[0].downsample.norm.bias.copy_(
                weights["state_dict"]["module.layers4.0.downsample.norm.bias"]
            )

    def forward(self, x_in):
        hidden_states_out = self.swinViT(x_in, self.normalize)
        enc0 = self.encoder1(x_in)
        enc1 = self.encoder2(hidden_states_out[0])
        enc2 = self.encoder3(hidden_states_out[1])
        enc3 = self.encoder4(hidden_states_out[2])
        dec4 = self.encoder10(hidden_states_out[4])
        dec3 = self.decoder5(dec4, hidden_states_out[3])
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0, enc0)
        logits = self.out(out)
        return logits


def window_partition(x, window_size):
    """窗口分区操作"""
    x_shape = x.size()
    if len(x_shape) == 5:
        b, d, h, w, c = x_shape
        x = x.view(
            b,
            d // window_size[0],
            window_size[0],
            h // window_size[1],
            window_size[1],
            w // window_size[2],
            window_size[2],
            c,
        )
        windows = (
            x.permute(0, 1, 3, 5, 2, 4, 6, 7)
            .contiguous()
            .view(-1, window_size[0] * window_size[1] * window_size[2], c)
        )
    elif len(x_shape) == 4:
        b, h, w, c = x_shape
        x = x.view(
            b,
            h // window_size[0],
            window_size[0],
            w // window_size[1],
            window_size[1],
            c,
        )
        windows = (
            x.permute(0, 1, 3, 2, 4, 5)
            .contiguous()
            .view(-1, window_size[0] * window_size[1], c)
        )
    return windows


def window_reverse(windows, window_size, dims):
    """窗口反转操作"""
    if len(dims) == 4:
        b, d, h, w = dims
        x = windows.view(
            b,
            d // window_size[0],
            h // window_size[1],
            w // window_size[2],
            window_size[0],
            window_size[1],
            window_size[2],
            -1,
        )
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(b, d, h, w, -1)
    elif len(dims) == 3:
        b, h, w = dims
        x = windows.view(
            b,
            h // window_size[0],
            w // window_size[1],
            window_size[0],
            window_size[1],
            -1,
        )
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x


def get_window_size(x_size, window_size, shift_size=None):
    """计算窗口大小"""
    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0
    if shift_size is None:
        return tuple(use_window_size)
    else:
        return (tuple(use_window_size), tuple(use_shift_size))


class MlpTokenTune(Mlp):

    def forward(self, x, hg_mask=None):
        """
        x: [B, D, H, W, C] (已 reshape 回空间维度)
        hg_mask: [B, D, H, W] (bool)
        """
        if hg_mask is None:
            return super().forward(x)
        b, *spatial, c = x.shape
        x_flat = x.reshape(-1, c)
        mask_flat = hg_mask.reshape(-1)
        x_active = x_flat[mask_flat]
        x_frozen = x_flat[~mask_flat]
        with torch.set_grad_enabled(torch.is_grad_enabled()):
            x_a = self.fn(self.linear1(x_active))
            x_a = self.drop1(x_a)
            x_a = self.linear2(x_a)
            x_a = self.drop2(x_a)
        with torch.no_grad():
            x_f = self.fn(self.linear1(x_frozen))
            x_f = self.drop1(x_f)
            x_f = self.linear2(x_f)
            x_f = self.drop2(x_f)
        x_out_flat = torch.zeros_like(x_flat)
        x_out_flat[mask_flat] = x_a
        x_out_flat[~mask_flat] = x_f
        return x_out_flat.reshape(b, *spatial, c)


class WindowAttentionTokenTune(nn.Module):
    """
    支持TOKENTUNE的窗口注意力模块
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Sequence[int],
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        prefix_ratio: float = 0.1,
    ) -> None:
        """
        Args:
            dim: 特征通道数。
            num_heads: 注意力头数。
            window_size: 局部窗口大小。
            qkv_bias: 是否为查询、键、值添加可学习偏置。
            attn_drop: 注意力dropout率。
            proj_drop: 输出dropout率。
            prefix_ratio: 选择prefix patches的比例。
        """
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** (-0.5)
        self.prefix_ratio = prefix_ratio
        mesh_args = torch.meshgrid.__kwdefaults__
        if len(self.window_size) == 3:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(
                    (2 * self.window_size[0] - 1)
                    * (2 * self.window_size[1] - 1)
                    * (2 * self.window_size[2] - 1),
                    num_heads,
                )
            )
            coords_d = torch.arange(self.window_size[0])
            coords_h = torch.arange(self.window_size[1])
            coords_w = torch.arange(self.window_size[2])
            if mesh_args is not None:
                coords = torch.stack(
                    torch.meshgrid(coords_d, coords_h, coords_w, indexing="ij")
                )
            else:
                coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.window_size[0] - 1
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 2] += self.window_size[2] - 1
            relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (
                2 * self.window_size[2] - 1
            )
            relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1
        elif len(self.window_size) == 2:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(
                    (2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads
                )
            )
            coords_h = torch.arange(self.window_size[0])
            coords_w = torch.arange(self.window_size[1])
            if mesh_args is not None:
                coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
            else:
                coords = torch.stack(torch.meshgrid(coords_h, coords_w))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.window_size[0] - 1
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask, hg_mask):
        """
        x: [B*Nw, N, C]
        mask: Swin Shift Mask [Nw, N, N]
        hg_mask: [B*Nw, N] (bool) 每一行Active数量必须一致！
        """
        B_Nw, N, C = x.shape
        x_flat = x.reshape(-1, C)
        hg_mask_flat = hg_mask.reshape(-1)
        x_active = x_flat[hg_mask_flat]
        x_frozen = x_flat[~hg_mask_flat]
        with torch.set_grad_enabled(torch.is_grad_enabled()):
            qkv_active = self.qkv(x_active)
        with torch.no_grad():
            qkv_frozen = self.qkv(x_frozen)
        qkv_flat = torch.zeros(B_Nw * N, 3 * C, dtype=x.dtype, device=x.device)
        qkv_flat[hg_mask_flat] = qkv_active
        qkv_flat[~hg_mask_flat] = qkv_frozen
        qkv = qkv_flat.reshape(B_Nw, N, 3, self.num_heads, C // self.num_heads).permute(
            2, 0, 3, 1, 4
        )
        q, k, v = (qkv[0], qkv[1], qkv[2])
        k_all, v_all = (k, v)
        relative_position_bias = (
            self.relative_position_bias_table[
                self.relative_position_index.clone()[:N, :N].reshape(-1)
            ]
            .reshape(N, N, -1)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .expand(B_Nw, -1, -1, -1)
        )
        if mask is not None:
            nw = mask.shape[0]
            mask_expanded = mask.unsqueeze(1).repeat(B_Nw // nw, 1, 1, 1)
        else:
            mask_expanded = None
        sorted_indices = torch.argsort(hg_mask.int(), dim=1, descending=True)
        num_active = hg_mask[0].sum().item()
        active_indices = sorted_indices[:, :num_active]
        frozen_indices = sorted_indices[:, num_active:]

        def expand_idx_q(idx):
            return idx.view(B_Nw, 1, -1, 1).expand(-1, self.num_heads, -1, q.shape[-1])

        def expand_idx_attn(idx):
            return idx.view(B_Nw, 1, -1, 1).expand(-1, self.num_heads, -1, N)

        with torch.set_grad_enabled(torch.is_grad_enabled()):
            q_act = torch.gather(q, 2, expand_idx_q(active_indices)) * self.scale
            bias_act = torch.gather(
                relative_position_bias, 2, expand_idx_attn(active_indices)
            )
            attn_act = q_act @ k_all.transpose(-2, -1) + bias_act
            if mask_expanded is not None:
                mask_act = torch.gather(
                    mask_expanded,
                    2,
                    active_indices.view(B_Nw, 1, -1, 1).expand(-1, 1, -1, N),
                )
                attn_act = attn_act + mask_act
            attn_act = self.softmax(attn_act)
            attn_act = self.attn_drop(attn_act)
            x_act = attn_act @ v_all
        with torch.no_grad():
            q_frz = torch.gather(q, 2, expand_idx_q(frozen_indices)) * self.scale
            bias_frz = torch.gather(
                relative_position_bias, 2, expand_idx_attn(frozen_indices)
            )
            attn_frz = q_frz @ k_all.transpose(-2, -1) + bias_frz
            if mask_expanded is not None:
                mask_frz = torch.gather(
                    mask_expanded,
                    2,
                    frozen_indices.view(B_Nw, 1, -1, 1).expand(-1, 1, -1, N),
                )
                attn_frz = attn_frz + mask_frz
            attn_frz = self.softmax(attn_frz)
            attn_frz = self.attn_drop(attn_frz)
            x_frz = attn_frz @ v_all
        x_out = torch.zeros_like(q)
        x_out.scatter_(2, expand_idx_q(active_indices), x_act)
        x_out.scatter_(2, expand_idx_q(frozen_indices), x_frz)
        x_out = x_out.transpose(1, 2).reshape(B_Nw, N, C)
        x_out = self.proj(x_out)
        return self.proj_drop(x_out)


class SwinTransformerBlockTokenTune(nn.Module):
    """
    支持TOKENTUNE方法的Swin Transformer块
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Sequence[int],
        shift_size: Sequence[int],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: str = "GELU",
        norm_layer: Type[LayerNorm] = nn.LayerNorm,
        prefix_ratio: float = 0.1,
        use_checkpoint: bool = False,
    ) -> None:
        """
        Args:
            dim: 特征通道数。
            num_heads: 注意力头数。
            window_size: 局部窗口大小。
            shift_size: 窗口移位大小。
            mlp_ratio: MLP隐藏层与嵌入层的比例。
            qkv_bias: 是否为查询、键、值添加偏置。
            drop: dropout率。
            attn_drop: 注意力dropout率。
            drop_path: 随机深度率。
            act_layer: 激活层类型。
            norm_layer: 归一化层类型。
            prefix_ratio: 选择prefix patches的比例。
            use_checkpoint: 是否使用梯度检查点。
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint
        self.norm1 = norm_layer(dim)
        self.prefix_ratio = prefix_ratio
        self.attn = WindowAttentionTokenTune(
            dim,
            window_size=self.window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            prefix_ratio=prefix_ratio,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MlpTokenTune(
            hidden_size=dim,
            mlp_dim=mlp_hidden_dim,
            act=act_layer,
            dropout_rate=drop,
            dropout_mode="swin",
        )

    def forward_split_norm(self, norm_layer, x, hg_mask_spatial):
        b, *spatial, c = x.shape
        x_flat = x.reshape(-1, c)
        mask_flat = hg_mask_spatial.reshape(-1)
        if mask_flat.device != x_flat.device:
            mask_flat = mask_flat.to(x_flat.device)
        x_active = x_flat[mask_flat]
        x_frozen = x_flat[~mask_flat]
        with torch.set_grad_enabled(torch.is_grad_enabled()):
            out_active = norm_layer(x_active)
        with torch.no_grad():
            out_frozen = norm_layer(x_frozen)
        x_out_flat = torch.zeros_like(x_flat)
        x_out_flat[mask_flat] = out_active
        x_out_flat[~mask_flat] = out_frozen
        return x_out_flat.reshape(b, *spatial, c)

    def forward(self, x, mask_matrix, prefix_ratio=0.1):
        """
        x: [B, C, D, H, W] or [B, C, H, W]
        """
        x_shape = x.size()
        x_norm = self.norm1(x)
        if len(x_shape) == 5:
            b, d, h, w, c = x_shape
            window_size, shift_size = get_window_size(
                (d, h, w), self.window_size, self.shift_size
            )
            pad_l = pad_t = pad_d0 = 0
            pad_d1 = (window_size[0] - d % window_size[0]) % window_size[0]
            pad_b = (window_size[1] - h % window_size[1]) % window_size[1]
            pad_r = (window_size[2] - w % window_size[2]) % window_size[2]
            x_norm = F.pad(x_norm, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
            _, dp, hp, wp, _ = x_norm.shape
            dims = [b, dp, hp, wp]
        else:
            b, h, w, c = x_shape
            window_size, shift_size = get_window_size(
                (h, w), self.window_size, self.shift_size
            )
            pad_l = pad_t = 0
            pad_b = (window_size[0] - h % window_size[0]) % window_size[0]
            pad_r = (window_size[1] - w % window_size[1]) % window_size[1]
            x_norm = F.pad(x_norm, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, hp, wp, _ = x_norm.shape
            dims = [b, hp, wp]
        if any((i > 0 for i in shift_size)):
            if len(x_shape) == 5:
                shifted_x = torch.roll(
                    x_norm,
                    shifts=(-shift_size[0], -shift_size[1], -shift_size[2]),
                    dims=(1, 2, 3),
                )
            else:
                shifted_x = torch.roll(
                    x_norm, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2)
                )
            attn_mask = mask_matrix
        else:
            shifted_x = x_norm
            attn_mask = None
        x_windows = window_partition(shifted_x, window_size)
        B_Nw, N, _ = x_windows.shape
        num_active = int(N * prefix_ratio)
        if num_active == 0:
            num_active = 1
        mask_template = torch.zeros(B_Nw, N, dtype=torch.bool, device=x.device)
        mask_template[:, :num_active] = True
        rand_idx = torch.rand(B_Nw, N, device=x.device).argsort(dim=1)
        hg_mask_windows = torch.gather(mask_template, 1, rand_idx)
        attn_windows = self.attn(x_windows, mask=attn_mask, hg_mask=hg_mask_windows)
        attn_windows = attn_windows.view(-1, *window_size + (c,))
        shifted_x = window_reverse(attn_windows, window_size, dims)
        hg_mask_spatial_shifted = window_reverse(
            hg_mask_windows.unsqueeze(-1), window_size, dims
        ).squeeze(-1)
        if any((i > 0 for i in shift_size)):
            if len(x_shape) == 5:
                x_attn = torch.roll(
                    shifted_x,
                    shifts=(shift_size[0], shift_size[1], shift_size[2]),
                    dims=(1, 2, 3),
                )
                hg_mask_spatial = torch.roll(
                    hg_mask_spatial_shifted,
                    shifts=(shift_size[0], shift_size[1], shift_size[2]),
                    dims=(1, 2, 3),
                )
            else:
                x_attn = torch.roll(
                    shifted_x, shifts=(shift_size[0], shift_size[1]), dims=(1, 2)
                )
                hg_mask_spatial = torch.roll(
                    hg_mask_spatial_shifted,
                    shifts=(shift_size[0], shift_size[1]),
                    dims=(1, 2),
                )
        else:
            x_attn = shifted_x
            hg_mask_spatial = hg_mask_spatial_shifted
        if len(x_shape) == 5:
            if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
                x_attn = x_attn[:, :d, :h, :w, :].contiguous()
                hg_mask_spatial = hg_mask_spatial[:, :d, :h, :w].contiguous()
        elif pad_r > 0 or pad_b > 0:
            x_attn = x_attn[:, :h, :w, :].contiguous()
            hg_mask_spatial = hg_mask_spatial[:, :h, :w].contiguous()
        x = x + self.drop_path(x_attn)
        x_mlp = self.mlp(
            self.forward_split_norm(self.norm2, x, hg_mask_spatial),
            hg_mask=hg_mask_spatial,
        )
        x = x + self.drop_path(x_mlp)
        return x

    def load_from(self, weights, n_block, layer):
        root = f"module.{layer}.0.blocks.{n_block}."
        block_names = [
            "norm1.weight",
            "norm1.bias",
            "attn.relative_position_bias_table",
            "attn.relative_position_index",
            "attn.qkv.weight",
            "attn.qkv.bias",
            "attn.proj.weight",
            "attn.proj.bias",
            "norm2.weight",
            "norm2.bias",
            "mlp.fc1.weight",
            "mlp.fc1.bias",
            "mlp.fc2.weight",
            "mlp.fc2.bias",
        ]
        with torch.no_grad():
            self.norm1.weight.copy_(weights["state_dict"][root + block_names[0]])
            self.norm1.bias.copy_(weights["state_dict"][root + block_names[1]])
            self.attn.relative_position_bias_table.copy_(
                weights["state_dict"][root + block_names[2]]
            )
            self.attn.relative_position_index.copy_(
                weights["state_dict"][root + block_names[3]]
            )
            self.attn.qkv.weight.copy_(weights["state_dict"][root + block_names[4]])
            self.attn.qkv.bias.copy_(weights["state_dict"][root + block_names[5]])
            self.attn.proj.weight.copy_(weights["state_dict"][root + block_names[6]])
            self.attn.proj.bias.copy_(weights["state_dict"][root + block_names[7]])
            self.norm2.weight.copy_(weights["state_dict"][root + block_names[8]])
            self.norm2.bias.copy_(weights["state_dict"][root + block_names[9]])
            self.mlp.linear1.weight.copy_(weights["state_dict"][root + block_names[10]])
            self.mlp.linear1.bias.copy_(weights["state_dict"][root + block_names[11]])
            self.mlp.linear2.weight.copy_(weights["state_dict"][root + block_names[12]])
            self.mlp.linear2.bias.copy_(weights["state_dict"][root + block_names[13]])


class PatchMergingV2(nn.Module):
    """Patch合并层V2版本"""

    def __init__(
        self,
        dim: int,
        norm_layer: Type[LayerNorm] = nn.LayerNorm,
        spatial_dims: int = 3,
    ) -> None:
        super().__init__()
        self.dim = dim
        if spatial_dims == 3:
            self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
            self.norm = norm_layer(8 * dim)
        elif spatial_dims == 2:
            self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
            self.norm = norm_layer(4 * dim)

    def forward(self, x):
        x_shape = x.size()
        if len(x_shape) == 5:
            b, d, h, w, c = x_shape
            pad_input = h % 2 == 1 or w % 2 == 1 or d % 2 == 1
            if pad_input:
                x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2, 0, d % 2))
            x = torch.cat(
                [
                    x[:, i::2, j::2, k::2, :]
                    for i, j, k in itertools.product(range(2), range(2), range(2))
                ],
                -1,
            )
        elif len(x_shape) == 4:
            b, h, w, c = x_shape
            pad_input = h % 2 == 1 or w % 2 == 1
            if pad_input:
                x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2))
            x = torch.cat(
                [x[:, j::2, i::2, :] for i, j in itertools.product(range(2), range(2))],
                -1,
            )
        x = self.norm(x)
        x = self.reduction(x)
        return x


class PatchMerging(PatchMergingV2):
    """原始Patch合并层"""

    def forward(self, x):
        x_shape = x.size()
        if len(x_shape) == 4:
            return super().forward(x)
        if len(x_shape) != 5:
            raise ValueError(f"期待5D输入，得到{x.shape}。")
        b, d, h, w, c = x_shape
        pad_input = h % 2 == 1 or w % 2 == 1 or d % 2 == 1
        if pad_input:
            x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2, 0, d % 2))
        x0 = x[:, 0::2, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, 0::2, :]
        x3 = x[:, 0::2, 0::2, 1::2, :]
        x4 = x[:, 1::2, 0::2, 1::2, :]
        x5 = x[:, 0::2, 1::2, 0::2, :]
        x6 = x[:, 0::2, 0::2, 1::2, :]
        x7 = x[:, 1::2, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)
        x = self.norm(x)
        x = self.reduction(x)
        return x


MERGING_MODE = {"merging": PatchMerging, "mergingv2": PatchMergingV2}


def compute_mask(dims, window_size, shift_size, device):
    """计算区域掩码"""
    cnt = 0
    if len(dims) == 3:
        d, h, w = dims
        img_mask = torch.zeros((1, d, h, w, 1), device=device)
        for d in (
            slice(-window_size[0]),
            slice(-window_size[0], -shift_size[0]),
            slice(-shift_size[0], None),
        ):
            for h in (
                slice(-window_size[1]),
                slice(-window_size[1], -shift_size[1]),
                slice(-shift_size[1], None),
            ):
                for w in (
                    slice(-window_size[2]),
                    slice(-window_size[2], -shift_size[2]),
                    slice(-shift_size[2], None),
                ):
                    img_mask[:, d, h, w, :] = cnt
                    cnt += 1
    elif len(dims) == 2:
        h, w = dims
        img_mask = torch.zeros((1, h, w, 1), device=device)
        for h in (
            slice(-window_size[0]),
            slice(-window_size[0], -shift_size[0]),
            slice(-shift_size[0], None),
        ):
            for w in (
                slice(-window_size[1]),
                slice(-window_size[1], -shift_size[1]),
                slice(-shift_size[1], None),
            ):
                img_mask[:, h, w, :] = cnt
                cnt += 1
    mask_windows = window_partition(img_mask, window_size)
    mask_windows = mask_windows.squeeze(-1)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
        attn_mask == 0, float(0.0)
    )
    return attn_mask


class BasicLayerTokenTune(nn.Module):
    """
    支持TOKENTUNE的基础Swin Transformer层
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: Sequence[int],
        drop_path: list,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        norm_layer: Type[LayerNorm] = nn.LayerNorm,
        downsample: Optional[nn.Module] = None,
        prefix_ratio: float = 0.1,
        use_checkpoint: bool = False,
    ) -> None:
        """
        Args:
            dim: 特征通道数。
            depth: 每个阶段的层数。
            num_heads: 注意力头数。
            window_size: 局部窗口大小。
            drop_path: 随机深度率。
            mlp_ratio: MLP隐藏层比例。
            qkv_bias: 是否添加偏置。
            drop: dropout率。
            attn_drop: 注意力dropout率。
            norm_layer: 归一化层。
            downsample: 可选的下采样层。
            prefix_ratio: 选择prefix patches的比例。
            use_checkpoint: 是否使用梯度检查点。
        """
        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple((i // 2 for i in window_size))
        self.no_shift = tuple((0 for i in window_size))
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlockTokenTune(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=self.window_size,
                    shift_size=self.no_shift if i % 2 == 0 else self.shift_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=(
                        drop_path[i] if isinstance(drop_path, list) else drop_path
                    ),
                    norm_layer=norm_layer,
                    prefix_ratio=prefix_ratio,
                    use_checkpoint=use_checkpoint,
                )
                for i in range(depth)
            ]
        )
        self.downsample = downsample
        if callable(self.downsample):
            self.downsample = downsample(
                dim=dim, norm_layer=norm_layer, spatial_dims=len(self.window_size)
            )

    def forward(self, x, prefix_ratio):
        x_shape = x.size()
        if len(x_shape) == 5:
            b, c, d, h, w = x_shape
            window_size, shift_size = get_window_size(
                (d, h, w), self.window_size, self.shift_size
            )
            x = rearrange(x, "b c d h w -> b d h w c")
            dp = int(np.ceil(d / window_size[0])) * window_size[0]
            hp = int(np.ceil(h / window_size[1])) * window_size[1]
            wp = int(np.ceil(w / window_size[2])) * window_size[2]
            attn_mask = compute_mask([dp, hp, wp], window_size, shift_size, x.device)
            for blk in self.blocks:
                x = blk(x, attn_mask, prefix_ratio)
            x = x.view(b, d, h, w, -1)
            if self.downsample is not None:
                x = self.downsample(x)
            x = rearrange(x, "b d h w c -> b c d h w")
        elif len(x_shape) == 4:
            b, c, h, w = x_shape
            window_size, shift_size = get_window_size(
                (h, w), self.window_size, self.shift_size
            )
            x = rearrange(x, "b c h w -> b h w c")
            hp = int(np.ceil(h / window_size[0])) * window_size[0]
            wp = int(np.ceil(w / window_size[1])) * window_size[1]
            attn_mask = compute_mask([hp, wp], window_size, shift_size, x.device)
            for blk in self.blocks:
                x = blk(x, attn_mask, prefix_ratio)
            x = x.view(b, h, w, -1)
            if self.downsample is not None:
                x = self.downsample(x)
            x = rearrange(x, "b h w c -> b c h w")
        return x


class SwinTransformerTokenTune(nn.Module):
    """
    支持TOKENTUNE方法的Swin Transformer
    """

    def __init__(
        self,
        in_chans: int,
        embed_dim: int,
        window_size: Sequence[int],
        patch_size: Sequence[int],
        depths: Sequence[int],
        num_heads: Sequence[int],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: Type[LayerNorm] = nn.LayerNorm,
        patch_norm: bool = False,
        use_checkpoint: bool = False,
        spatial_dims: int = 3,
        downsample="merging",
        prefix_ratio: float = 0.1,
    ) -> None:
        """
        Args:
            in_chans: 输入通道数。
            embed_dim: 线性投影输出通道数。
            window_size: 局部窗口大小。
            patch_size: patch大小。
            depths: 每个阶段的层数。
            num_heads: 注意力头数。
            mlp_ratio: MLP隐藏层比例。
            qkv_bias: 是否添加偏置。
            drop_rate: dropout率。
            attn_drop_rate: 注意力dropout率。
            drop_path_rate: 随机深度率。
            norm_layer: 归一化层。
            patch_norm: 是否在patch嵌入后归一化。
            use_checkpoint: 是否使用梯度检查点。
            spatial_dims: 空间维度。
            downsample: 下采样模块类型。
            prefix_ratio: prefix patches比例。
        """
        super().__init__()
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.window_size = window_size
        self.patch_size = patch_size
        self.prefix_ratio = prefix_ratio
        self.patch_embed = PatchEmbed(
            patch_size=self.patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
            spatial_dims=spatial_dims,
        )
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers1 = nn.ModuleList()
        self.layers2 = nn.ModuleList()
        self.layers3 = nn.ModuleList()
        self.layers4 = nn.ModuleList()
        down_sample_mod = (
            look_up_option(downsample, MERGING_MODE)
            if isinstance(downsample, str)
            else downsample
        )
        for i_layer in range(self.num_layers):
            layer = BasicLayerTokenTune(
                dim=int(embed_dim * 2**i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=self.window_size,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                norm_layer=norm_layer,
                downsample=down_sample_mod,
                prefix_ratio=prefix_ratio,
                use_checkpoint=use_checkpoint,
            )
            if i_layer == 0:
                self.layers1.append(layer)
            elif i_layer == 1:
                self.layers2.append(layer)
            elif i_layer == 2:
                self.layers3.append(layer)
            elif i_layer == 3:
                self.layers4.append(layer)
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))

    def proj_out(self, x, normalize=False):
        if normalize:
            x_shape = x.size()
            if len(x_shape) == 5:
                n, ch, d, h, w = x_shape
                x = rearrange(x, "n c d h w -> n d h w c")
                x = F.layer_norm(x, [ch])
                x = rearrange(x, "n d h w c -> n c d h w")
            elif len(x_shape) == 4:
                n, ch, h, w = x_shape
                x = rearrange(x, "n c h w -> n h w c")
                x = F.layer_norm(x, [ch])
                x = rearrange(x, "n h w c -> n c h w")
        return x

    def forward(self, x, normalize=True):
        x0 = self.patch_embed(x)
        x0 = self.pos_drop(x0)
        x0_out = self.proj_out(x0, normalize)
        x1 = self.layers1[0](x0.contiguous(), self.prefix_ratio)
        x1_out = self.proj_out(x1, normalize)
        x2 = self.layers2[0](x1.contiguous(), self.prefix_ratio)
        x2_out = self.proj_out(x2, normalize)
        x3 = self.layers3[0](x2.contiguous(), self.prefix_ratio)
        x3_out = self.proj_out(x3, normalize)
        x4 = self.layers4[0](x3.contiguous(), self.prefix_ratio)
        x4_out = self.proj_out(x4, normalize)
        return [x0_out, x1_out, x2_out, x3_out, x4_out]
