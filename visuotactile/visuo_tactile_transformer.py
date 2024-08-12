import math
from functools import partial
from typing import Dict, List

import torch
import torch.nn as nn

# from tensordict import TensorDict
from visual.utils import trunc_normal_
from visual.vision_transformer import Block, Mlp


class PatchEmbed(nn.Module):
    """Input to Patch Embedding"""

    def __init__(
        self,
        input_size,
        patch_size,
        embed_dim=768,
    ):
        super().__init__()
        self.size = input_size
        self.patch_size = patch_size
        if len(self.size) == 3:  # image input
            self.num_patches = (input_size[1] // patch_size) * (input_size[2] // patch_size)
            in_channels = self.size[0]
            self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        elif len(self.size) == 1:  # tactile input
            self.num_patches = input_size[0] // patch_size
            self.proj = nn.Conv1d(1, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        if len(self.size) == 3:
            B, C, H, W = x.shape
            x = self.proj(x).flatten(2).transpose(1, 2)
        if len(self.size) == 1:
            x = x.unsqueeze(1)  # add a channel dimension
            x = self.proj(x).transpose(1, 2)
        return x


class VisuoTactileTransformer(nn.Module):

    def __init__(
        self,
        input_sizes,
        patch_sizes=[16, 16, 16, 2],
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        **kwargs,
    ):
        super().__init__()
        self.input_sizes = input_sizes
        self.patch_sizes = patch_sizes
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = nn.ModuleList()
        self.pos_embed = nn.ParameterList()
        for i, input_size in enumerate(self.input_sizes):
            patch_embed = PatchEmbed(input_size=input_size, patch_size=self.patch_sizes[i], embed_dim=embed_dim)
            self.patch_embed.append(patch_embed)
            pos_embed = nn.Parameter(torch.zeros(1, patch_embed.num_patches, embed_dim))
            self.pos_embed.append(pos_embed)

        # num_patches = sum([pe.num_patches for pe in self.patch_embed])

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token_pos_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        for pos_embed in self.pos_embed:
            trunc_normal_(pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, w, h, pos_embed):
        npatch = x.shape[1] - 1
        N = pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return pos_embed
        patch_pos_embed = pos_embed
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode="bicubic",
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return patch_pos_embed

    def prepare_tokens(self, x: List):
        B = x[0].shape[0]
        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = [cls_tokens + self.cls_token_pos_embed]
        for k, xk in enumerate(x):
            shape = xk.shape
            xk = self.patch_embed[k](xk)  # patch linear embedding
            # add positional encoding to each token (depends on dim of xk)
            if len(shape) == 3:
                # interpolate pos encoding for image input, resolution at inference time can be different
                _, nc, w, h = shape
                xk = xk + self.interpolate_pos_encoding(xk, w, h, self.pos_embed[k])
            if len(shape) == 1:
                # the size of 1D inputs does not change so there is no need to interpolate
                xk = xk + self.pos_embed[k]
            tokens.append(xk)

        x = torch.cat(tokens, dim=1)
        return self.pos_drop(x)

    def forward(self, x: List):
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]


def vitact_tiny(input_sizes, patch_size=16, **kwargs):
    model = VisuoTactileTransformer(
        input_sizes=input_sizes,
        patch_size=patch_size,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def vitact_small(input_sizes, patch_size=16, **kwargs):
    model = VisuoTactileTransformer(
        input_sizes=input_sizes,
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def vitact_base(input_sizes, patch_size=16, **kwargs):
    model = VisuoTactileTransformer(
        input_sizes=input_sizes,
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model
