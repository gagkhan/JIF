import math
from functools import partial
from typing import Dict

import torch
import torch.nn as nn
from tensordict import TensorDict
from visual.transformer import *
from visual.utils import trunc_normal_


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(
        self,
        dims=2,
        size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
    ):
        super().__init__()
        self.dims = dims
        num_patches = (size // patch_size) ** dims
        self.size = size  # size along first dim
        self.patch_size = patch_size
        self.num_patches = num_patches

        if self.dims == 2:
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        elif self.dims == 1:
            self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: TensorDict):
        if self.dims == 2:
            B, C, H, W = x.shape
            x = self.proj(x).flatten(2).transpose(1, 2)
        elif self.dims == 1:
            x = self.proj(x).transpose(1, 2)
        return x


class MultimodalTransformer(nn.Module):

    def __init__(
        self,
        nmod=4,
        keys=["rgb1", "rgb2", "sensor1", "sensor2"],
        sizes=[224, 224, 10, 2],
        dims=[2, 2, 1, 1],
        channels=[3, 3, 1, 1],
        patch_sizes=[16, 16, 5, 2],
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
        self.nmod = 4
        self.sizes = sizes
        self.dims = dims
        self.channels = channels
        self.patch_sizes = patch_sizes
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = nn.ModuleList()
        self.pos_embed = nn.ParameterList()
        for i in range(nmod):
            size, dim, in_chans, patch_size = sizes[i], dims[i], channels[i], patch_sizes[i]
            patch_embed = PatchEmbed(dims=dim, size=size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
            self.patch_embed.append(patch_embed)
            pos_embed = nn.Parameter(torch.zeros(1, patch_embed.num_patches, embed_dim))
            self.pos_embed.append(pos_embed)

        num_patches = sum([pe.num_patches for pe in self.patch_embed])

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

    def prepare_tokens(self, x: TensorDict):
        B = x.batch_size[0]
        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = [cls_tokens + self.cls_token_pos_embed]
        for j, (k, xk) in enumerate(x.items()):
            shape = xk.shape
            xk = self.patch_embed[j](xk)  # patch linear embedding
            # add positional encoding to each token
            if self.dims[j] == 2:
                # interpolate pos encoding for 2D inputs, resolution at inference time can be different
                _, nc, w, h = shape
                xk = xk + self.interpolate_pos_encoding(xk, w, h, self.pos_embed[j])
            elif self.dims[j] == 1:
                # the size of 1D inputs does not change so there is no need to interpolate
                xk = xk + self.pos_embed[j]
            tokens.append(xk)

        x = torch.cat(tokens, dim=1)

        return self.pos_drop(x)

    def forward(self, x: TensorDict):
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]


def test_mm_transformer():

    # Create a dataset class
    from torch.utils.data import DataLoader, Dataset

    class DummyMultiModalDataset(Dataset):

        def __getitem__(self, index):

            img1 = torch.rand((3, 224, 224))
            img2 = torch.rand((3, 224, 224))
            proprio = torch.rand((1, 10))
            touch = torch.rand((1, 2))

            data = {
                "rgb1": img1,
                "rgb2": img2,
                "sensor1": proprio,
                "sensor2": touch,
            }

            return data

        def __len__(self):
            return 128

    dummy_dataset = DummyMultiModalDataset()
    dataloader = DataLoader(dummy_dataset, batch_size=4, pin_memory=True)
    for batch_ in dataloader:
        batch = TensorDict(batch_, batch_size=4, device="cuda:0")

    mmt = MultimodalTransformer()
    mmt.to(device="cuda:0")

    for batch_ in dataloader:
        batch = TensorDict(batch_, batch_size=4, device="cuda:0")

    out = mmt(batch)

    print(out.shape)


if __name__ == "__main__":

    test_mm_transformer()
