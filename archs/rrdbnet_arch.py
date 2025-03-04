import torch
from torch import nn as nn
from torch.nn import functional as F

from basicsr.utils.registry import ARCH_REGISTRY
from .arch_util import default_init_weights, make_layer, pixel_unshuffle
from basicsr.archs.swinir_arch import PatchEmbed, PatchUnEmbed, RSTB, BasicLayer, Upsample, UpsampleOneStep
from basicsr.archs.arch_util import to_2tuple, trunc_normal_


class ResidualDenseBlock(nn.Module):
    """Residual Dense Block.

    Used in RRDB block in ESRGAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat=64, num_grow_ch=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        default_init_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        # Empirically, we use 0.2 to scale the residual for better performance
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block.

    Used in RRDB-Net in ESRGAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat, num_grow_ch=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        # Empirically, we use 0.2 to scale the residual for better performance
        return out * 0.2 + x


@ARCH_REGISTRY.register()
class RRDBNet(nn.Module):
    def __init__(self, num_in_ch, num_out_ch, scale=4, num_feat=64, num_block=23, num_grow_ch=32):
        super(RRDBNet, self).__init__()
        self.scale = scale
        if scale == 2:
            num_in_ch = num_in_ch * 4
        elif scale == 1:
            num_in_ch = num_in_ch * 16

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(RRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        # upsample layers
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)  # 추가된 업샘플링 레이어

        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        if self.scale == 2:
            feat = pixel_unshuffle(x, scale=2)
        elif self.scale == 1:
            feat = pixel_unshuffle(x, scale=4)
        else:
            feat = x

        feat = self.conv_first(feat)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat

        # upsampling
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up3(F.interpolate(feat, scale_factor=1.25, mode='bilinear', align_corners=False)))

        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out

class UpsampleNearest5x(nn.Module):
    def __init__(self, num_feat, num_out_ch):
        super().__init__()
        # 필요한 Conv 레이어들을 정의합니다.
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr  = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        # 먼저 5배로 nearest interpolation을 수행합니다.
        x = F.interpolate(x, scale_factor=5, mode='nearest')
        # 이후 여러 Conv 레이어와 활성화 함수를 통해 세부 조정을 합니다.
        x = self.lrelu(self.conv_up1(x))
        x = self.lrelu(self.conv_up2(x))
        x = self.lrelu(self.conv_hr(x))
        x = self.conv_last(x)
        return x

@ARCH_REGISTRY.register()
class HybridGenerator(nn.Module):
    def __init__(self, num_in_ch, num_out_ch, scale=5,
                 embed_dim=96, depths=(6,6,6,6), num_heads=(6,6,6,6),
                 window_size=5, mlp_ratio=4., img_size=20, **kwargs):
        super(HybridGenerator, self).__init__()
        self.scale = scale
        # Shallow feature extraction (RRDBNet 스타일)
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        # Deep feature extraction: CNN -> patch embedding -> Transformer 블록 -> patch unembedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=1,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=None)

        self.transformer_layers = nn.ModuleList([
            RSTB(dim=embed_dim,
                 input_resolution=(img_size, img_size),
                 depth=depths[i],
                 num_heads=num_heads[i],
                 window_size=window_size,
                 mlp_ratio=mlp_ratio,
                 qkv_bias=True,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.1,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 img_size=img_size,
                 patch_size=1,
                 resi_connection='1conv')
            for i in range(len(depths))
        ])

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=1,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=None)

        self.conv_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

        # Reconstruction & Upsampling: scale==5이면 UpsampleNearest5x 사용
        if scale == 5:
            self.upsample = UpsampleNearest5x(embed_dim, num_out_ch)
        else:
            # 기존의 업샘플링 모듈을 사용할 수 있습니다.
            self.upsample = Upsample(scale, num_feat=embed_dim)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        # Shallow feature extraction
        feat = self.conv_first(x)

        # Deep feature extraction: patch embedding → Transformer 블록 → patch unembedding
        B, C, H, W = feat.shape
        x_patch = self.patch_embed(feat)  # (B, H*W, C)
        for layer in self.transformer_layers:
            x_patch = layer(x_patch, (H, W))
        feat_trans = self.patch_unembed(x_patch, (H, W))

        # Skip connection + 정리
        feat = feat + self.conv_body(feat_trans)

        # Upsampling & Reconstruction
        out = self.upsample(self.lrelu(feat))
        return out
