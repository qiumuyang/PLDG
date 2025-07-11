from itertools import chain

import torch
import torch.nn as nn


class ConvBlock(nn.Sequential):

    def __init__(self, num_stages, in_channels, out_channels):
        layers = []
        for i in range(num_stages):
            layers.extend([
                nn.Conv3d(in_channels if i == 0 else out_channels,
                          out_channels,
                          kernel_size=3,
                          padding=1),
                nn.BatchNorm3d(out_channels),
                nn.ReLU()
            ])
        super().__init__(*layers)


class DownBlock(nn.Sequential):

    def __init__(self, in_channels, out_channels, use_conv=True):
        if use_conv:
            down = nn.Conv3d(in_channels,
                             out_channels,
                             kernel_size=2,
                             padding=0,
                             stride=2)
        else:
            down = nn.Sequential(
                nn.MaxPool3d(2),
                nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            )
        super().__init__(
            down,
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
        )


class UpBlock(nn.Sequential):

    def __init__(self, in_channels, out_channels, use_conv=True):
        if use_conv:
            up = nn.ConvTranspose3d(in_channels,
                                    out_channels,
                                    kernel_size=2,
                                    padding=0,
                                    stride=2)
        else:
            up = nn.Sequential(
                nn.Upsample(scale_factor=2,
                            mode="trilinear",
                            align_corners=False),
                nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            )
        super().__init__(
            up,
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
        )


class Encoder(nn.Module):

    def __init__(
        self,
        in_channels: int,
        num_filters: int,
        stages: list[int] = [1, 2, 3, 3, 3],
    ):
        super().__init__()
        self.num_filters = num_filters
        self.init_conv = ConvBlock(stages[0], in_channels, num_filters)
        self.down = nn.ModuleList([
            nn.Sequential(
                DownBlock(in_channels=num_filters * 2**(i - 1),
                          out_channels=num_filters * 2**i),
                ConvBlock(s,
                          in_channels=num_filters * 2**i,
                          out_channels=num_filters * 2**i),
            ) for i, s in enumerate(stages[1:], 1)
        ])

    def forward(self, x):
        x = self.init_conv(x)
        feat = [x]
        for down in self.down:
            x = down(x)
            feat.append(x)
        return feat

    @property
    def out_channels(self) -> list[int]:
        return [self.num_filters * 2**i for i in range(len(self.down) + 1)]


class Decoder(nn.Module):

    def __init__(
        self,
        out_channels: int,
        num_filters: int,
        stages: list[int] = [1, 2, 3, 3, 3],
        multi_binary: bool = False,
        lightweight: bool = True,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.num_filters = num_filters
        self.stages = stages
        self.multi_binary = multi_binary
        self.lightweight = lightweight
        self.up = nn.ModuleList([
            nn.Sequential(
                UpBlock(in_channels=num_filters * 2**i,
                        out_channels=num_filters * 2**(i - 1)),
                ConvBlock(s,
                          in_channels=num_filters * 2**(i - 1),
                          out_channels=num_filters * 2**(i - 1)),
            ) for i, s in reversed(list(enumerate(stages[:-1], 1)))
        ])
        self.out_conv = nn.Conv3d(num_filters, out_channels, kernel_size=1)
        if multi_binary:
            if not lightweight:
                self.mlp = nn.Sequential(
                    nn.Conv3d(num_filters, num_filters, kernel_size=1),
                    nn.BatchNorm3d(num_filters),
                    nn.ReLU(),
                )
            else:
                self.mlp = nn.Identity()
            self.out_binary = nn.ModuleList([
                nn.Conv3d(num_filters, 1, kernel_size=1)
                for _ in range(out_channels - 1)
            ])

    def forward(self, feat):
        x = feat.pop()
        feat_dec = []
        for up, conv in self.up:  # type: ignore
            x = up(x)
            x = x + feat.pop()
            x = conv(x)
            feat_dec.append(x)
        if self.multi_binary:
            x = self.mlp(x)
            bins = [out(x) for out in self.out_binary]
            bins = torch.cat(bins, dim=1)
            return self.out_conv(x), bins, feat_dec
        return self.out_conv(x), feat_dec

    @property
    def feat_channels(self) -> list[int]:
        return [self.num_filters * 2**i for i in reversed(range(len(self.up)))]


class VNet(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_filters: int,
                 stages: list[int] = [1, 2, 3, 3, 3],
                 multi_binary: bool = False,
                 lightweight: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_filters = num_filters
        self.stages = stages
        self.multi_binary = multi_binary
        self.encoder = Encoder(in_channels, num_filters, stages)
        self.decoder = Decoder(out_channels,
                               num_filters,
                               stages,
                               multi_binary=multi_binary,
                               lightweight=lightweight)

    def forward(self, x, return_feats=False, return_binary=False):
        feat = self.encoder(x)
        if self.multi_binary:
            out, out_bin, feat_dec = self.decoder(feat.copy())
        else:
            out, feat_dec = self.decoder(feat.copy())
            out_bin = None
        if return_binary and return_feats:
            return out, feat + feat_dec, out_bin
        if return_feats:
            return out, feat + feat_dec
        if return_binary:
            return out, out_bin
        return out

    @property
    def feat_channels(self) -> list[int]:
        return self.encoder.out_channels + self.decoder.feat_channels
