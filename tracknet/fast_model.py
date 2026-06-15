"""
FastTrackNet - a leaner, sharper redesign of the 2019 TrackNet.

The original (model.py) is a VGG16-style encoder/decoder: 18 plain 3x3 conv
blocks, no skip connections, nearest-neighbour upsampling, and a 256-way
softmax head. It localizes well but is FLOP-heavy and rebuilds the heatmap from
a coarse H/8 bottleneck, so it needs very wide channels to stay accurate.

This redesign keeps the heatmap formulation (so the training/decoding code is
unchanged) but fixes the structural inefficiencies:

  1. U-Net skip connections        -> sharp localization without wide channels
  2. Depthwise-separable convs      -> ~8-9x fewer FLOPs per 3x3 op
  3. Conv -> BN -> ReLU (standard)  -> cleaner optimization than TrackNet's order
  4. Learned bilinear+conv upsample -> no blocky nearest-neighbour artifacts
  5. 1-channel sigmoid heatmap head -> drops the wasteful 256-way softmax
  6. Motion-attention gate          -> a cheap temporal cue (TrackNetV4 idea):
     gate spatial features by inter-frame difference so the net keys on motion

Input  : 3 stacked RGB frames -> 9 channels, [B, 9, H, W].
Output : [B, 1, H, W] raw logits. sigmoid -> heatmap; reuse decode_heatmap().

`width` scales every channel count, so the same design spans an embedded-tiny
student and a server-side teacher from one definition.
"""
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class DSConv(nn.Module):
    """Depthwise-separable conv: 3x3 depthwise + 1x1 pointwise, each BN+ReLU.

    Replaces a dense Conv3x3 at a fraction of the cost: a dense 3x3 with C_in,
    C_out costs 9*C_in*C_out per pixel; this costs 9*C_in + C_in*C_out, which is
    ~8-9x cheaper at the channel widths used here.
    """

    def __init__(self, in_c: int, out_c: int, stride: int = 1):
        super().__init__()
        self.dw = nn.Conv2d(in_c, in_c, 3, stride=stride, padding=1,
                            groups=in_c, bias=False)
        self.dw_bn = nn.BatchNorm2d(in_c)
        self.pw = nn.Conv2d(in_c, out_c, 1, bias=False)
        self.pw_bn = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.act(self.dw_bn(self.dw(x)))
        x = self.act(self.pw_bn(self.pw(x)))
        return x


class EncBlock(nn.Module):
    """Two DSConvs; the first optionally downsamples by 2."""

    def __init__(self, in_c: int, out_c: int, down: bool = True):
        super().__init__()
        self.c1 = DSConv(in_c, out_c, stride=2 if down else 1)
        self.c2 = DSConv(out_c, out_c)

    def forward(self, x):
        return self.c2(self.c1(x))


class UpBlock(nn.Module):
    """Bilinear upsample -> concat skip -> two DSConvs (U-Net decoder stage)."""

    def __init__(self, in_c: int, skip_c: int, out_c: int):
        super().__init__()
        self.reduce = nn.Conv2d(in_c, out_c, 1, bias=False)
        self.reduce_bn = nn.BatchNorm2d(out_c)
        self.c1 = DSConv(out_c + skip_c, out_c)
        self.c2 = DSConv(out_c, out_c)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear",
                          align_corners=False)
        x = F.relu(self.reduce_bn(self.reduce(x)), inplace=True)
        x = torch.cat([x, skip], dim=1)
        return self.c2(self.c1(x))


class MotionAttention(nn.Module):
    """Cheap temporal gate from inter-frame differences.

    Splits the 9-channel input into 3 RGB frames, builds |f2-f1| and |f1-f0|
    motion maps, and learns a 1-channel spatial gate in [0,1]. Multiplying the
    stem features by this gate pushes the network to attend to moving regions
    (the ball) and suppress static distractors (logos, lines) - the same effect
    TrackNetV4 gets from frame-difference feature modulation, at ~zero cost.
    """

    def __init__(self, feat_c: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(6, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
        )

    def forward(self, feat, frames9):
        f0, f1, f2 = frames9[:, 0:3], frames9[:, 3:6], frames9[:, 6:9]
        motion = torch.cat([(f1 - f0).abs(), (f2 - f1).abs()], dim=1)
        motion = F.interpolate(motion, size=feat.shape[-2:], mode="bilinear",
                               align_corners=False)
        gate = torch.sigmoid(self.net(motion))
        return feat * (1.0 + gate)          # residual gate: never zeros features


class FastTrackNet(nn.Module):
    def __init__(self, in_channels: int = 9, width: float = 1.0,
                 motion_attention: bool = True):
        super().__init__()
        c: List[int] = [int(w * width) for w in (32, 64, 96, 128)]

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(c[0]),
            nn.ReLU(inplace=True),
            DSConv(c[0], c[0]),
        )
        self.motion = MotionAttention(c[0]) if motion_attention else None

        # Encoder (e1 stays full-res; e2..e4 halve each time -> /2, /4, /8)
        self.e1 = EncBlock(c[0], c[0], down=False)
        self.e2 = EncBlock(c[0], c[1], down=True)
        self.e3 = EncBlock(c[1], c[2], down=True)
        self.e4 = EncBlock(c[2], c[3], down=True)

        # Decoder with skips back to full resolution
        self.u3 = UpBlock(c[3], skip_c=c[2], out_c=c[2])
        self.u2 = UpBlock(c[2], skip_c=c[1], out_c=c[1])
        self.u1 = UpBlock(c[1], skip_c=c[0], out_c=c[0])
        self.head = nn.Conv2d(c[0], 1, 1)

    def forward(self, x):
        s = self.stem(x)
        if self.motion is not None:
            s = self.motion(s, x)
        e1 = self.e1(s)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        d3 = self.u3(e4, e3)
        d2 = self.u2(d3, e2)
        d1 = self.u1(d2, e1)
        return self.head(d1)                # [B, 1, H, W] logits


if __name__ == "__main__":
    for w in (0.5, 1.0):
        net = FastTrackNet(width=w)
        n = sum(p.numel() for p in net.parameters())
        x = torch.randn(1, 9, 360, 640)
        y = net(x)
        print(f"width={w}: params={n/1e6:.2f}M  in{tuple(x.shape)} -> out{tuple(y.shape)}")
