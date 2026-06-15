"""
TrackNetX - a larger, accuracy-first redesign of TrackNet for tennis.

Where the 2019 model is a flat VGG encoder/decoder with no skips and only
3-frame early fusion, TrackNetX is built to *predict better*, not cheaper:

  1. Deep RESIDUAL encoder (ResBlocks)      -> more capacity, stable training
  2. True U-NET SKIP CONNECTIONS            -> pixel-precise heatmaps
  3. Longer temporal window (T frames) with
     explicit TEMPORAL CROSS-ATTENTION       -> the network *learns* ball motion
     (aggregates all T frames into the centre frame), instead of leaving
     continuity to a hand-built Viterbi/constant-velocity tracker
  4. SPATIAL SELF-ATTENTION at the bottleneck -> global court/player context
  5. MULTI-TASK heads:
        - heatmap (Gaussian, 1-ch sigmoid)   -> where the ball is
        - sub-pixel OFFSET (2-ch)            -> CenterNet-style refinement
        - VISIBILITY logit                   -> explicit ball-present/absent
  6. Learned bilinear+conv upsampling        -> no nearest-neighbour artifacts

Input : a window of T RGB frames, [B, T, 3, H, W]  (also accepts [B, 3T, H, W]).
        Heads predict for the CENTRE frame, using past+future context.
Output: dict(heatmap=[B,1,H,W] logits, offset=[B,2,H,W], vis=[B,1] logit).

Speed is intentionally not a goal here - this is the high-capacity "teacher".
"""
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Standard pre-activation-free residual block: (Conv-BN-ReLU)x2 + skip."""

    def __init__(self, in_c: int, out_c: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU(inplace=True)
        if stride != 1 or in_c != out_c:
            self.down = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride, bias=False),
                nn.BatchNorm2d(out_c),
            )
        else:
            self.down = nn.Identity()

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + self.down(x))


def _stage(in_c: int, out_c: int, stride: int, n: int = 2) -> nn.Sequential:
    blocks = [ResBlock(in_c, out_c, stride)]
    for _ in range(n - 1):
        blocks.append(ResBlock(out_c, out_c, 1))
    return nn.Sequential(*blocks)


class Encoder(nn.Module):
    """Shared per-frame residual encoder. Returns multi-scale skips + bottleneck.

    Stem keeps full resolution (no aggressive early downsample) because the ball
    is tiny - we cannot afford to throw away spatial detail up front.
    """

    def __init__(self, in_c: int = 3, c: List[int] = (64, 128, 256, 512)):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_c, c[0], 3, 1, 1, bias=False),
            nn.BatchNorm2d(c[0]),
            nn.ReLU(inplace=True),
        )
        self.s1 = _stage(c[0], c[0], stride=1)   # full res
        self.s2 = _stage(c[0], c[1], stride=2)   # /2
        self.s3 = _stage(c[1], c[2], stride=2)   # /4
        self.s4 = _stage(c[2], c[3], stride=2)   # /8 (bottleneck)

    def forward(self, x):
        x = self.stem(x)
        e1 = self.s1(x)
        e2 = self.s2(e1)
        e3 = self.s3(e2)
        e4 = self.s4(e3)
        return e1, e2, e3, e4


class TemporalCrossAttention(nn.Module):
    """Per-pixel attention across the T frames; centre frame attends to all.

    Reshapes bottleneck features [B,T,C,h,w] so every spatial location is a
    length-T sequence, then the centre frame (query) attends over all frames
    (key/value). This is where the network reasons about *where the ball is
    heading* - the learned replacement for a constant-velocity linker.
    """

    def __init__(self, dim: int, heads: int = 8):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, feats):                         # [B, T, C, h, w]
        B, T, C, h, w = feats.shape
        center = T // 2
        seq = feats.permute(0, 3, 4, 1, 2).reshape(B * h * w, T, C)
        q = seq[:, center:center + 1, :]              # [N,1,C]
        out, _ = self.attn(q, seq, seq)               # centre <- all frames
        out = self.norm(out + q).reshape(B, h, w, C).permute(0, 3, 1, 2)
        return out                                    # [B, C, h, w]


class SpatialSelfAttention(nn.Module):
    """Global self-attention over the (small) bottleneck grid for court context."""

    def __init__(self, dim: int, heads: int = 8):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(),
                                nn.Linear(dim * 2, dim))
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):                             # [B, C, h, w]
        B, C, h, w = x.shape
        t = x.flatten(2).transpose(1, 2)              # [B, hw, C]
        a, _ = self.attn(t, t, t)
        t = self.norm(t + a)
        t = self.norm2(t + self.ff(t))
        return t.transpose(1, 2).reshape(B, C, h, w)


class UpBlock(nn.Module):
    """Bilinear upsample -> concat U-Net skip -> two residual blocks."""

    def __init__(self, in_c: int, skip_c: int, out_c: int):
        super().__init__()
        self.fuse = _stage(in_c + skip_c, out_c, stride=1, n=2)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear",
                          align_corners=False)
        return self.fuse(torch.cat([x, skip], dim=1))


class TrackNetX(nn.Module):
    def __init__(self, frames: int = 5, c: List[int] = (64, 128, 256, 512),
                 heads: int = 8):
        super().__init__()
        self.frames = frames
        self.center = frames // 2
        self.encoder = Encoder(3, c)
        self.temporal = TemporalCrossAttention(c[3], heads)
        self.spatial = SpatialSelfAttention(c[3], heads)

        self.u3 = UpBlock(c[3], c[2], c[2])           # /8 -> /4
        self.u2 = UpBlock(c[2], c[1], c[1])           # /4 -> /2
        self.u1 = UpBlock(c[1], c[0], c[0])           # /2 -> full

        self.head_hm = nn.Conv2d(c[0], 1, 1)
        self.head_off = nn.Conv2d(c[0], 2, 1)
        self.head_vis = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(c[3], 128), nn.ReLU(inplace=True), nn.Linear(128, 1))

    def _to_window(self, x):
        if x.dim() == 4:                              # [B, 3T, H, W] -> [B,T,3,H,W]
            B, C, H, W = x.shape
            x = x.view(B, C // 3, 3, H, W)
        return x

    def forward(self, x) -> Dict[str, torch.Tensor]:
        x = self._to_window(x)                        # [B, T, 3, H, W]
        B, T, _, H, W = x.shape
        e1, e2, e3, e4 = self.encoder(x.reshape(B * T, 3, H, W))

        ch = e4.shape[1]
        bott = e4.view(B, T, ch, *e4.shape[-2:])
        fused = self.temporal(bott)                   # [B, C, h, w] centre-fused
        fused = self.spatial(fused)

        # decoder skips come from the centre frame's per-scale features
        def center_of(feat):
            return feat.view(B, T, feat.shape[1], *feat.shape[-2:])[:, self.center]

        d3 = self.u3(fused, center_of(e3))
        d2 = self.u2(d3, center_of(e2))
        d1 = self.u1(d2, center_of(e1))
        return {
            "heatmap": self.head_hm(d1),              # [B, 1, H, W] logits
            "offset": self.head_off(d1),              # [B, 2, H, W]
            "vis": self.head_vis(fused).squeeze(-1),  # [B] logit
        }


if __name__ == "__main__":
    net = TrackNetX(frames=5)
    n = sum(p.numel() for p in net.parameters())
    x = torch.randn(2, 5, 3, 288, 512)
    y = net(x)
    print(f"TrackNetX params: {n/1e6:.2f}M")
    for k, v in y.items():
        print(f"  {k}: {tuple(v.shape)}")
