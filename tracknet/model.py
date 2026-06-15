"""
TrackNet - heatmap-based tennis-ball tracker (VGG16-style encoder/decoder).

This matches the reference implementation (yastrebksv/TrackNet, after the
original Huang et al. 2019 "TrackNet") *exactly* at the module level so the
public pretrained weights load with no key remapping:

    18 conv blocks, each = [Conv2d(3x3, pad 1) -> ReLU -> BatchNorm2d],
    named conv1..conv18, with 3 MaxPool (encoder) / 3 Upsample (decoder) stages.

Input  : 3 stacked RGB frames -> 9 channels, [B, 9, H, W]  (H=360, W=640).
Output : [B, 256, H, W]. The 256 channels are a per-pixel softmax over ball
         intensity levels 0..255 (the Gaussian heatmap is quantised to 8-bit).
         At inference: argmax over the channel dim -> a 0..255 grayscale
         heatmap; threshold + connected-component centroid -> ball (x, y).
         If no pixel clears the threshold the frame has NO ball - the native
         no-ball signal we want.

Why 3 frames in: a single frame can't separate a static round logo from a
ball. Stacking consecutive frames lets the conv stack key on *motion*, which
is what makes a flying ball distinct.
"""
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv(3x3) -> ReLU -> BatchNorm. The repeating TrackNet unit.

    Submodule name `block` and the [conv, relu, bn] ordering are chosen so the
    state_dict keys (conv{i}.block.0 / .block.2) match the pretrained weights.
    """

    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_c),
        )

    def forward(self, x):
        return self.block(x)


class TrackNet(nn.Module):
    def __init__(self, in_channels: int = 9, out_channels: int = 256,
                 reg_head: bool = False):
        """reg_head=False -> faithful 256-class softmax head (pretrained).
        reg_head=True  -> plain 1-channel Conv2d head emitting a heatmap logit,
        for lightweight higher-resolution fine-tuning (BCE on a Gaussian target).
        """
        super().__init__()
        self.out_channels = 1 if reg_head else out_channels
        self.reg_head = reg_head

        # ---- Encoder ----
        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 64)
        self.conv3 = ConvBlock(64, 128)
        self.conv4 = ConvBlock(128, 128)
        self.conv5 = ConvBlock(128, 256)
        self.conv6 = ConvBlock(256, 256)
        self.conv7 = ConvBlock(256, 256)
        self.conv8 = ConvBlock(256, 512)
        self.conv9 = ConvBlock(512, 512)
        self.conv10 = ConvBlock(512, 512)
        # ---- Decoder ----
        self.conv11 = ConvBlock(512, 256)
        self.conv12 = ConvBlock(256, 256)
        self.conv13 = ConvBlock(256, 256)
        self.conv14 = ConvBlock(256, 128)
        self.conv15 = ConvBlock(128, 128)
        self.conv16 = ConvBlock(128, 64)
        self.conv17 = ConvBlock(64, 64)
        self.conv18 = (nn.Conv2d(64, 1, kernel_size=1) if reg_head
                       else ConvBlock(64, out_channels))

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool(x)            # H/2

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool(x)            # H/4

        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.pool(x)            # H/8

        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)

        x = self.up(x)              # H/4
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)

        x = self.up(x)              # H/2
        x = self.conv14(x)
        x = self.conv15(x)

        x = self.up(x)              # H
        x = self.conv16(x)
        x = self.conv17(x)
        x = self.conv18(x)          # [B, 256, H, W]
        return x


def load_pretrained(weights_path: str, device: str = "cpu") -> "TrackNet":
    """Build TrackNet and load reference weights (raw state_dict checkpoint)."""
    net = TrackNet()
    sd = torch.load(weights_path, map_location=device, weights_only=False)
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    net.load_state_dict(sd)
    return net.to(device).eval()


def build_finetune(pretrained_path: str = None, device: str = "cpu") -> "TrackNet":
    """1-channel-head TrackNet with the pretrained backbone (conv1..conv17) loaded.

    Only the final layer (conv18) is reinitialised - everything the encoder and
    decoder learned about ball-shaped motion transfers, which is exactly what
    makes fine-tuning at a new resolution converge fast.
    """
    net = TrackNet(reg_head=True)
    if pretrained_path:
        sd = torch.load(pretrained_path, map_location=device, weights_only=False)
        if isinstance(sd, dict) and "model_state_dict" in sd:
            sd = sd["model_state_dict"]
        backbone = {k: v for k, v in sd.items() if not k.startswith("conv18.")}
        missing, unexpected = net.load_state_dict(backbone, strict=False)
        kept = len(sd) - len(unexpected)
        print(f"[finetune] loaded {kept} backbone tensors; "
              f"reinit head (missing={[m for m in missing if m.startswith('conv18')]})")
    return net.to(device)


if __name__ == "__main__":
    net = TrackNet()
    n_params = sum(p.numel() for p in net.parameters())
    dummy = torch.randn(1, 9, 360, 640)
    out = net(dummy)
    print(f"TrackNet params: {n_params/1e6:.2f}M")
    print(f"in {tuple(dummy.shape)} -> out {tuple(out.shape)}")
