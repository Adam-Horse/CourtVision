"""
Dataset for the public TrackNet tennis set (game*/Clip*/####.jpg + Label.csv).

Each sample is a stack of 3 consecutive frames (9 channels, most-recent-first)
and a single Gaussian heatmap target centred on the ball in the *current*
frame. Frames are resized (stretched) to a configurable (W, H) so we can train
at higher resolution than the 360x640 the reference used. Visibility 0 (ball
absent/occluded) yields an all-zero target - the explicit no-ball signal.

Splitting is grouped by *game* (leave-games-out) so no clip leaks between
train and val - the same evaluation rigor used elsewhere in CourtVision.
"""
import csv
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .heatmap import make_gaussian_heatmap


def _read_clip(clip_dir: Path):
    """Return list of (frame_path, visibility, x, y) sorted by frame name."""
    label = clip_dir / "Label.csv"
    if not label.exists():
        return []
    rows = []
    with open(label, newline="") as fh:
        for r in csv.DictReader(fh):
            name = r["file name"]
            p = clip_dir / name
            if not p.exists():
                continue
            try:
                vis = int(r["visibility"])
                x = float(r["x-coordinate"]) if r["x-coordinate"] else -1.0
                y = float(r["y-coordinate"]) if r["y-coordinate"] else -1.0
            except (ValueError, KeyError):
                continue
            rows.append((p, vis, x, y))
    rows.sort(key=lambda t: t[0].name)
    return rows


def build_samples(root, games):
    """Build triplet samples for the given games.

    Returns list of dicts: {paths:[cur,prev,prev2], vis, x, y, native_wh}.
    """
    root = Path(root)
    samples = []
    for g in games:
        gdir = root / g
        if not gdir.is_dir():
            continue
        for clip in sorted(gdir.iterdir()):
            if not clip.is_dir():
                continue
            rows = _read_clip(clip)
            for i in range(2, len(rows)):
                cur, prev, prev2 = rows[i], rows[i - 1], rows[i - 2]
                samples.append({
                    "paths": [cur[0], prev[0], prev2[0]],
                    "vis": cur[1], "x": cur[2], "y": cur[3],
                })
    return samples


def list_games(root):
    root = Path(root)
    return sorted([p.name for p in root.iterdir()
                   if p.is_dir() and p.name.startswith("game")])


class TrackNetDataset(Dataset):
    def __init__(self, samples, width=1280, height=720, sigma=5.0):
        self.samples = samples
        self.W, self.H = width, height
        self.sigma = sigma

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        chans = []
        nat_w = nat_h = None
        for p in s["paths"]:
            img = cv2.imread(str(p))
            if img is None:
                img = np.zeros((self.H, self.W, 3), np.uint8)
            if nat_w is None:
                nat_h, nat_w = img.shape[:2]
            r = cv2.resize(img, (self.W, self.H))
            chans.append(r.astype(np.float32) / 255.0)
        inp = np.concatenate(chans, axis=2)              # H,W,9
        inp = torch.from_numpy(inp).permute(2, 0, 1)     # 9,H,W

        if s["vis"] > 0 and s["x"] >= 0 and nat_w:
            cx = s["x"] * self.W / nat_w
            cy = s["y"] * self.H / nat_h
            hm = make_gaussian_heatmap(self.H, self.W, cx, cy, self.sigma)
        else:
            hm = np.zeros((self.H, self.W), np.float32)
        target = torch.from_numpy(hm).unsqueeze(0)       # 1,H,W
        return inp, target
