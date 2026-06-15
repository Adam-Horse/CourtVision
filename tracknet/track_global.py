"""
Offline TrackNet ball tracking: static-peak suppression + global trajectory.

Pass 1 runs TrackNet once over the whole video and collects the top-K heatmap
candidates per frame. We then index recurring (static) peaks and blacklist
them, solve a global Viterbi trajectory through what remains, interpolate short
gaps, and drop spurious short runs. Pass 2 re-reads the video and renders.

Usage:
    python -m tracknet.track_global --video input_videos/videoclip.mp4 \
        --out output_videos/tracknet_global.mp4 --log /tmp/global.csv
"""
import argparse
import collections
import pickle
import os

import cv2
import numpy as np
import torch

from .model import load_pretrained
from .heatmap import logits_to_heatmap, topk_peaks
from .infer import preprocess_triplet, IN_W, IN_H
from . import trajectory as tj


def collect_candidates(args, device):
    """Pass 1: TrackNet -> top-K candidates per frame (original coords)."""
    cache = args.cache
    if cache and os.path.exists(cache):
        with open(cache, "rb") as fh:
            blob = pickle.load(fh)
        print(f"[global] loaded {len(blob['cands'])} frames of candidates from cache")
        return blob["cands"], blob["W"], blob["H"], blob["fps"]

    net = load_pretrained(args.weights, device=device)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    sx, sy = W / IN_W, H / IN_H

    buf = collections.deque(maxlen=3)
    cands_per_frame = []
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        buf.appendleft(frame)
        n += 1
        if len(buf) < 3:
            cands_per_frame.append([])
            continue
        inp = preprocess_triplet(list(buf)).to(device)
        with torch.no_grad():
            hm = logits_to_heatmap(net(inp)[0])
        peaks = topk_peaks(hm, k=args.topk, min_val=args.min_val)
        cands_per_frame.append([(x * sx, y * sy, v) for (x, y, v) in peaks])
        if n % 100 == 0:
            print(f"  pass1 {n} frames", flush=True)
    cap.release()
    if cache:
        with open(cache, "wb") as fh:
            pickle.dump({"cands": cands_per_frame, "W": W, "H": H, "fps": fps}, fh)
    return cands_per_frame, W, H, fps


def render(args, path, W, H, fps):
    """Pass 2: re-read video, draw trajectory + rally state."""
    cap = cv2.VideoCapture(args.video)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, fps, (W, H))
    trail = collections.deque(maxlen=args.trail)
    t = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        ball = path[t] if t < len(path) else None
        trail.appendleft(ball)
        vis = frame
        tl = list(trail)
        for i in range(1, len(tl)):
            a, b = tl[i], tl[i - 1]
            if a is None or b is None:
                continue
            if abs(a[0] - b[0]) + abs(a[1] - b[1]) > args.max_seg:
                continue
            cv2.line(vis, tuple(map(int, a)), tuple(map(int, b)), (0, 255, 255), 2)
        if ball is not None:
            cv2.circle(vis, tuple(map(int, ball)), 6, (0, 0, 255), -1)
            cv2.circle(vis, tuple(map(int, ball)), 11, (0, 255, 255), 2)
            label, color = "RALLY LIVE", (0, 220, 0)
        else:
            label, color = "BETWEEN POINTS", (160, 160, 160)
        cv2.rectangle(vis, (15, 15), (15 + 13 * len(label) + 20, 55), (0, 0, 0), -1)
        cv2.putText(vis, label, (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        writer.write(vis)
        t += 1
    cap.release()
    writer.release()


def run(args):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"[global] device={device}")
    cands, W, H, fps = collect_candidates(args, device)

    static_mask, static_list = tj.index_static_peaks(
        cands, still_radius=args.still_radius, still_frames=args.still_frames,
        max_gap=args.still_gap)
    n_dropped = sum(sum(m) for m in static_mask)
    print(f"\n[global] static-peak index: {len(static_list)} distractor spots, "
          f"{n_dropped} candidate instances dropped")
    for x, y, k in static_list[:14]:
        print(f"    ({x:6.0f},{y:6.0f})  still in {k} frames")

    clean = tj.suppress_static(cands, static_mask)
    path = tj.viterbi_trajectory(clean, max_jump=args.max_jump,
                                 w_acc=args.w_acc, miss_cost=args.miss_cost)
    path = tj.drop_short_runs(path, min_len=args.min_run)
    path = tj.interpolate_gaps(path, max_gap=args.max_gap)

    n_live = sum(1 for p in path if p is not None)
    print(f"[global] trajectory: {n_live}/{len(path)} frames have a ball "
          f"({100*n_live/max(1,len(path)):.1f}%)")

    if args.log:
        import csv
        with open(args.log, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["frame", "x", "y"])
            for i, p in enumerate(path):
                w.writerow([i, p[0] if p else "", p[1] if p else ""])
        print(f"[global] wrote log -> {args.log}")

    if args.out:
        render(args, path, W, H, fps)
        print(f"[global] wrote video -> {args.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--weights", default="tracknet/weights/tracknet_pretrained.pt")
    ap.add_argument("--out", default="output_videos/tracknet_global.mp4")
    ap.add_argument("--log", default=None)
    ap.add_argument("--cache", default=None, help="pickle cache of pass-1 candidates")
    ap.add_argument("--topk", type=int, default=6)
    ap.add_argument("--min-val", type=float, default=40.0)
    # static suppression
    ap.add_argument("--still-radius", type=float, default=14.0)
    ap.add_argument("--still-frames", type=int, default=8)
    ap.add_argument("--still-gap", type=int, default=5)
    # viterbi
    ap.add_argument("--max-jump", type=float, default=260.0)
    ap.add_argument("--w-acc", type=float, default=0.6)
    ap.add_argument("--miss-cost", type=float, default=140.0)
    # post
    ap.add_argument("--min-run", type=int, default=4)
    ap.add_argument("--max-gap", type=int, default=7)
    # render
    ap.add_argument("--trail", type=int, default=12)
    ap.add_argument("--max-seg", type=float, default=250.0)
    ap.add_argument("--cpu", action="store_true")
    run(ap.parse_args())
