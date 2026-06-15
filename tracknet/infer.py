"""
Run TrackNet on a video: stack 3 consecutive frames -> heatmap -> ball (x, y).

This is the temporal ball detector. Unlike the per-frame YOLO detector it keys
on *motion*, so a ball sitting in a player's hand or a static logo produces no
heatmap peak -> native "no ball this frame" signal (the corner-lock fix).

Usage:
    python -m tracknet.infer --video input_videos/videoclip.mp4 \
        --weights tracknet/weights/tracknet_pretrained.pt \
        --out output_videos/tracknet_demo.mp4
"""
import argparse
import collections

import cv2
import numpy as np
import torch

from .model import load_pretrained
from .heatmap import decode_logits

IN_W, IN_H = 640, 360  # TrackNet input resolution (W, H)


def preprocess_triplet(frames):
    """frames = [f_t, f_{t-1}, f_{t-2}] (BGR, original size). -> [1,9,H,W] tensor.

    Most-recent-first stacking matches the reference; the network predicts the
    ball for the current (first) frame. Channels: BGR per frame, normalised /255.
    """
    chans = []
    for f in frames:
        r = cv2.resize(f, (IN_W, IN_H))
        chans.append(r.astype(np.float32) / 255.0)
    stacked = np.concatenate(chans, axis=2)          # H,W,9
    t = torch.from_numpy(stacked).permute(2, 0, 1)   # 9,H,W
    return t.unsqueeze(0)


def run(args):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    net = load_pretrained(args.weights, device=device)
    print(f"[tracknet] device={device}  weights={args.weights}")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sx, sy = W / IN_W, H / IN_H

    writer = None
    if args.out:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.out, fourcc, fps, (W, H))

    buf = collections.deque(maxlen=3)          # last 3 original frames
    trail = collections.deque(maxlen=args.trail)
    n_ball = n_seen = 0
    log_rows = []                              # (frame_idx, x, y, peak)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        buf.appendleft(frame.copy())            # buf[0]=current
        n_seen += 1

        ball = None
        if len(buf) == 3:
            inp = preprocess_triplet(list(buf)).to(device)
            with torch.no_grad():
                logits = net(inp)[0]            # [256,H,W]
            x, y, peak = decode_logits(logits, thresh=args.thresh)
            if x is not None:
                ball = (x * sx, y * sy)
                n_ball += 1
            log_rows.append((n_seen - 1,
                             ball[0] if ball else "",
                             ball[1] if ball else "",
                             round(peak, 1)))
        trail.appendleft(ball)

        if writer is not None:
            vis = frame.copy()
            # Draw a segment only between two consecutive detected frames whose
            # jump is plausible - breaks the trail at no-ball frames and scene
            # cuts instead of drawing a line straight across the gap.
            tl = list(trail)
            for i in range(1, len(tl)):
                a, b = tl[i], tl[i - 1]
                if a is None or b is None:
                    continue
                if abs(a[0] - b[0]) + abs(a[1] - b[1]) > args.max_seg:
                    continue
                cv2.line(vis, tuple(map(int, a)), tuple(map(int, b)),
                         (0, 255, 255), 2)
            if ball is not None:
                cv2.circle(vis, tuple(map(int, ball)), 6, (0, 0, 255), -1)
                cv2.circle(vis, tuple(map(int, ball)), 10, (0, 255, 255), 2)
                cv2.putText(vis, "BALL", (int(ball[0]) + 12, int(ball[1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            else:
                cv2.putText(vis, "no ball", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60, 60, 220), 2)
            writer.write(vis)

        if n_seen % 100 == 0:
            print(f"  {n_seen}/{n_total}  ball-rate={n_ball/max(1,n_seen-2):.2f}", flush=True)

    cap.release()
    if writer is not None:
        writer.release()
    if args.log:
        import csv
        with open(args.log, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["frame", "x", "y", "peak"])
            w.writerows(log_rows)
        print(f"[tracknet] wrote per-frame log -> {args.log}")
    print(f"[tracknet] frames={n_seen}  ball_detected={n_ball} "
          f"({100*n_ball/max(1,n_seen-2):.1f}%)  out={args.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--weights", default="tracknet/weights/tracknet_pretrained.pt")
    ap.add_argument("--out", default="output_videos/tracknet_demo.mp4")
    ap.add_argument("--thresh", type=float, default=128.0,
                    help="ball heatmap threshold 0..255 (lower = more detections)")
    ap.add_argument("--trail", type=int, default=12, help="trajectory trail length")
    ap.add_argument("--max-seg", type=float, default=150.0,
                    help="max px jump to connect two trail points (breaks cuts)")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--log", default=None, help="optional CSV of per-frame (x,y,peak)")
    run(ap.parse_args())
