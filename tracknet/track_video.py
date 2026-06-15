"""
TrackNet + trajectory linker + rally-state machine on a video.

This is the "continuation-aware" runner: top-K heatmap peaks per frame ->
BallTracker (constant-velocity association + state machine) -> clean ball
positions that no longer teleport onto distractors, plus a rally-state badge
that reads IDLE during serve-prep / between points.

Usage:
    python -m tracknet.track_video --video input_videos/videoclip.mp4 \
        --out output_videos/tracknet_tracked.mp4 --log /tmp/tracked.csv
"""
import argparse
import collections

import cv2
import numpy as np
import torch

from .model import load_pretrained
from .heatmap import logits_to_heatmap, topk_peaks
from .track import BallTracker, RallyState
from .infer import preprocess_triplet, IN_W, IN_H

_BADGE = {
    RallyState.IDLE:       ("BETWEEN POINTS", (160, 160, 160)),
    RallyState.CONFIRMING: ("TRACKING...",    (0, 200, 255)),
    RallyState.LIVE:       ("RALLY LIVE",     (0, 220, 0)),
}


def run(args):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    net = load_pretrained(args.weights, device=device)
    print(f"[track] device={device}")

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

    tracker = BallTracker()
    buf = collections.deque(maxlen=3)
    trail = collections.deque(maxlen=args.trail)
    n_live = n_seen = 0
    log_rows = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        buf.appendleft(frame.copy())
        n_seen += 1

        state, ball = RallyState.IDLE, None
        if len(buf) == 3:
            inp = preprocess_triplet(list(buf)).to(device)
            with torch.no_grad():
                hm = logits_to_heatmap(net(inp)[0])
            cands = [(x * sx, y * sy, v)
                     for (x, y, v) in topk_peaks(hm, k=args.topk, min_val=args.min_val)]
            state, ball = tracker.update(n_seen - 1, cands)
            if state == RallyState.LIVE:
                n_live += 1
            log_rows.append((n_seen - 1, state.value,
                             ball[0] if ball else "", ball[1] if ball else ""))

        trail.appendleft(ball if state == RallyState.LIVE else None)

        if writer is not None:
            vis = frame.copy()
            tl = list(trail)
            for i in range(1, len(tl)):
                a, b = tl[i], tl[i - 1]
                if a is None or b is None:
                    continue
                if abs(a[0] - b[0]) + abs(a[1] - b[1]) > args.max_seg:
                    continue
                cv2.line(vis, tuple(map(int, a)), tuple(map(int, b)), (0, 255, 255), 2)
            if ball is not None and state == RallyState.LIVE:
                cv2.circle(vis, tuple(map(int, ball)), 6, (0, 0, 255), -1)
                cv2.circle(vis, tuple(map(int, ball)), 11, (0, 255, 255), 2)
            label, color = _BADGE[state]
            cv2.rectangle(vis, (15, 15), (15 + 12 * len(label) + 20, 55), (0, 0, 0), -1)
            cv2.putText(vis, label, (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            writer.write(vis)

        if n_seen % 100 == 0:
            print(f"  {n_seen}/{n_total}  live={n_live}", flush=True)

    cap.release()
    if writer is not None:
        writer.release()
    if args.log:
        import csv
        with open(args.log, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["frame", "state", "x", "y"])
            w.writerows(log_rows)
        print(f"[track] wrote log -> {args.log}")
    print(f"[track] frames={n_seen}  live_frames={n_live} "
          f"({100*n_live/max(1,n_seen-2):.1f}%)  out={args.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--weights", default="tracknet/weights/tracknet_pretrained.pt")
    ap.add_argument("--out", default="output_videos/tracknet_tracked.mp4")
    ap.add_argument("--log", default=None)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--min-val", type=float, default=64.0)
    ap.add_argument("--trail", type=int, default=12)
    ap.add_argument("--max-seg", type=float, default=250.0)
    ap.add_argument("--cpu", action="store_true")
    run(ap.parse_args())
