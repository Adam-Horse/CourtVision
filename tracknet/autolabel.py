"""
Teacher -> pseudo-label pipeline for distillation.

Runs a trained TrackNet teacher (the 720p fine-tune by default) over raw match
videos and writes pseudo-labels in the SAME layout as the public dataset
(game*/Clip*/####.jpg + Label.csv), so the existing TrackNetDataset loader reads
them with no changes. A `conf` column (0..1 heatmap peak) is added so training
can filter or weight by teacher confidence.

This is the unlock for training a bigger model (TrackNetX): the teacher labels
unlimited unlabeled broadcast video for free, giving the match/court *diversity*
the ~10-game public set lacks, without human labeling.

To avoid storing near-duplicate adjacent frames, frames are written in short
CLIPS (clip-len consecutive frames, so motion is intact for 3-frame stacking)
sampled every clip-stride frames across the match.

Usage:
    python -m tracknet.autolabel --videos input_videos/ \
        --weights tracknet/weights/tracknet_720.pt \
        --out datasets/distill --game-prefix yt --conf 0.5
"""
import argparse
import csv
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch


def _load_teacher(weights, device, force_cls):
    """Return (forward_fn, in_w, in_h). forward_fn(triplet9) -> 0..1 heatmap np."""
    ckpt = torch.load(weights, map_location=device, weights_only=False)
    reg = (isinstance(ckpt, dict) and ckpt.get("reg_head")) and not force_cls
    if reg:
        from .model import build_finetune
        net = build_finetune(None, device=device)
        net.load_state_dict(ckpt["model_state_dict"])
        net.eval()
        in_w = int(ckpt.get("width", 1280))
        in_h = int(ckpt.get("height", 720))

        def fwd(inp):
            with torch.no_grad():
                hm = torch.sigmoid(net(inp))[0, 0]      # H,W in [0,1]
            return hm.float().cpu().numpy()
        return fwd, in_w, in_h

    # 256-class pretrained head
    from .model import load_pretrained
    from .heatmap import logits_to_heatmap
    net = load_pretrained(weights, device=device)
    in_w, in_h = 640, 360

    def fwd(inp):
        with torch.no_grad():
            logits = net(inp)[0]                        # 256,H,W
        return logits_to_heatmap(logits) / 255.0        # -> 0..1
    return fwd, in_w, in_h


def _load_play_gate(weights, device):
    """Return predict(frame)->P(play) using the trained YOLOv8n-cls model, or None.

    Drops non-play frames (crowd / replay / closeups) before ball auto-labeling
    so the distillation set is court-view play frames only.
    """
    if not weights:
        return None
    from ultralytics import YOLO
    m = YOLO(weights)
    play_idx = next((i for i, n in m.names.items() if n.lower() == "play"), None)
    if play_idx is None:
        print(f"  !! 'play' class not in {m.names}; play-gate disabled")
        return None

    def predict(frame):
        r = m.predict(frame, verbose=False, device=device)[0]
        return float(r.probs.data[play_idx])
    return predict


def _triplet(frames, in_w, in_h, device):
    chans = [cv2.resize(f, (in_w, in_h)).astype(np.float32) / 255.0 for f in frames]
    stacked = np.concatenate(chans, axis=2)             # H,W,9
    t = torch.from_numpy(stacked).permute(2, 0, 1).unsqueeze(0)
    return t.to(device)


def _in_clip(idx, clip_len, clip_stride):
    """True when frame idx falls inside a sampled clip window."""
    return (idx % clip_stride) < clip_len


def process_video(path, out_game_dir, fwd, in_w, in_h, args, device, play_fn=None):
    from .heatmap import decode_heatmap
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"  !! cannot open {path}")
        return 0
    nat_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    nat_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sx, sy = nat_w / in_w, nat_h / in_h

    buf = []                                            # [cur, prev, prev2]
    idx = 0
    clip_id = -1
    clip_dir = None
    rows = []
    play_votes = []                                     # per-frame play flags
    written = dropped = 0

    def flush():
        nonlocal dropped
        if clip_dir is None or not rows:
            return 0
        # Whole-clip play gate: keep only if enough frames are play (preserves
        # frame-to-frame continuity within kept clips).
        if play_fn is not None:
            frac = sum(play_votes) / max(1, len(play_votes))
            if frac < args.play_min:
                shutil.rmtree(clip_dir, ignore_errors=True)
                dropped += 1
                return 0
        with open(clip_dir / "Label.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["file name", "visibility", "x-coordinate",
                        "y-coordinate", "status", "conf"])
            w.writerows(rows)
        return len(rows)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        buf.insert(0, frame)
        if len(buf) > 3:
            buf.pop()

        if len(buf) == 3 and _in_clip(idx, args.clip_len, args.clip_stride):
            cid = idx // args.clip_stride
            if cid != clip_id:                          # new clip -> new folder
                written += flush()
                rows, play_votes = [], []
                clip_id = cid
                clip_dir = out_game_dir / f"Clip{cid:04d}"
                clip_dir.mkdir(parents=True, exist_ok=True)

            if play_fn is not None:
                play_votes.append(1 if play_fn(frame) >= args.play_thresh else 0)

            inp = _triplet(buf, in_w, in_h, device)
            hm = fwd(inp)
            x, y, peak = decode_heatmap(hm, thresh=args.conf)
            vis = 1 if x is not None else 0
            xc = round(x * sx, 1) if x is not None else ""
            yc = round(y * sy, 1) if y is not None else ""

            fname = f"{idx % args.clip_stride:04d}.jpg"
            save = frame
            if args.save_width and args.save_height:
                save = cv2.resize(frame, (args.save_width, args.save_height))
                if vis:
                    xc = round(x * sx * args.save_width / nat_w, 1)
                    yc = round(y * sy * args.save_height / nat_h, 1)
            cv2.imwrite(str(clip_dir / fname), save,
                        [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_q])
            rows.append([fname, vis, xc, yc, 0, round(float(peak), 3)])

        idx += 1
        if idx % 500 == 0:
            print(f"    {path.name}: {idx} frames, {written} kept, "
                  f"{dropped} clips dropped", flush=True)

    written += flush()
    cap.release()
    if play_fn is not None:
        print(f"  {path.name}: {written} play frames kept, {dropped} non-play "
              f"clips dropped")
    return written


def main(args):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    fwd, in_w, in_h = _load_teacher(args.weights, device, args.cls)
    play_fn = _load_play_gate(args.play_weights, device)
    print(f"[autolabel] teacher={args.weights} in={in_w}x{in_h} device={device} "
          f"play_gate={'on' if play_fn else 'off'}")

    vids = sorted([p for p in Path(args.videos).iterdir()
                   if p.suffix.lower() in (".mp4", ".mkv", ".mov", ".avi")]) \
        if Path(args.videos).is_dir() else [Path(args.videos)]
    out_root = Path(args.out)
    total = 0
    for i, v in enumerate(vids):
        game_dir = out_root / f"{args.game_prefix}{i:03d}"
        print(f"[autolabel] {v.name} -> {game_dir.name}")
        total += process_video(v, game_dir, fwd, in_w, in_h, args, device, play_fn)
    print(f"[autolabel] done. {len(vids)} videos, {total} pseudo-labeled frames "
          f"-> {out_root}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True, help="video file or directory")
    ap.add_argument("--weights", default="tracknet/weights/tracknet_720.pt")
    ap.add_argument("--out", default="datasets/distill")
    ap.add_argument("--game-prefix", default="yt")
    ap.add_argument("--conf", type=float, default=0.5,
                    help="min heatmap peak (0..1) to mark the ball visible")
    ap.add_argument("--clip-len", type=int, default=60,
                    help="consecutive frames written per sampled clip")
    ap.add_argument("--clip-stride", type=int, default=300,
                    help="frame gap between sampled clips (diversity vs redundancy)")
    ap.add_argument("--save-width", type=int, default=0)
    ap.add_argument("--save-height", type=int, default=0)
    ap.add_argument("--jpeg-q", type=int, default=90)
    ap.add_argument("--cls", action="store_true", help="teacher uses 256-class head")
    ap.add_argument("--play-weights", default=None,
                    help="YOLOv8n-cls play/court model; drops non-play clips")
    ap.add_argument("--play-thresh", type=float, default=0.5,
                    help="per-frame P(play) cutoff")
    ap.add_argument("--play-min", type=float, default=0.6,
                    help="min fraction of a clip's frames that must be play to keep it")
    ap.add_argument("--cpu", action="store_true")
    main(ap.parse_args())
