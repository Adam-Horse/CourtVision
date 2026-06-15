"""
Throughput benchmark for the ball-tracking models.

Measures real ms/iteration (forward, and forward+backward) at a given
resolution / batch / temporal window, then projects epoch time and $ cost for a
target dataset size. Use it to replace back-of-envelope cost guesses with
measured numbers (e.g. for TrackNetX vs the TrackNet baseline).

Examples:
    python -m tracknet.bench --arch tracknet  --height 720 --width 1280 --batch 20
    python -m tracknet.bench --arch tracknetx --height 288 --width 512 --batch 8 --frames 5
"""
import argparse
import time

import torch


def build(arch: str, frames: int, reg: bool):
    if arch == "tracknet":
        from .model import TrackNet
        net = TrackNet(reg_head=reg)
        make_in = lambda b, H, W: torch.randn(b, 9, H, W)
    elif arch == "fast":
        from .fast_model import FastTrackNet
        net = FastTrackNet()
        make_in = lambda b, H, W: torch.randn(b, 9, H, W)
    elif arch == "tracknetx":
        from .tracknetx import TrackNetX
        net = TrackNetX(frames=frames)
        make_in = lambda b, H, W: torch.randn(b, frames, 3, H, W)
    else:
        raise SystemExit(f"unknown arch {arch}")
    return net, make_in


def loss_of(out):
    """Scalar loss for a backward pass; handles dict (multi-head) or tensor."""
    if isinstance(out, dict):
        return sum(v.float().mean() for v in out.values())
    return out.float().mean()


def main(args):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    net, make_in = build(args.arch, args.frames, not args.cls)
    net = net.to(device)
    n_params = sum(p.numel() for p in net.parameters())
    x = make_in(args.batch, args.height, args.width).to(device)

    if args.backward:
        net.train()
        opt = torch.optim.SGD(net.parameters(), lr=0.0)
    else:
        net.eval()

    def step():
        if args.backward:
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=args.amp):
                loss = loss_of(net(x))
            loss.backward()
            opt.step()
        else:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16,
                                                 enabled=args.amp):
                net(x)

    for _ in range(args.warmup):                       # warmup / cudnn autotune
        step()
    if device == "cuda":
        torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(args.iters):
        step()
    if device == "cuda":
        torch.cuda.synchronize()
    dt = (time.time() - t0) / args.iters

    sps = args.batch / dt
    mem = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0
    epoch_s = args.train_frames / sps
    gpu_hours = epoch_s * args.epochs / 3600.0
    cost = gpu_hours * args.cost_per_hr

    mode = "fwd+bwd" if args.backward else "fwd"
    print(f"arch={args.arch}  params={n_params/1e6:.2f}M  device={device}  "
          f"amp={args.amp}  mode={mode}")
    print(f"input={tuple(x.shape)}  batch={args.batch}")
    print(f"  {dt*1000:.1f} ms/it   {sps:.1f} samples/s   peak_mem={mem:.1f} GB")
    print(f"projection @ {args.train_frames} frames/epoch x {args.epochs} epochs:")
    print(f"  {epoch_s/60:.1f} min/epoch   {gpu_hours:.1f} GPU-hours   "
          f"${cost:.0f} @ ${args.cost_per_hr}/hr")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="tracknetx",
                    choices=["tracknet", "fast", "tracknetx"])
    ap.add_argument("--height", type=int, default=288)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--frames", type=int, default=5, help="temporal window (tracknetx)")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--amp", action="store_true", help="autocast fp16")
    ap.add_argument("--backward", action="store_true", help="time fwd+bwd, not just fwd")
    ap.add_argument("--cls", action="store_true", help="tracknet 256-class head (default reg)")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--train-frames", type=int, default=100_000)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--cost-per-hr", type=float, default=2.0)
    main(ap.parse_args())
