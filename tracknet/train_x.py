"""
Train TrackNetX (multi-task: heatmap + sub-pixel offset + visibility).

Supports multiple dataset roots so the human-labeled public set and the
teacher-pseudo-labeled (distillation) set can be mixed in one run. Splitting is
grouped by game (leave-games-out) across all roots, so no clip leaks into val.
Per-sample teacher confidence weights the loss (human labels weight 1.0).

Example (mix public + distilled, train from scratch):
    python -m tracknet.train_x \
        --data datasets/tracknet_ds/Dataset datasets/distill \
        --frames 5 --width 512 --height 288 --batch 8 --epochs 40 \
        --val-games game9 game10 --out tracknet/weights/tracknetx.pt
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .tracknetx import TrackNetX
from .dataset import build_window_samples, list_games_multi, WindowDataset
from .heatmap import decode_heatmap


def multitask_loss(out, tgt, hm_lossfn, w_off, w_vis):
    """Weighted heatmap BCE + masked-L1 offset + visibility BCE (per-sample conf)."""
    wt = tgt["weight"].to(out["heatmap"].device)                # [B]
    hm_l = hm_lossfn(out["heatmap"], tgt["heatmap"])            # [B,1,H,W]
    hm_l = (hm_l.mean(dim=(1, 2, 3)) * wt).mean()

    mask = tgt["mask"]
    denom = mask.sum().clamp(min=1.0)
    off_l = (torch.abs(out["offset"] - tgt["offset"]) * mask).sum() / denom

    vis_l = torch.nn.functional.binary_cross_entropy_with_logits(
        out["vis"], tgt["vis"], reduction="none")
    vis_l = (vis_l * wt).mean()
    return hm_l + w_off * off_l + w_vis * vis_l, (hm_l.item(), off_l.item(), vis_l.item())


@torch.no_grad()
def evaluate(net, loader, device, tol_frac=0.0125):
    net.eval()
    hit = tot = 0
    errs = []
    for inp, tgt in loader:
        inp = inp.to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
            pred = torch.sigmoid(net(inp)["heatmap"])
        pred = pred.float().cpu().numpy()
        ty = tgt["heatmap"].numpy()
        H, W = pred.shape[-2:]
        tol = tol_frac * (H * W) ** 0.5
        for b in range(pred.shape[0]):
            if ty[b, 0].max() <= 0:
                continue
            gy, gx = np.unravel_index(int(ty[b, 0].argmax()), ty[b, 0].shape)
            px, py, _ = decode_heatmap(pred[b, 0], thresh=0.5)
            tot += 1
            if px is not None:
                d = ((px - gx) ** 2 + (py - gy) ** 2) ** 0.5
                errs.append(d)
                if d <= tol:
                    hit += 1
    return hit / max(1, tot), (float(np.median(errs)) if errs else float("nan")), tot


def main(args):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    pairs = list_games_multi(args.data)
    val_set = set(args.val_games or [])
    tr_pairs = [(r, g) for r, g in pairs if g not in val_set]
    va_pairs = [(r, g) for r, g in pairs if g in val_set]
    if not va_pairs:                                  # fall back: last 2 games
        va_pairs = pairs[-2:]
        tr_pairs = pairs[:-2]
    print(f"[train_x] train games={len(tr_pairs)} val games={len(va_pairs)}")

    def collect(pp):
        out = []
        for r, g in pp:
            out += build_window_samples(r, [g], args.frames, args.min_conf)
        return out

    tr, va = collect(tr_pairs), collect(va_pairs)
    print(f"[train_x] samples train={len(tr)} val={len(va)} "
          f"frames={args.frames} res={args.width}x{args.height}")

    tr_ds = WindowDataset(tr, args.width, args.height, args.sigma)
    va_ds = WindowDataset(va, args.width, args.height, args.sigma)
    tr_ld = DataLoader(tr_ds, batch_size=args.batch, shuffle=True,
                       num_workers=args.workers, pin_memory=True, drop_last=True,
                       persistent_workers=args.workers > 0)
    va_ld = DataLoader(va_ds, batch_size=args.batch, shuffle=False,
                       num_workers=args.workers, pin_memory=True)

    net = TrackNetX(frames=args.frames).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    hm_lossfn = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([args.pos_weight], device=device), reduction="none")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    best = -1.0
    for ep in range(args.epochs):
        net.train()
        t0, run = time.time(), 0.0
        for i, (inp, tgt) in enumerate(tr_ld):
            inp = inp.to(device)
            tgt = {k: v.to(device) for k, v in tgt.items()}
            with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                loss, parts = multitask_loss(net(inp), tgt, hm_lossfn,
                                             args.w_offset, args.w_vis)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run += loss.item()
            if i % 50 == 0:
                print(f"  ep{ep} it{i}/{len(tr_ld)} loss={loss.item():.4f} "
                      f"(hm={parts[0]:.4f} off={parts[1]:.4f} vis={parts[2]:.4f})",
                      flush=True)
        sched.step()
        acc, mede, n = evaluate(net, va_ld, device)
        print(f"[train_x] epoch {ep}: train_loss={run/max(1,len(tr_ld)):.4f} "
              f"val_acc={acc:.3f} val_med_err={mede:.1f}px (n={n}) "
              f"{time.time()-t0:.0f}s", flush=True)
        if acc > best:
            best = acc
            torch.save({"model_state_dict": net.state_dict(),
                        "frames": args.frames, "width": args.width,
                        "height": args.height, "val_acc": acc}, out)
            print(f"[train_x] saved best -> {out} (acc={acc:.3f})", flush=True)
    print(f"[train_x] done. best val_acc={best:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=["datasets/tracknet_ds/Dataset"])
    ap.add_argument("--out", default="tracknet/weights/tracknetx.pt")
    ap.add_argument("--val-games", nargs="*", default=None)
    ap.add_argument("--frames", type=int, default=5)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=288)
    ap.add_argument("--sigma", type=float, default=4.0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--pos-weight", type=float, default=200.0)
    ap.add_argument("--w-offset", type=float, default=1.0)
    ap.add_argument("--w-vis", type=float, default=0.1)
    ap.add_argument("--min-conf", type=float, default=0.0,
                    help="drop pseudo-labels below this teacher confidence")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cpu", action="store_true")
    main(ap.parse_args())
