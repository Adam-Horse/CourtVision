"""
Fine-tune TrackNet at higher resolution on the public tennis dataset.

Loads the pretrained backbone (conv1..conv17) with a fresh 1-channel heatmap
head, then trains with weighted BCE on Gaussian targets. Grouped val split
(leave games out) so no clip leaks between train and val.

Example (cloud A100, full 720p):
    python -m tracknet.train --data datasets/tracknet_ds/Dataset \
        --width 1280 --height 720 --batch 16 --epochs 8 \
        --val-games game9 game10 --out tracknet/weights/tracknet_720.pt
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .model import build_finetune
from .dataset import build_samples, list_games, TrackNetDataset
from .heatmap import decode_heatmap


def evaluate(net, loader, device, tol_frac=0.0125):
    """Detection accuracy: predicted peak within tol of GT (when ball present)."""
    net.eval()
    hit = tot = 0
    errs = []
    with torch.no_grad():
        for inp, target in loader:
            inp = inp.to(device)
            with torch.autocast("cuda", dtype=torch.float16):
                pred = torch.sigmoid(net(inp))
            pred = pred.float().cpu().numpy()
            tgt = target.numpy()
            H, W = pred.shape[-2:]
            tol = tol_frac * (H * W) ** 0.5
            for b in range(pred.shape[0]):
                ty = tgt[b, 0]
                if ty.max() <= 0:            # no-ball GT, skip for detection rate
                    continue
                gy, gx = np.unravel_index(int(ty.argmax()), ty.shape)
                px, py, peak = decode_heatmap(pred[b, 0], thresh=0.5)
                tot += 1
                if px is not None:
                    d = ((px - gx) ** 2 + (py - gy) ** 2) ** 0.5
                    errs.append(d)
                    if d <= tol:
                        hit += 1
    acc = hit / max(1, tot)
    mede = float(np.median(errs)) if errs else float("nan")
    return acc, mede, tot


def main(args):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    games = list_games(args.data)
    val_games = args.val_games or games[-2:]
    train_games = [g for g in games if g not in val_games]
    print(f"[train] games train={train_games} val={val_games}")

    tr = build_samples(args.data, train_games)
    va = build_samples(args.data, val_games)
    print(f"[train] samples train={len(tr)} val={len(va)}  res={args.width}x{args.height}")

    tr_ds = TrackNetDataset(tr, args.width, args.height, args.sigma)
    va_ds = TrackNetDataset(va, args.width, args.height, args.sigma)
    tr_ld = DataLoader(tr_ds, batch_size=args.batch, shuffle=True,
                       num_workers=args.workers, pin_memory=True, drop_last=True,
                       persistent_workers=args.workers > 0)
    va_ld = DataLoader(va_ds, batch_size=args.batch, shuffle=False,
                       num_workers=args.workers, pin_memory=True)

    net = build_finetune(args.weights, device=device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler()
    pos_weight = torch.tensor([args.pos_weight], device=device)
    lossfn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    best = -1.0
    for ep in range(args.epochs):
        net.train()
        t0 = time.time()
        run = 0.0
        for i, (inp, target) in enumerate(tr_ld):
            inp, target = inp.to(device), target.to(device)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = net(inp)
                loss = lossfn(logits, target)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run += loss.item()
            if i % 50 == 0:
                print(f"  ep{ep} it{i}/{len(tr_ld)} loss={loss.item():.4f} "
                      f"({(time.time()-t0)/max(1,i+1)*1000:.0f} ms/it)", flush=True)
        sched.step()
        acc, mede, n = evaluate(net, va_ld, device)
        print(f"[train] epoch {ep}: train_loss={run/len(tr_ld):.4f} "
              f"val_acc={acc:.3f} val_med_err={mede:.1f}px (n={n}) "
              f"{time.time()-t0:.0f}s", flush=True)
        if acc > best:
            best = acc
            torch.save({"model_state_dict": net.state_dict(),
                        "width": args.width, "height": args.height,
                        "sigma": args.sigma, "reg_head": True,
                        "val_acc": acc}, out)
            print(f"[train] saved best -> {out} (acc={acc:.3f})", flush=True)
    print(f"[train] done. best val_acc={best:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets/tracknet_ds/Dataset")
    ap.add_argument("--weights", default="tracknet/weights/tracknet_pretrained.pt")
    ap.add_argument("--out", default="tracknet/weights/tracknet_720.pt")
    ap.add_argument("--val-games", nargs="*", default=None)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--sigma", type=float, default=5.0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--pos-weight", type=float, default=200.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cpu", action="store_true")
    main(ap.parse_args())
