"""
Heatmap utilities for TrackNet: build Gaussian targets from (x, y) ball
labels, and decode a predicted heatmap back to a ball position (or None).
"""
import numpy as np


def make_gaussian_heatmap(h, w, cx, cy, sigma=5.0):
    """Render a Gaussian blob centred at (cx, cy) into an h*w heatmap in [0,1].

    `cx, cy` are in heatmap pixel coords. Returns zeros if the ball is absent
    (cx or cy is None / NaN) - that is the 'no ball this frame' target.
    """
    hm = np.zeros((h, w), dtype=np.float32)
    if cx is None or cy is None or np.isnan(cx) or np.isnan(cy):
        return hm
    # Only evaluate the Gaussian on a local window for speed.
    r = int(3 * sigma)
    x0, x1 = max(0, int(cx) - r), min(w, int(cx) + r + 1)
    y0, y1 = max(0, int(cy) - r), min(h, int(cy) + r + 1)
    if x0 >= x1 or y0 >= y1:
        return hm
    ys, xs = np.mgrid[y0:y1, x0:x1]
    hm[y0:y1, x0:x1] = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma ** 2))
    return hm


def topk_peaks(hm, k=5, min_val=64, supp=12):
    """Greedy non-max suppression on a 0..255 heatmap -> up to k (x, y, val).

    Unlike a single argmax this surfaces *all* plausible ball candidates in a
    frame, including the real ball when a brighter distractor (logo, scoreboard,
    line junction) outshines it. The trajectory linker then picks the candidate
    consistent with recent motion instead of the globally brightest one.
    """
    hm = np.asarray(hm, dtype=np.float32).copy()
    out = []
    for _ in range(k):
        idx = int(np.argmax(hm))
        y, x = np.unravel_index(idx, hm.shape)
        v = float(hm[y, x])
        if v < min_val:
            break
        out.append((float(x), float(y), v))
        y0, y1 = max(0, y - supp), min(hm.shape[0], y + supp + 1)
        x0, x1 = max(0, x - supp), min(hm.shape[1], x + supp + 1)
        hm[y0:y1, x0:x1] = 0.0
    return out


def logits_to_heatmap(logits):
    """256-class TrackNet output [256,H,W] -> 0..255 grayscale heatmap (numpy)."""
    try:
        import torch
        if isinstance(logits, torch.Tensor):
            return logits.argmax(dim=0).to(torch.float32).cpu().numpy()
    except ImportError:
        pass
    return np.argmax(logits, axis=0).astype(np.float32)


def decode_logits(logits, thresh=128):
    """Decode the 256-class TrackNet output -> (x, y, peak).

    `logits` is the raw network output for one frame, shape [256, H, W]
    (torch tensor or numpy). The 256 channels are per-pixel ball-intensity
    classes; argmax over the channel dim gives a 0..255 grayscale heatmap.
    Below `thresh` (0..255) there is no detectable ball -> (None, None, peak).
    """
    return decode_heatmap(logits_to_heatmap(logits), thresh=thresh)


def decode_heatmap(hm, thresh=0.5):
    """Decode one predicted heatmap -> (x, y, peak) or (None, None, peak).

    The ball is the global maximum; if it's below `thresh` the frame has no
    detectable ball. A small intensity-weighted centroid around the peak gives
    sub-pixel accuracy.
    """
    peak = float(hm.max())
    if peak < thresh:
        return None, None, peak
    py, px = np.unravel_index(int(np.argmax(hm)), hm.shape)
    r = 2
    y0, y1 = max(0, py - r), min(hm.shape[0], py + r + 1)
    x0, x1 = max(0, px - r), min(hm.shape[1], px + r + 1)
    patch = hm[y0:y1, x0:x1].astype(np.float64)
    s = patch.sum()
    if s <= 0:
        return float(px), float(py), peak
    ys, xs = np.mgrid[y0:y1, x0:x1]
    cx = float((xs * patch).sum() / s)
    cy = float((ys * patch).sum() / s)
    return cx, cy, peak
