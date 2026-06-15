"""
Offline ball-trajectory recovery from per-frame TrackNet candidates.

This is the principled replacement for the greedy online linker, which could
not undo a wrong commit (it locked onto a camera-pan-drifted court line and
then rejected the real ball). Two complementary stages:

  1. STATIC-PEAK SUPPRESSION (index_static_peaks / suppress_static)
     TrackNet keys on inter-frame motion, and a slowly panning broadcast camera
     makes static court features (baseline T, painted court text) "move", so
     they light up at max intensity every frame. But across the *whole* clip
     they sit in the same place. We bin candidate locations into a coarse grid,
     count how many frames each cell is occupied, and blacklist cells that
     recur far more than any real ball could (a ball never lingers in one cell).
     The blacklist is an explicit indexed list of (x, y, count) you can inspect.

  2. GLOBAL TRAJECTORY (viterbi_trajectory)
     With distractors removed, find the single best path through the remaining
     top-K candidates by dynamic programming over a trellis whose states are
     {candidates} + a MISS state (ball absent/occluded). Edge cost rewards
     small, smooth (low-acceleration) motion and gates impossible jumps; MISS
     costs a fixed amount so the solver only "gives up" when no plausible
     continuation exists. Because it optimises globally, a short distractor
     segment can never win over the long, coherent real-ball trajectory, and
     the ball is re-acquired after gaps - no irreversible commit.
"""
import math

INF = float("inf")


# --------------------------------------------------------------------------- #
# Stage 1: static-peak suppression
# --------------------------------------------------------------------------- #
def index_static_peaks(cands_per_frame, still_radius=14.0, still_frames=8,
                       max_gap=5):
    """Flag candidate instances that are world-fixed distractors, not the ball.

    Discriminator = *stillness*, made gap-tolerant. A static court feature
    (baseline T, painted text, signage) keeps reappearing at the same spot
    frame after frame, moving only a few px (camera pan), even if TrackNet
    fires on it only intermittently. The ball moves ~17 px/frame and never
    returns to the same pixel.

    We grow "still chains": a chain links to a candidate within `still_radius`
    of its anchor, tolerating up to `max_gap` frames with no match (so the
    intermittent text peaks still chain up). A chain seen in >= `still_frames`
    frames is a distractor; only those specific candidate instances are flagged,
    so the ball can pass through the same place at another time.

    Returns (static_mask, static_list) with static_list = [(x, y, count)].
    """
    static_mask = [[False] * len(cs) for cs in cands_per_frame]
    static_list = []
    chains = []  # dict(x, y, count, miss, members=[(t,k)])

    def retire(ch):
        if ch["count"] >= still_frames:
            for (mt, mk) in ch["members"]:
                static_mask[mt][mk] = True
            static_list.append((ch["x"], ch["y"], ch["count"]))

    for t, cs in enumerate(cands_per_frame):
        used = [False] * len(cs)
        nxt = []
        for ch in chains:
            best, bd = None, still_radius
            for k, (x, y, _v) in enumerate(cs):
                if used[k]:
                    continue
                d = math.hypot(x - ch["x"], y - ch["y"])
                if d <= bd:
                    bd, best = d, k
            if best is None:
                ch["miss"] += 1
                if ch["miss"] > max_gap:
                    retire(ch)
                else:
                    nxt.append(ch)        # keep coasting; anchor unchanged
                continue
            used[best] = True
            x, y, _v = cs[best]
            ch["x"] = 0.7 * ch["x"] + 0.3 * x   # slow EMA follows pan drift
            ch["y"] = 0.7 * ch["y"] + 0.3 * y
            ch["count"] += 1
            ch["miss"] = 0
            ch["members"].append((t, best))
            if ch["count"] >= still_frames:
                for (mt, mk) in ch["members"]:
                    static_mask[mt][mk] = True
            nxt.append(ch)
        for k, (x, y, _v) in enumerate(cs):
            if not used[k]:
                nxt.append({"x": x, "y": y, "count": 1, "miss": 0,
                            "members": [(t, k)]})
        chains = nxt
    for ch in chains:
        retire(ch)
    static_list.sort(key=lambda t: -t[2])
    return static_mask, static_list


def suppress_static(cands_per_frame, static_mask):
    """Drop candidate instances flagged static. Returns filtered copy."""
    out = []
    for cs, mask in zip(cands_per_frame, static_mask):
        out.append([c for c, m in zip(cs, mask) if not m])
    return out


# --------------------------------------------------------------------------- #
# Stage 2: global trajectory via Viterbi
# --------------------------------------------------------------------------- #
def viterbi_trajectory(cands_per_frame, max_jump=260.0, w_acc=0.6,
                       miss_cost=140.0, gap_cost=90.0):
    """Best path through candidates + MISS. Returns list of (x,y) or None.

    Cost model (all positive, minimised):
      emission : real candidate 0 ; MISS = miss_cost
      transition real_i -> real_j : dist + w_acc * |accel|   (INF if dist>gate)
                  real <-> MISS     : gap_cost
                  MISS  -> MISS      : 0
    Acceleration uses the position two frames back via the best back-pointer
    (a 2nd-order approximation), which is what penalises near-static linear
    drift relative to fast ballistic ball motion.
    """
    T = len(cands_per_frame)
    if T == 0:
        return []

    # states[t]: list of positions; last entry None = MISS.
    states = []
    for t in range(T):
        s = [(c[0], c[1]) for c in cands_per_frame[t]]
        s.append(None)
        states.append(s)

    cost = [[INF] * len(states[t]) for t in range(T)]
    back = [[-1] * len(states[t]) for t in range(T)]

    for j, p in enumerate(states[0]):
        cost[0][j] = 0.0 if p is not None else miss_cost

    for t in range(1, T):
        for j, pj in enumerate(states[t]):
            emis = 0.0 if pj is not None else miss_cost
            best, bi = INF, -1
            for i, pi in enumerate(states[t - 1]):
                ci = cost[t - 1][i]
                if ci == INF:
                    continue
                tr = _transition(pi, pj, back, states, t - 1, i, w_acc,
                                 max_jump, gap_cost)
                if tr == INF:
                    continue
                c = ci + tr
                if c < best:
                    best, bi = c, i
            if best < INF:
                cost[t][j] = best + emis
                back[t][j] = bi

    # Backtrack from the cheapest terminal state.
    last = min(range(len(states[T - 1])), key=lambda j: cost[T - 1][j])
    path = [None] * T
    j = last
    for t in range(T - 1, -1, -1):
        path[t] = states[t][j]
        j = back[t][j]
        if j < 0 and t > 0:
            # Should not happen, but guard against broken chains.
            j = len(states[t - 1]) - 1
    return path


def _transition(pi, pj, back, states, t_prev, i, w_acc, max_jump, gap_cost):
    if pi is None and pj is None:
        return 0.0
    if pi is None or pj is None:
        return gap_cost
    dist = math.hypot(pj[0] - pi[0], pj[1] - pi[1])
    if dist > max_jump:
        return INF
    acc = 0.0
    h = back[t_prev][i]
    if h >= 0 and t_prev - 1 >= 0:
        ph = states[t_prev - 1][h]
        if ph is not None:
            vx_prev, vy_prev = pi[0] - ph[0], pi[1] - ph[1]
            vx_cur, vy_cur = pj[0] - pi[0], pj[1] - pi[1]
            acc = math.hypot(vx_cur - vx_prev, vy_cur - vy_prev)
    return dist + w_acc * acc


# --------------------------------------------------------------------------- #
# Post-processing
# --------------------------------------------------------------------------- #
def interpolate_gaps(path, max_gap=7):
    """Linearly fill MISS runs of length <= max_gap between two detections."""
    out = list(path)
    n = len(out)
    t = 0
    while t < n:
        if out[t] is not None:
            t += 1
            continue
        j = t
        while j < n and out[j] is None:
            j += 1
        if t > 0 and j < n:  # bounded gap
            gap = j - t
            if gap <= max_gap:
                (x0, y0), (x1, y1) = out[t - 1], out[j]
                for k in range(gap):
                    a = (k + 1) / (gap + 1)
                    out[t + k] = (x0 + a * (x1 - x0), y0 + a * (y1 - y0))
        t = j
    return out


def drop_short_runs(path, min_len=4):
    """Null out isolated detection runs shorter than min_len frames."""
    out = list(path)
    n = len(out)
    t = 0
    while t < n:
        if out[t] is None:
            t += 1
            continue
        j = t
        while j < n and out[j] is not None:
            j += 1
        if j - t < min_len:
            for k in range(t, j):
                out[k] = None
        t = j
    return out
