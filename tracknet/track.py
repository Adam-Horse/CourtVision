"""
Online ball tracker that adds the *continuation* TrackNet lacks.

TrackNet emits a per-frame heatmap with no memory: taking the global argmax
makes the ball "teleport" onto whichever distractor (logo, scoreboard, line
junction) is momentarily brightest. We fix that with classic single-object
tracking:

  1. Each frame -> top-K heatmap peaks (candidates), not just the argmax.
  2. A constant-velocity motion model predicts where the ball should be; among
     the candidates we pick the one nearest the *prediction*, gated by a radius
     that grows while coasting. A physically impossible jump fails the gate and
     is rejected - the real ball (usually a secondary peak) is chosen instead.
  3. A rally-state machine (IDLE -> CONFIRMING -> LIVE) only declares the ball
     "live" once a short run of consistent motion exists. During serve-prep the
     ball is in-hand (no stable moving peak), so the state stays IDLE and we
     report "between points" even though the court is on screen - closing the
     court-visible-but-not-rallying gap the play/non-play classifier can't.
"""
import math
from enum import Enum


class RallyState(Enum):
    IDLE = "idle"            # no plausible ball trajectory
    CONFIRMING = "confirming"  # tentative track, not yet trusted
    LIVE = "live"           # confirmed moving ball


class BallTracker:
    def __init__(self,
                 base_gate=220.0,      # px radius around prediction (frame res)
                 grow=90.0,            # gate growth per coasted frame
                 max_coast=8,          # frames to coast before dropping a track
                 confirm_frames=4,     # consistent detections to reach LIVE
                 seed_min_val=128.0,   # min peak intensity to start a track
                 max_seed_jump=180.0): # max move while CONFIRMING
        self.base_gate = base_gate
        self.grow = grow
        self.max_coast = max_coast
        self.confirm_frames = confirm_frames
        self.seed_min_val = seed_min_val
        self.max_seed_jump = max_seed_jump
        self.reset()

    def reset(self):
        self.state = RallyState.IDLE
        self.pos = None          # last accepted (x, y)
        self.vel = (0.0, 0.0)    # px/frame
        self.last_frame = None
        self.coast = 0
        self.hits = 0            # consecutive consistent detections

    def _predict(self, frame_idx):
        if self.pos is None or self.last_frame is None:
            return None
        dt = frame_idx - self.last_frame
        return (self.pos[0] + self.vel[0] * dt,
                self.pos[1] + self.vel[1] * dt)

    def _accept(self, frame_idx, cand):
        x, y = cand[0], cand[1]
        if self.pos is not None and self.last_frame is not None:
            dt = max(1, frame_idx - self.last_frame)
            nvx = (x - self.pos[0]) / dt
            nvy = (y - self.pos[1]) / dt
            # EMA-smooth velocity so a single noisy step can't whip the model.
            self.vel = (0.6 * self.vel[0] + 0.4 * nvx,
                        0.6 * self.vel[1] + 0.4 * nvy)
        self.pos = (x, y)
        self.last_frame = frame_idx
        self.coast = 0

    def update(self, frame_idx, candidates):
        """Feed one frame's top-K candidates [(x,y,val), ...].

        Returns (state, ball_xy_or_None). ball_xy is reported only when LIVE.
        """
        pred = self._predict(frame_idx)

        chosen = None
        if pred is not None:
            # Associate: nearest candidate to the prediction within the gate.
            gate = self.base_gate + self.grow * self.coast
            best_d = None
            for c in candidates:
                d = math.hypot(c[0] - pred[0], c[1] - pred[1])
                if d <= gate and (best_d is None or d < best_d):
                    best_d, chosen = d, c

        if chosen is not None:
            # While confirming, also require the raw step to be plausible.
            if self.state == RallyState.CONFIRMING and self.pos is not None:
                if math.hypot(chosen[0] - self.pos[0],
                              chosen[1] - self.pos[1]) > self.max_seed_jump:
                    chosen = None

        if chosen is not None:
            self._accept(frame_idx, chosen)
            self.hits += 1
            if self.state == RallyState.IDLE:
                self.state = RallyState.CONFIRMING
            elif (self.state == RallyState.CONFIRMING
                  and self.hits >= self.confirm_frames):
                self.state = RallyState.LIVE
        else:
            # No association this frame: coast on the motion model.
            self.coast += 1
            self.hits = 0
            if self.pos is not None:
                self.pos = pred if pred is not None else self.pos
                self.last_frame = frame_idx
            if self.coast > self.max_coast:
                # Track is dead. Try to (re)seed from the brightest candidate.
                self._seed(frame_idx, candidates)

        if self.state == RallyState.IDLE and self.pos is None:
            self._seed(frame_idx, candidates)

        ball = self.pos if self.state == RallyState.LIVE else None
        return self.state, ball

    def _seed(self, frame_idx, candidates):
        """Start a fresh tentative track from the strongest candidate."""
        strong = [c for c in candidates if c[2] >= self.seed_min_val]
        if not strong:
            self.reset()
            return
        c = max(strong, key=lambda c: c[2])
        self.state = RallyState.CONFIRMING
        self.pos = (c[0], c[1])
        self.vel = (0.0, 0.0)
        self.last_frame = frame_idx
        self.coast = 0
        self.hits = 1
