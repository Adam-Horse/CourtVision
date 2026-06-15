#!/usr/bin/env bash
# One-command TrackNetX run: (optionally) auto-label raw videos with the 720p
# teacher, then train TrackNetX on the public set + the pseudo-labeled set.
#
# Usage:
#   bash tracknet/run_tracknetx.sh                 # train on public set only
#   bash tracknet/run_tracknetx.sh /path/to/videos # auto-label, then train on both
set -euo pipefail
export PIP_BREAK_SYSTEM_PACKAGES=1

VIDEOS="${1:-}"                                   # optional dir of match videos
TEACHER="tracknet/weights/tracknet_720.pt"
PUBLIC="datasets/tracknet_ds/Dataset"
DISTILL="datasets/distill"
CONF="${CONF:-0.5}"                               # teacher confidence threshold
EPOCHS="${EPOCHS:-40}"
BATCH="${BATCH:-8}"

PLAY="${PLAY_WEIGHTS:-runs/classify/runs/play_classifier_final/weights/best.pt}"

DATA_ROOTS="$PUBLIC"
if [ -n "$VIDEOS" ] && [ -d "$VIDEOS" ]; then
  PLAY_ARG=""
  if [ -f "$PLAY" ]; then
    echo "==> play gate ON ($PLAY) - non-play clips will be dropped"
    PLAY_ARG="--play-weights $PLAY"
  else
    echo "==> play gate OFF (no classifier at $PLAY)"
  fi
  echo "==> auto-labeling $VIDEOS with teacher $TEACHER (conf>=$CONF)"
  python3 -m tracknet.autolabel --videos "$VIDEOS" --weights "$TEACHER" \
      --out "$DISTILL" --conf "$CONF" --save-width 1280 --save-height 720 $PLAY_ARG
  DATA_ROOTS="$PUBLIC $DISTILL"
else
  echo "==> no videos dir given; training on public set only"
fi

echo "==> training TrackNetX on: $DATA_ROOTS"
python3 -m tracknet.train_x \
    --data $DATA_ROOTS \
    --frames 5 --width 512 --height 288 --batch "$BATCH" --epochs "$EPOCHS" \
    --val-games game9 game10 --min-conf "$CONF" \
    --out tracknet/weights/tracknetx.pt

echo "==> done. best model -> tracknet/weights/tracknetx.pt"
