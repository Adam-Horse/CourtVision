#!/usr/bin/env bash
# Download the curated match videos (<=720p) for distillation.
# Usage:  bash tracknet/fetch_videos.sh [out_dir] [list_file]
set -euo pipefail

OUT="${1:-datasets/raw_videos}"
LIST="${2:-tracknet/videos.txt}"
mkdir -p "$OUT"

command -v yt-dlp >/dev/null 2>&1 || { export PIP_BREAK_SYSTEM_PACKAGES=1; pip install -q yt-dlp; }

echo "==> downloading (<=720p) from $LIST -> $OUT"
# 720p is plenty: the teacher labels fine at 720p and we train at 512x288.
# --download-archive lets you re-run to resume without re-fetching.
yt-dlp -f "bestvideo[height<=720]+bestaudio/best[height<=720]" \
    --merge-output-format mp4 \
    --download-archive "$OUT/.archive" \
    -o "$OUT/%(title).70s [%(id)s].%(ext)s" \
    -a "$LIST"

echo "==> done."
du -sh "$OUT" 2>/dev/null || true
ls -1 "$OUT"
