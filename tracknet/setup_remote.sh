#!/usr/bin/env bash
# One-shot setup for a fresh cloud GPU box (A100). Run from the repo root after
# cloning. Installs deps, downloads the public TrackNet dataset + pretrained
# weights, and verifies CUDA. Then launch training with tracknet/train.py.
set -euo pipefail

echo "==> Python / CUDA check"
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

echo "==> Installing deps"
pip install -q gdown opencv-python-headless numpy

mkdir -p datasets/tracknet_ds tracknet/weights

echo "==> Downloading pretrained TrackNet weights"
[ -f tracknet/weights/tracknet_pretrained.pt ] || \
  gdown "https://drive.google.com/uc?id=1XEYZ4myUN7QT-NeBYJI0xteLsvs-ZAOl" \
        -O tracknet/weights/tracknet_pretrained.pt

echo "==> Downloading dataset (2.56 GB)"
if [ ! -d datasets/tracknet_ds/Dataset ]; then
  gdown "https://drive.google.com/uc?id=1DQ3ZbvokTsgOq6x-ay6O8U2W4a8e3LFw" \
        -O datasets/tracknet_ds/Dataset.zip
  python3 -c "import zipfile; zipfile.ZipFile('datasets/tracknet_ds/Dataset.zip').extractall('datasets/tracknet_ds')"
  rm -f datasets/tracknet_ds/Dataset.zip
fi
echo "==> Dataset games:" && ls datasets/tracknet_ds/Dataset | grep game | tr '\n' ' '; echo

cat <<'EOF'

==> Ready. Launch 720p fine-tune (A100 80GB can handle batch 16-32):

    python -m tracknet.train --data datasets/tracknet_ds/Dataset \
        --width 1280 --height 720 --batch 16 --epochs 8 \
        --val-games game9 game10 --out tracknet/weights/tracknet_720.pt

EOF
