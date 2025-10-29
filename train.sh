#!/usr/bin/env bash
# Wrapper script to launch OTPNet training with default settings.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "${ROOT_DIR}/train.py" \
  --train-file "${ROOT_DIR}/dataset/train_wv3.h5" \
  --valid-file "${ROOT_DIR}/dataset/valid_wv3.h5" \
  --output-dir "${ROOT_DIR}/checkpoints" \
  "$@"
