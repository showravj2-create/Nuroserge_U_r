#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:-/mnt/data/BraTS2020}"

if [ ! -d "$DATA_ROOT" ]; then
  echo "BraTS dataset not found at $DATA_ROOT"
  echo "Mount the dataset first or pass a different path: ./cloud_run.sh /path/to/BraTS2020"
  exit 1
fi

echo "Checking dataset at $DATA_ROOT"
python scripts/check_brats_dataset.py --data-root "$DATA_ROOT"

echo "Starting training"
python scripts/train.py --config configs/brats_2p5d.yaml --data-root "$DATA_ROOT"
