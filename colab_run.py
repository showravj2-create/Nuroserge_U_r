# Colab-friendly runner for BraTS 2020
# Usage in Google Colab:
#   !git clone https://github.com/showravj2-create/Nuroserg_U_r.git
#   %cd Nuroserg_U_r
#   !pip install -r requirements.txt
#   !python colab_run.py --dataset-root /content/BraTS2020 --config configs/brats_2p5d.yaml  (passes --data-root to train.py)

import argparse
import subprocess
import sys
from pathlib import Path


def ensure_repo_root():
    root = Path(__file__).resolve().parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def maybe_download_brats(dataset_root: str):
    root = Path(dataset_root)
    if root.exists() and (
        any(root.rglob("*.nii.gz")) or any(root.rglob("*.nii"))
    ):
        print(f"BraTS dataset already exists at {root}")
        return

    print(f"BraTS dataset not found at {root}.")
    print("Please download BraTS 2020 from the official challenge site and extract it here.")
    print("Expected structure: root/Training/BraTS20_Training_001/...")
    raise FileNotFoundError(f"Dataset root not found: {root}")


def main():
    parser = argparse.ArgumentParser(description="Run BraTS training in Google Colab.")
    parser.add_argument("--dataset-root", required=True, help="Path to the extracted BraTS 2020 dataset root")
    parser.add_argument("--config", default="configs/brats_2p5d.yaml", help="Config file path")
    args = parser.parse_args()

    ensure_repo_root()
    maybe_download_brats(args.dataset_root)

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    print("Starting training...")
    subprocess.run(
        [
            sys.executable,
            "scripts/train.py",
            "--config",
            str(cfg_path),
            "--data-root",
            args.dataset_root,
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
