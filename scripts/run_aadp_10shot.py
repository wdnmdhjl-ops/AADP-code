#!/usr/bin/env python3
"""Run or evaluate one AADP 10-shot target-dataset experiment.

The script keeps experiment-specific values explicit and does not download data or
weights. Prepare those assets locally, then invoke this file from the repository root.
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    "DIOR": ROOT / "configs/aadp/vitl_shot10_dior.yaml",
    "Clipart1k": ROOT / "configs/aadp/vitl_shot10_clipart1k.yaml",
}

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=sorted(CONFIGS), required=True)
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Directory relative to the repository root unless absolute")
    p.add_argument("--weights", type=Path, default=Path("weights/trained/few-shot/vitl_0089999.pth"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--C", type=float, default=12.8, dest="constant")
    p.add_argument("--alpha-iou", type=float, default=0.5)
    p.add_argument("--num-gpus", type=int, default=1)
    p.add_argument("--eval-only", action="store_true",
                   help="Evaluate MODEL.WEIGHTS instead of training")
    return p.parse_args()

def inside_root(path: Path) -> Path:
    path = path if path.is_absolute() else ROOT / path
    return path.resolve()

def main() -> int:
    args = parse_args()
    output = inside_root(args.output_dir)
    weights = inside_root(args.weights)
    output.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "tools/train_net.py",
        "--num-gpus", str(args.num_gpus),
        "--config-file", str(CONFIGS[args.dataset]),
        "OUTPUT_DIR", str(output),
        "SEED", str(args.seed),
        "MODEL.WEIGHTS", str(weights),
        "DE.OFFLINE_RPN_CONFIG",
        "configs/RPN/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml",
        "SOLVER.IMS_PER_BATCH", "16",
        "SOLVER.BASE_LR", "0.002",
        "SOLVER.MAX_ITER", "500",
    ]
    if args.eval_only:
        command.insert(2, "--eval-only")
    env = os.environ.copy()
    env["CDFSOD_NWD_CONSTANT"] = str(args.constant)
    env["CDFSOD_IOU_RATIO"] = str(args.alpha_iou)
    print("Working directory:", ROOT)
    print("Command:", " ".join(map(str, command)))
    print("CDFSOD_NWD_CONSTANT=", env["CDFSOD_NWD_CONSTANT"])
    print("CDFSOD_IOU_RATIO=", env["CDFSOD_IOU_RATIO"])
    return subprocess.run(command, cwd=ROOT, env=env).returncode

if __name__ == "__main__":
    raise SystemExit(main())
