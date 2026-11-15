#!/usr/bin/env python3
"""Visualize COCO-format predictions on one image.

This demo is intentionally independent of the training pipeline. It accepts
publicly available or user-owned image and prediction files and never assumes
a server-specific path.
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--predictions", type=Path, required=True,
                   help="COCO detections JSON: a list with image_id, category_id, bbox, score")
    p.add_argument("--image-id", type=int, default=None,
                   help="Filter detections by image_id; omit when the JSON contains one image")
    p.add_argument("--threshold", type=float, default=0.0)
    p.add_argument("--output", type=Path, default=None,
                   help="Save a PNG instead of opening an interactive window")
    return p.parse_args()


def main():
    args = parse_args()
    image = Image.open(args.image).convert("RGB")
    detections = json.loads(args.predictions.read_text(encoding="utf-8"))
    if isinstance(detections, dict) and "annotations" in detections:
        detections = detections["annotations"]
    if args.image_id is not None:
        detections = [d for d in detections if d.get("image_id") == args.image_id]
    detections = [d for d in detections if float(d.get("score", 1.0)) >= args.threshold]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image)
    for det in detections:
        x, y, w, h = det["bbox"]
        score = float(det.get("score", 1.0))
        category = det.get("category_id", "?")
        ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor="lime", linewidth=2))
        ax.text(x, y, f"{category}:{score:.2f}", color="white", backgroundcolor="black")
    ax.axis("off")
    fig.tight_layout(pad=0)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"saved {args.output} ({len(detections)} detections)")
    else:
        plt.show()


if __name__ == "__main__":
    main()
