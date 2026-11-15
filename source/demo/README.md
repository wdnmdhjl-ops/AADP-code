# Demo

The demo visualizes COCO-format detection results without any server-specific paths.

```bash
python demo/demo.py \
  --image path/to/image.jpg \
  --predictions path/to/coco_instances_results.json \
  --threshold 0.25 \
  --output output/demo/preview.png
```

The prediction file may be a list of detections or a COCO result object containing an `annotations` list. Add `--image-id` when the file contains multiple images.
