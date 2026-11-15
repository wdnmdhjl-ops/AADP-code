# AADP code release snapshot

This directory is a source snapshot for the AADP implementation used in the revised experiments. It is prepared for code review and reproduction. It does not include datasets, pretrained weights, checkpoints, logs, or private server files.

## Scope

- DIOR and Clipart1k, 10-shot, ViT-L.
- Training defaults used in the reported supplementary runs: batch size 16, base learning rate 0.002, 500 iterations, one GPU, seed 0 unless changed explicitly.
- The Wasserstein/NWD constant `C` and the IoU-mixture coefficient `alpha_iou` are controlled by the environment variables `CDFSOD_NWD_CONSTANT` and `CDFSOD_IOU_RATIO`. The helper script exposes them as `--C` and `--alpha-iou`.

## Layout

- `detectron2/`: detector implementation, including the AADP/CD-ViTO changes.
- `configs/aadp/`: explicit 10-shot DIOR and Clipart1k configurations with batch size 16.
- `scripts/run_aadp_10shot.py`: one-command training or evaluation entry point.
- `tools/`: prototype extraction and training utilities.
- `README_UPSTREAM.md`: the original upstream project README retained for provenance.
- `SOURCE_PROVENANCE.json`: source commit, snapshot date, and release exclusions.

## Environment

The source was developed against Python 3.9, PyTorch 1.13.1, torchvision 0.14.1, CUDA 11.7, and cuDNN 8.5. Install the declared research dependencies first:

```bash
conda create -n aadp python=3.9
conda activate aadp
pip install -r requirements.txt
pip install -e .
```

The Detectron2 C++/CUDA extension must be built on the target machine. A compatible CUDA compiler and GPU are required for training.

## Data and pretrained assets

Datasets and pretrained assets are intentionally excluded from this snapshot. Obtain them from their original sources and follow their licenses. Place the target datasets under the paths expected by `detectron2/data/datasets/builtin.py`, the DE-ViT pretrained checkpoint at `weights/trained/few-shot/vitl_0089999.pth`, and the target prototype/background assets under `prototypes_init/` and `weights/initial/`. Do not commit these assets or private absolute paths to a public repository.

## Run one experiment

From the repository root:

```bash
python scripts/run_aadp_10shot.py --dataset DIOR --output-dir output/aadp/dior_seed0 --seed 0 --C 12.8 --alpha-iou 0.5
python scripts/run_aadp_10shot.py --dataset Clipart1k --output-dir output/aadp/clipart1k_seed0 --seed 0 --C 12.8 --alpha-iou 0.5
```

To evaluate a saved checkpoint, add `--eval-only` and pass its path with `--weights`:

```bash
python scripts/run_aadp_10shot.py --dataset DIOR --output-dir output/aadp/dior_seed0 --weights output/aadp/dior_seed0/model_final.pth --eval-only
```

The command records the resolved configuration and evaluation log in the output directory. For an exact comparison, keep the dataset split, support set, pretrained checkpoint, seed, and GPU count fixed.

## Reproducibility boundary

This package contains the source and configurations required to run the method. It does not claim that a clean-machine reproduction was completed as part of this packaging step. Reported metrics and checkpoints remain in the project’s separate experiment archive. The public code link, release tag, and any model/checkpoint availability statement should be filled only after the package has been uploaded and downloaded once from the final reviewer-accessible location.

## License and attribution

The repository contains upstream Detectron2/CD-ViTO components and their notices. See `LICENSE`, `NOTICE`, and `README_UPSTREAM.md`. Check the licenses of datasets and pretrained assets separately before redistribution.


## Included examples

- `logs/` contains two sanitized seed-5 evaluation logs, one for DIOR and one for Clipart1k.
- `output/examples/` contains the matching small JSON metric summaries. Checkpoints and datasets remain excluded.
- `demo/` contains a path-independent COCO prediction visualizer and its command example.

## Code availability

The reviewer-oriented source archive is publicly available at [AADP-code](https://github.com/wdnmdhjl-ops/AADP-code). The repository root contains this README, the license and notice files, and `aadp_code_release_20261115.zip`, which contains the complete source snapshot. The archive excludes datasets, pretrained weights, checkpoints, and private experiment outputs.
