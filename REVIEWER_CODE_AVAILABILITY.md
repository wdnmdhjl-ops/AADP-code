# Reviewer code-availability checklist

## Included

- AADP detector implementation and the dataset registration change.
- Explicit DIOR and Clipart1k 10-shot ViT-L configs.
- A training/evaluation entry point with explicit seed, batch, learning-rate, iteration, C and alpha_iou arguments.
- Installation requirements, upstream notices, and a source provenance record.

## Before claiming resolution in the response letter

1. Upload this directory to an authorized public or anonymous repository.
2. Download the archive into a clean Linux/CUDA environment.
3. Install the documented dependencies and build the Detectron2 extension.
4. Run one evaluation using a permitted checkpoint and compare the output metrics with the archived run.
5. Record the real URL, tag/commit, checkpoint scope, and validation command in the manuscript and response letter.

The current local package is prepared, but no public URL is asserted until those steps are completed.

## Current access

The package has been uploaded to the public repository: https://github.com/wdnmdhjl-ops/AADP-code

The repository contains the complete source snapshot as `aadp_code_release_20261115.zip`. A clean-machine reproduction and checkpoint download have not yet been completed, so the response letter should state the actual validation scope rather than claim a full independent reproduction.
