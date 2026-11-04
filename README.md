# SHIELD — Toward Real-World Holistic Privacy-Preserving Person Re-Identification

> **Status:** research code snapshot from the authors' archive. Datasets, pretrained weights, and precomputed artifacts are not included, and the three-stage pipeline has not been validated end-to-end.

## Citation

Qianxiang Meng, He Li, Min Cao, and Mang Ye, "Toward Real-World Holistic Privacy-Preserving Person Re-Identification," *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 48, no. 6, pp. 6587–6600, June 2026. DOI: [10.1109/TPAMI.2026.3660922](https://doi.org/10.1109/TPAMI.2026.3660922). Paper: [IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/11370689).

```bibtex
@article{meng2026shield,
  title   = {Toward Real-World Holistic Privacy-Preserving Person Re-Identification},
  author  = {Meng, Qianxiang and Li, He and Cao, Min and Ye, Mang},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume  = {48},
  number  = {6},
  pages   = {6587--6600},
  year    = {2026},
  month   = {June},
  doi     = {10.1109/TPAMI.2026.3660922}
}
```

## About this code

SHIELD is a holistic privacy-preserving person re-identification framework. The paper describes three components: protected dataset generation (PDG), original feature deconstruction (OFD), and protected feature alignment (PFA). This repository is organized as a two-stage workflow: generate a protected version of the training data, then train and evaluate a re-identification system model on protected data.

This is a structural description of the repository only; the paper-to-code mapping has not been audited and is not claimed.

## Repository layout

| Path | Role |
|---|---|
| `dataset_generation.py` | Stage 1 entry point: protected dataset generation. |
| `train_latent.py` | Stage 2 entry point: system-model training. |
| `test_latent.py` | Stage 3 entry point: evaluation. |
| `config/`, `configs/` | yacs configuration defaults and the Market-1501 / MSMT17 YAMLs. |
| `datasets/` | ReID dataset classes and DataLoader factory — **first-party source code, not a data directory**. |
| `model/`, `processor/`, `loss/`, `solver/`, `utils/` | Model, training/evaluation loops, losses, optimizer/scheduler, and utilities. |
| `diffattack/` | Vendored DiffAttack code (Apache-2.0); used by the processors. |
| `MSINet_in/` | Vendored MSINet code; not imported by the main pipeline. |
| `boost/`, `bosst_test/` | Empty placeholder directories for stage-1 sample images. |
| `diffattack/pytorch_fid/*.npz` | Files in the original archive: precomputed FID statistics (~96 MB), not used by the three-stage pipeline; planned for exclusion from Git. |

The original snapshot also had no project-level `requirements.txt`, environment lock file, or `LICENSE`; only `diffattack/.gitignore` existed inside the vendored component.

## Preparation before running

### Data

* **Market-1501** is expected under `DATASETS.ROOT_DIR` as `market1501/Market-1501-v15.09.15/` with `bounding_box_train`, `query`, and `bounding_box_test` subdirectories. **MSMT17** is expected as `MSMT17/` with `train` and `test` subdirectories plus `list_train.txt`, `list_val.txt`, `list_query.txt`, and `list_gallery.txt`.
* **Precomputed latent dictionaries** (`img_path -> latent tensor`) are loaded by every dataloader from code-level paths in `datasets/bases.py` (separate files for Market-1501/MSMT17 train and evaluation). All three stages need them; they are not included and no generator script for them is present.
* **Protected-data dictionaries** (`.pth`, `img_path -> image tensor`) are loaded from code-level paths in `processor/processor_latent.py`; training validation and evaluation need them, and they are not included.

### Pretrained weights and model directories

* `MODEL.PRETRAIN_PATH` (pretrained ViT-S) and `MODEL.SUPERVISOR_PRETRAIN_PATH` (supervisor checkpoint) must point to real files; the shipped YAMLs use author-machine paths.
* `TEST.WEIGHT` must point to a system-model checkpoint produced by stage 2 (the shipped YAMLs use the filename pattern `transformermodel_reid_30.pth`).
* `MODEL.PRETRAINED_DIFFUSION_PATH` must point to a local Stable Diffusion pipeline directory. Stage 1 needs it, and evaluation also still loads it even though the inference loop does not use it.
* A local CLIP ViT-L/14 directory is hard-coded in `processor/dataset_generation.py` and is loaded at stage 1 even though CLIP losses are commented out; edit that path or provide the directory.
* No download URLs are shipped, and no environment versions are claimed to be verified.

### Environment

There is no root requirements file or lock file. The only requirements file in the snapshot is `diffattack/requirements.txt`, which belongs to the vendored DiffAttack demo and is not a complete environment for the SHIELD pipeline (it does not list, for example, `transformers`, `scikit-learn`, `matplotlib`, `seaborn`, or `yacs`). Install and record the environment needed by the code yourself. A CUDA GPU is effectively required: the processors use `device = "cuda"` directly, so set a valid GPU id.

### Configuration

The shipped YAMLs contain the authors' absolute paths and GPU ids for `DATASETS.ROOT_DIR`, `MODEL.PRETRAIN_PATH`, `MODEL.SUPERVISOR_PRETRAIN_PATH`, `TEST.WEIGHT`, `OUTPUT_DIR`, `IMG_DIR`/`IMG_TEST_DIR`, and `MODEL.DEVICE_ID`. Before running, edit a copy of the YAML for your own paths.

Some required paths are hard-coded in Python and cannot be overridden from the YAML: the latent dictionaries in `datasets/bases.py`, the CLIP directory and stage-1 protected-train output path in `processor/dataset_generation.py`, and the protected-data dictionary paths in `processor/processor_latent.py`. These must be edited or populated, and missing data and weights must be added.

On yacs types: `MODEL.DEVICE_ID` defaults to a string. If it is overridden on the command line, the value must stay a string, e.g. `MODEL.DEVICE_ID "'0'"`; editing the YAML is safer.

## Three-stage commands

After the preparation above (data, weights, latent dictionaries, protected-data dictionaries, and edited YAMLs), the intended entry points are:

These commands are templates; the known blockers below must be resolved before running them.

```bash
python dataset_generation.py --config_file configs/market1501/vit_small_ice_generation.yml
python train_latent.py --config_file configs/market1501/vit_small_ics.yml
python test_latent.py --config_file configs/market1501/vit_small_ics.yml
```

For MSMT17 training/evaluation, use `configs/msmt17/vit_small_ics.yml`; the snapshot has no MSMT17 dataset-generation YAML. Extra `KEY VALUE` overrides are accepted after the config file, but they cannot replace the code-level paths or supply the missing artifacts, so editing the configuration and code paths first is required.

## Known blockers (confirmed)

* Missing original datasets, pretrained weights, latent dictionaries, and protected-data dictionaries; none are included and no download links are shipped.
* Several required artifact paths are hard-coded in Python instead of being configuration keys.
* The stage-1 output path for the protected training dictionary differs from the path stage 2 reads, so a manual copy or a source edit is needed in between.
* Protected query/gallery generation (`generate_testset`) exists, but its call in the stage-1 entry point is commented out and the extra models it needs are not constructed; stage 3 has no wired generator for its required inputs.
* No verified root environment file or lock file exists; the vendored DiffAttack requirements are incomplete for this pipeline.

## Outputs

* Training writes a checkpoint under `OUTPUT_DIR` named from `MODEL.NAME` plus the epoch, for example `transformermodel_reid_30.pth`; the supervisor model is not saved. Training and evaluation logs are `1.txt` and `TSNE.txt` under `OUTPUT_DIR`.
* Evaluation also writes t-SNE PNGs into `OUTPUT_DIR`; rank/mAP values are printed to the log only, with no metrics file.
* Stage 1 writes the protected training dictionary to the hard-coded output path and writes sample JPGs under `IMG_DIR`/`IMG_TEST_DIR`.

No reproduction results, checkpoints, logs, weights, or paper metric tables are included in this snapshot, and no paper result is claimed here.

## License and third-party components

* `diffattack/` is vendored DiffAttack code, licensed Apache-2.0; the license text is retained at `diffattack/LICENSE` (its copyright placeholder is unfilled).
* `diffattack/pytorch_fid/` (pytorch-fid code and the FID `.npz` statistics) and `diffattack/torch_nets/` are vendored components, and `MSINet_in/` is a vendored third-party codebase; license files or statements for these are not included in the snapshot, so their upstream licensing should be confirmed before any public redistribution. Keep all third-party files and notices intact.
* The repository has no project-level `LICENSE`; the project license is pending the authors' decision (do not assume MIT, and do not link a root license that does not exist).
