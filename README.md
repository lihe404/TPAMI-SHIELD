# SHIELD — Toward Real-World Holistic Privacy-Preserving Person Re-Identification

## Repository layout

| Path | Description |
| --- | --- |
| `dataset_generation.py` | Stage 1 entry point: protected dataset generation. |
| `train_latent.py` | Stage 2 entry point: system-model training. |
| `test_latent.py` | Stage 3 entry point: evaluation. |
| `config/`, `configs/` | yacs configuration defaults and the Market-1501 / MSMT17 YAMLs. |
| `datasets/` | ReID dataset classes and DataLoader factory — **first-party source code, not a data directory**. |
| `model/`, `processor/`, `loss/`, `solver/`, `utils/` | Model, training/evaluation loops, losses, optimizer/scheduler, and utilities. |

## Environment

The code uses Python and PyTorch with CUDA. Core dependencies include torchvision, NumPy, yacs, Pillow, timm, transformers, and diffusers.

There is no root `requirements.txt` or environment lock file. `diffattack/requirements.txt` covers the DiffAttack component only; the full SHIELD environment must be prepared separately.

## Data and weights

* **Market-1501** is expected under `DATASETS.ROOT_DIR` as `market1501/Market-1501-v15.09.15/` with `bounding_box_train`, `query`, and `bounding_box_test` subdirectories. **MSMT17** is expected as `MSMT17/` with `train` and `test` subdirectories plus `list_train.txt`, `list_val.txt`, `list_query.txt`, and `list_gallery.txt`. The datasets themselves are not included in this repository.
* **Precomputed latent dictionaries** (`img_path -> latent tensor`) are loaded by every dataloader from code-level paths in `datasets/bases.py` (separate files for Market-1501/MSMT17 train and evaluation). All three stages need them; they are not included and no generator script for them is present.
* **Protected-data dictionaries** (`.pth`, `img_path -> image tensor`) are loaded from code-level paths in `processor/processor_latent.py`; training validation and evaluation need them, and they are not included.
* **Pretrained weights.** Weight files are not included. The shipped YAMLs contain author-machine paths for `MODEL.PRETRAIN_PATH`, `MODEL.SUPERVISOR_PRETRAIN_PATH`, and `TEST.WEIGHT`; provide local files for these keys before running.

## Configuration

The YAMLs contain absolute paths and GPU ids for `DATASETS.ROOT_DIR`, `MODEL.PRETRAIN_PATH`, `MODEL.SUPERVISOR_PRETRAIN_PATH`, `TEST.WEIGHT`, `OUTPUT_DIR`, `IMG_DIR`/`IMG_TEST_DIR`, and `MODEL.DEVICE_ID`. Before running, edit a copy of the YAML for your own paths.

Some required paths are hard-coded in Python and cannot be overridden from the YAML: the latent dictionaries in `datasets/bases.py`, the CLIP directory and stage-1 protected-train output path in `processor/dataset_generation.py`, and the protected-data dictionary paths in `processor/processor_latent.py`. These must be edited or populated, and missing data and weights must be added.

On yacs types: `MODEL.DEVICE_ID` defaults to a string. If it is overridden on the command line, the value must stay a string, e.g. `MODEL.DEVICE_ID "'0'"`; editing the YAML is safer.

## Usage

After the preparation above (data, weights, latent dictionaries, protected-data dictionaries, and edited YAMLs), the intended entry points are:

These commands are templates; the setup issues described above must be resolved before running them.

### Stage 1: Protected dataset generation

```bash
python dataset_generation.py --config_file configs/market1501/vit_small_ice_generation.yml
```

### Stage 2: System-model training

```bash
python train_latent.py --config_file configs/market1501/vit_small_ics.yml
```

### Stage 3: Evaluation

```bash
python test_latent.py --config_file configs/market1501/vit_small_ics.yml
```

For MSMT17 training/evaluation, use `configs/msmt17/vit_small_ics.yml`.

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

## Acknowledgements

We thank the authors of [DiffAttack](https://github.com/WindVChen/DiffAttack) and [MSINet](https://github.com/vimar-gu/MSINet) for their inspiring work. Parts of our code are based on their implementations.
