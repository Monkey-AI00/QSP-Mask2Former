# QSP-Mask2Former

This repository contains the public configuration and execution layer prepared from the currently available QSP-Mask2Former project files. It focuses on QSP experiment configurations, COCO-style plug-dataset registration, training and evaluation entry points, and paper-related ablations.

## Source completeness notice

The current public repository is a dependency overlay, not a complete standalone implementation of QSP-Mask2Former. The source originally committed to this repository does not contain the modified Mask2Former decoder, criterion, shape-prior fusion module, or `PlugQSPInstanceDatasetMapper` implementation.

To execute the QSP configurations faithfully, provide the original QSP-modified Mask2Former checkout through `MASK2FORMER_ROOT`. An unmodified upstream Mask2Former checkout does not implement the additional `MODEL.MASK_FORMER.PRIOR_*` options or the `plug_qsp_instance` mapper used by these configurations.

## Repository structure

```text
QSP-Mask2Former/
├── configs/
│   ├── base/
│   ├── qsp/
│   └── ablations/
├── qsp_mask2former/
│   └── data/
├── tools/
│   └── train_net.py
├── .gitignore
├── README.md
└── requirements.txt
```

| Path | Function |
| --- | --- |
| `configs/base/` | Plug instance-segmentation schedules that inherit from external Mask2Former R50 or Swin-L configurations. |
| `configs/qsp/` | Main QSP and QSP augmentation configurations. |
| `configs/ablations/` | Table 2 component ablations, geometry-loss comparison, and prior-bank-size experiments. |
| `qsp_mask2former/data/` | COCO dataset loading, background-category filtering, and train/validation registration. |
| `tools/train_net.py` | Unified Detectron2 launcher for training and `--eval-only` evaluation. |
| `requirements.txt` | Direct Python dependencies; Detectron2 and the QSP-modified Mask2Former checkout are installed separately. |

## External prerequisites

- Python 3.8 or later
- PyTorch and torchvision compatible with the selected CUDA runtime
- Detectron2 compatible with the supplied Mask2Former checkout
- The original QSP-modified Mask2Former checkout
- A compiled `MultiScaleDeformableAttention` extension from Mask2Former

Install the direct Python packages:

```bash
python -m pip install -r requirements.txt
```

Install Detectron2 using the version required by the QSP-modified Mask2Former checkout. Compile the Mask2Former CUDA operation from its source tree:

```bash
cd /path/to/QSP-modified-Mask2Former/mask2former/modeling/pixel_decoder/ops
sh make.sh
```

Set the source location before launching an experiment:

```bash
export MASK2FORMER_ROOT=/path/to/QSP-modified-Mask2Former
```

On Windows PowerShell:

```powershell
$env:MASK2FORMER_ROOT = 'C:\path\to\QSP-modified-Mask2Former'
```

The launcher checks for `${MASK2FORMER_ROOT}/train_net.py` and `${MASK2FORMER_ROOT}/mask2former/__init__.py`. It copies the local configuration tree to a temporary directory and resolves the `${MASK2FORMER_ROOT}` placeholders before Detectron2 loads the selected YAML file.

## Dataset format

Training and validation data use COCO instance-segmentation JSON. Each split directory contains its images and annotation file:

```text
datasets/
├── plug_train/
│   ├── images or image files
│   └── plug_train.json
└── plug_val/
    ├── images or image files
    └── plug_val.json
```

Images may be stored directly in the split directory or in paths referenced by `file_name` in the JSON. Categories with an empty name or the exact name `_background_` are excluded. Remaining category IDs are converted to contiguous zero-based IDs. Images without a retained foreground annotation are excluded.

## Training

R50 QSP with the augmentation configuration:

```bash
python tools/train_net.py \
  --config-file configs/qsp/mask2former_R50_plug_qsp_aug.yaml \
  --num-gpus 1 \
  --train-dataset-root /data/plug_train \
  --train-json-file plug_train.json \
  --val-dataset-root /data/plug_val \
  --val-json-file plug_val.json \
  --output-dir outputs/r50_qsp \
  MODEL.MASK_FORMER.PRIOR_PATH /data/priors/plug_prior_bank.npy
```

Add `--weights /path/to/pretrained.pkl` to initialize from a local checkpoint. Detectron2-supported URL values are also accepted.

Swin-L uses the corresponding configuration:

```bash
python tools/train_net.py \
  --config-file configs/qsp/mask2former_SwinL_plug_qsp_aug.yaml \
  --num-gpus 1 \
  --train-dataset-root /data/plug_train \
  --val-dataset-root /data/plug_val \
  MODEL.MASK_FORMER.PRIOR_PATH /data/priors/plug_prior_bank.npy
```

## Evaluation

Use the same launcher with `--eval-only` and a trained checkpoint:

```bash
python tools/train_net.py \
  --config-file configs/qsp/mask2former_R50_plug_qsp_aug.yaml \
  --eval-only \
  --num-gpus 1 \
  --train-dataset-root /data/plug_train \
  --val-dataset-root /data/plug_val \
  MODEL.WEIGHTS /path/to/model_final.pth \
  MODEL.MASK_FORMER.PRIOR_PATH /data/priors/plug_prior_bank.npy
```

The launcher registers both splits because the existing trainer setup expects complete dataset metadata even in evaluation mode.


## Acknowledgements

This project builds on [Detectron2](https://github.com/facebookresearch/detectron2) and [Mask2Former](https://github.com/facebookresearch/Mask2Former). Follow their license and citation requirements when supplying the external dependencies.

## Citation

The paper citation should be added here after its bibliographic metadata is finalized.

## License status

This repository currently has no top-level open-source license. The repository owner must add an appropriate license before describing this release as licensed open-source software. External Detectron2 and Mask2Former components remain subject to their own licenses.
