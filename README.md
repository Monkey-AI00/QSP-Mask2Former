# QSP-Mask2Former

This repository primarily contains configurations for QSP experiments, COCO-format plug dataset registration modules, training and evaluation entry points, and configurations for related ablation studies.

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
| `configs/ablations/` | Prior-bank-size experiments. |
| `qsp_mask2former/data/` | COCO dataset loading, background-category filtering, and train/validation registration. |
| `tools/train_net.py` | Unified Detectron2 launcher for training and `--eval-only` evaluation. |
| `requirements.txt` | Direct Python dependencies; Detectron2 and the QSP-modified Mask2Former checkout are installed separately. |

## External prerequisites

- Python 3.8 or later
- PyTorch and torchvision compatible with the selected CUDA runtime
- Detectron2 compatible with the supplied Mask2Former checkout

Install the direct Python packages:

```bash
python -m pip install -r requirements.txt
```

## Dataset

The Plug dataset used in this project is available from Google Drive.

- [Download the plug_data](https://drive.google.com/uc?export=download&id=19NNCDXooxJ7B8T9f3zx7ATTTcE2NONFU)


## Training

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
  --config-file configs/qsp/mask2former_SwinL_plug_qsp_aug.yaml \
  --eval-only \
  --num-gpus 1 \
  --train-dataset-root /data/plug_train \
  --val-dataset-root /data/plug_val \
  MODEL.WEIGHTS /path/to/model_final.pth \
  MODEL.MASK_FORMER.PRIOR_PATH /data/priors/plug_prior_bank.npy
```


## Acknowledgements

This project builds on [Detectron2](https://github.com/facebookresearch/detectron2) and [Mask2Former](https://github.com/facebookresearch/Mask2Former). Follow their license and citation requirements when supplying the external dependencies.

## Citation

The paper citation should be added here after its bibliographic metadata is finalized.

## License status

This repository currently has no top-level open-source license. The repository owner must add an appropriate license before describing this release as licensed open-source software. External Detectron2 and Mask2Former components remain subject to their own licenses.
