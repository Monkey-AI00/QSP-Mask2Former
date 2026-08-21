# Table-2 消融复现命令（R50）

下面命令基于以下脚本：
- 训练：`src/detectron2/projects/CV/train/train_mask2former_plug.py`
- 评测：`src/detectron2/projects/CV/tools/visualize/benchmark_extended_metrics.py`

## 1) 公共环境变量

```bash
export WORKSPACE_ROOT=/home/users1/sjw/cursor/workspace
export MASK2FORMER_ROOT=/home/users1/sjw/cursor/Mask2Former
export PYTHONPATH=${WORKSPACE_ROOT}/src/detectron2:${WORKSPACE_ROOT}/src/detectron2/projects/CV:${MASK2FORMER_ROOT}
export CUDA_VISIBLE_DEVICES=1

TRAIN_ROOT=${WORKSPACE_ROOT}/datasets/gangkou/plug_train_merged_0429_train
VAL_ROOT=${WORKSPACE_ROOT}/datasets/gangkou/plug_train_merged_0429_val
TRAIN_JSON=plug_train.json
VAL_JSON=plug_val.json

# 你指定的训练超参（560 train / 70 val）
# 等价含义：batch=4 时，每轮约 560/4=140 iter，5600 iter ≈ 40 epochs
TRAIN_OPTS="SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.00005 SOLVER.MAX_ITER 5600 SOLVER.STEPS (4480,5208)"
```

## 2) 四个消融变体训练命令

### 2.1 Base (Mask2Former)

```bash
python -u ${WORKSPACE_ROOT}/src/detectron2/projects/CV/train/train_mask2former_plug.py \
  --config-file ${WORKSPACE_ROOT}/src/detectron2/projects/CV/configs/InstanceSegmentation/mask2former_R50_plug_ablation_table2_base.yaml \
  --train-dataset-root ${TRAIN_ROOT} \
  --train-dataset-name plug_train_0429_train \
  --train-json-file ${TRAIN_JSON} \
  --val-dataset-root ${VAL_ROOT} \
  --val-dataset-name plug_train_0429_val \
  --val-json-file ${VAL_JSON} \
  --output-dir ${WORKSPACE_ROOT}/outputs/gangkou/output/table2_ablation/base \
  --num-gpus 1 \
  --opts ${TRAIN_OPTS}
```

### 2.2 +Prior-bank

```bash
python -u ${WORKSPACE_ROOT}/src/detectron2/projects/CV/train/train_mask2former_plug.py \
  --config-file ${WORKSPACE_ROOT}/src/detectron2/projects/CV/configs/InstanceSegmentation/mask2former_R50_plug_ablation_table2_priorbank.yaml \
  --train-dataset-root ${TRAIN_ROOT} \
  --train-dataset-name plug_train_0429_train \
  --train-json-file ${TRAIN_JSON} \
  --val-dataset-root ${VAL_ROOT} \
  --val-dataset-name plug_train_0429_val \
  --val-json-file ${VAL_JSON} \
  --output-dir ${WORKSPACE_ROOT}/outputs/gangkou/output/table2_ablation/priorbank \
  --num-gpus 1 \
  --opts ${TRAIN_OPTS}
```

### 2.3 Prior-bank + STN Align

```bash
python -u ${WORKSPACE_ROOT}/src/detectron2/projects/CV/train/train_mask2former_plug.py \
  --config-file ${WORKSPACE_ROOT}/src/detectron2/projects/CV/configs/InstanceSegmentation/mask2former_R50_plug_ablation_table2_priorbank_stn.yaml \
  --train-dataset-root ${TRAIN_ROOT} \
  --train-dataset-name plug_train_0429_train \
  --train-json-file ${TRAIN_JSON} \
  --val-dataset-root ${VAL_ROOT} \
  --val-dataset-name plug_train_0429_val \
  --val-json-file ${VAL_JSON} \
  --output-dir ${WORKSPACE_ROOT}/outputs/gangkou/output/table2_ablation/priorbank_stn \
  --num-gpus 1 \
  --opts ${TRAIN_OPTS}
```

### 2.4 Ours (Prior-bank + STN Align + Gated Correction)

```bash
python -u ${WORKSPACE_ROOT}/src/detectron2/projects/CV/train/train_mask2former_plug.py \
  --config-file ${WORKSPACE_ROOT}/src/detectron2/projects/CV/configs/InstanceSegmentation/mask2former_R50_plug_ablation_table2_ours.yaml \
  --train-dataset-root ${TRAIN_ROOT} \
  --train-dataset-name plug_train_0429_train \
  --train-json-file ${TRAIN_JSON} \
  --val-dataset-root ${VAL_ROOT} \
  --val-dataset-name plug_train_0429_val \
  --val-json-file ${VAL_JSON} \
  --output-dir ${WORKSPACE_ROOT}/outputs/gangkou/output/table2_ablation/ours \
  --num-gpus 1 \
  --opts ${TRAIN_OPTS}
```

## 3) Severity=1.00 评测并导出表2

```bash
PRIOR_BANK_PATH=${WORKSPACE_ROOT}/outputs/gangkou/plug_prior/plug_canonical_prior_bank_0429.npy
CLEAN_ROOT=${WORKSPACE_ROOT}/datasets/gangkou/plug_train_merged_0429

python -u ${WORKSPACE_ROOT}/src/detectron2/projects/CV/tools/visualize/benchmark_extended_metrics.py \
  --config-file ${WORKSPACE_ROOT}/src/detectron2/projects/CV/configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_plug.yaml \
  --config-file-mask2former ${WORKSPACE_ROOT}/src/detectron2/projects/CV/configs/InstanceSegmentation/mask2former_R50_plug.yaml \
  --config-file-mask2former-priorbank ${WORKSPACE_ROOT}/src/detectron2/projects/CV/configs/InstanceSegmentation/mask2former_R50_plug_ablation_table2_priorbank.yaml \
  --config-file-mask2former-priorbank-stn ${WORKSPACE_ROOT}/src/detectron2/projects/CV/configs/InstanceSegmentation/mask2former_R50_plug_ablation_table2_priorbank_stn.yaml \
  --config-file-mask2former-ours ${WORKSPACE_ROOT}/src/detectron2/projects/CV/configs/InstanceSegmentation/mask2former_R50_plug_ablation_table2_ours.yaml \
  --mask2former-root ${MASK2FORMER_ROOT} \
  --clean-root ${CLEAN_ROOT} \
  --json-file plug_train.json \
  --out-root ${WORKSPACE_ROOT}/outputs/gangkou/output/table2_ablation_eval \
  --weights-base ${WORKSPACE_ROOT}/src/detectron2/projects/CV/output/plug_pointrend_0511/model_final.pth \
  --weights-prior ${WORKSPACE_ROOT}/src/detectron2/projects/CV/output/plug_pointrend_ft_0511/model_final.pth \
  --weights-mask2former ${WORKSPACE_ROOT}/outputs/gangkou/output/table2_ablation/base/model_final.pth \
  --weights-mask2former-priorbank ${WORKSPACE_ROOT}/outputs/gangkou/output/table2_ablation/priorbank/model_final.pth \
  --weights-mask2former-priorbank-stn ${WORKSPACE_ROOT}/outputs/gangkou/output/table2_ablation/priorbank_stn/model_final.pth \
  --weights-mask2former-ours ${WORKSPACE_ROOT}/outputs/gangkou/output/table2_ablation/ours/model_final.pth \
  --prior-path-mask2former-qsp ${PRIOR_BANK_PATH} \
  --severities 1.0 \
  --table2-severity 1.0 \
  --score-thr 0.5
```

运行完成后会在 `out-root/tables/` 下生成：
- `table2_ablation.csv`
- `table2_ablation.md`
