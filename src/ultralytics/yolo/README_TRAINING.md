# YOLO11 训练指南

本指南将帮助您使用自定义数据集训练YOLO11模型。

## 📋 目录结构

```
Yolo_pointrend/
├── dataset_v1/              # 原始数据集（图片和标签）
│   ├── 1_Color.png
│   ├── 1_Color.txt
│   └── ...
├── dataset/                 # 组织后的数据集（运行脚本后生成）
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   └── labels/
│       ├── train/
│       └── val/
├── dataset.yaml             # 数据集配置文件
├── organize_dataset.py      # 数据集组织脚本
├── train_yolo11.py          # 训练脚本
└── yolo11n.pt               # 预训练模型（可选）
```

## 🚀 快速开始

### 步骤 1: 安装依赖

确保已安装必要的Python包：

```bash
pip install ultralytics scikit-learn
```

或者如果使用conda：

```bash
conda install -c conda-forge ultralytics scikit-learn
```

### 步骤 2: 组织数据集

运行数据集组织脚本，将 `dataset_v1` 中的数据按照YOLO标准格式组织：

```bash
python organize_dataset.py
```

此脚本会：
- 自动将数据集按 80% 训练集 / 20% 验证集分割
- 创建标准的YOLO目录结构（images/train, images/val, labels/train, labels/val）
- 复制图片和对应的标签文件到相应目录

### 步骤 3: 检查数据集配置

打开 `dataset.yaml` 文件，确认以下信息：

- **path**: 数据集根目录路径（已设置为绝对路径）
- **nc**: 类别数量（当前为1，如果您的数据集有多个类别，请修改）
- **names**: 类别名称（当前为 "object"，请根据实际情况修改）

### 步骤 4: 开始训练

运行训练脚本：

```bash
python train_yolo11.py
```

或者使用命令行直接训练：

```bash
yolo detect train data=dataset.yaml model=yolo11n.pt epochs=100 imgsz=640 batch=16
```

## ⚙️ 训练参数说明

### 模型选择

在 `train_yolo11.py` 中可以修改 `model_name` 来选择不同的模型：

- `yolo11n.pt` - Nano（最小最快，适合资源受限环境）
- `yolo11s.pt` - Small（平衡速度和精度）
- `yolo11m.pt` - Medium（更好的精度）
- `yolo11l.pt` - Large（高精度）
- `yolo11x.pt` - XLarge（最高精度，但最慢）

### 关键参数调整

根据您的硬件和需求，可以调整以下参数：

1. **batch**: 批次大小
   - GPU内存 4GB: batch=8
   - GPU内存 8GB: batch=16
   - GPU内存 16GB+: batch=32 或更大

2. **epochs**: 训练轮数
   - 小数据集（<1000张）: 100-200
   - 中等数据集（1000-10000张）: 50-100
   - 大数据集（>10000张）: 30-50

3. **imgsz**: 输入图片尺寸
   - 默认: 640
   - 更高精度: 1280（但训练更慢）
   - 更快训练: 416（但精度可能降低）

4. **device**: 设备选择
   - `0` 或 `'cuda:0'`: 使用第一个GPU
   - `'cpu'`: 使用CPU（训练会很慢）
   - `[0, 1]`: 使用多个GPU

## 📊 训练结果

训练完成后，结果会保存在 `runs/detect/yolo11_custom/` 目录下：

```
runs/detect/yolo11_custom/
├── weights/
│   ├── best.pt      # 最佳模型（验证集上表现最好）
│   └── last.pt      # 最后一个epoch的模型
├── results.png      # 训练曲线图
├── confusion_matrix.png  # 混淆矩阵
└── ...
```

## 🔍 使用训练好的模型

### Python代码

```python
from ultralytics import YOLO

# 加载训练好的模型
model = YOLO('runs/detect/yolo11_custom/weights/best.pt')

# 单张图片推理
results = model('path/to/image.jpg')

# 显示结果
results[0].show()

# 保存结果
results[0].save('output.jpg')
```

### 命令行

```bash
# 单张图片
yolo predict model=runs/detect/yolo11_custom/weights/best.pt source=path/to/image.jpg

# 视频
yolo predict model=runs/detect/yolo11_custom/weights/best.pt source=path/to/video.mp4

# 摄像头
yolo predict model=runs/detect/yolo11_custom/weights/best.pt source=0
```

## 🐛 常见问题

### 1. 内存不足（CUDA out of memory）

**解决方案**：
- 减小 `batch` 大小（如从16改为8）
- 减小 `imgsz`（如从640改为416）
- 减少 `workers` 数量

### 2. 训练损失不下降

**解决方案**：
- 检查数据集标注是否正确
- 增加训练轮数 `epochs`
- 调整学习率 `lr0`（尝试0.001或0.0001）
- 使用更大的模型（如从yolo11n改为yolo11s）

### 3. 验证集mAP很低

**解决方案**：
- 增加训练数据量
- 检查验证集标注质量
- 使用数据增强（默认已启用）
- 尝试更长的训练时间

### 4. 找不到预训练模型

**解决方案**：
- 确保网络连接正常（会自动下载）
- 或手动下载模型文件到项目目录

## 📚 更多资源

- [Ultralytics YOLO11 官方文档](https://docs.ultralytics.com/)
- [YOLO11 模型文档](https://docs.ultralytics.com/models/yolo11/)
- [训练模式文档](https://docs.ultralytics.com/modes/train/)

## 💡 提示

1. **数据质量很重要**：确保标注准确、完整
2. **数据增强**：YOLO默认启用数据增强，有助于提高模型泛化能力
3. **监控训练**：使用TensorBoard查看训练过程（如果安装了tensorboard）
4. **保存检查点**：训练过程中会自动保存最佳模型和最新模型
5. **验证集**：确保验证集具有代表性，能反映真实场景

祝训练顺利！🎉

