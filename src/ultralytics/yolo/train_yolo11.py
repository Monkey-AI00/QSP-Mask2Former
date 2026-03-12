#!/usr/bin/env python3
"""
YOLO11 训练脚本
使用自定义数据集训练YOLO11模型
"""
from ultralytics import YOLO
import os
import argparse
from pathlib import Path

def train_yolo11():
    """
    训练YOLO11模型
    """
    # 检查数据集配置文件
    dataset_yaml = "dataset.yaml"
    if not os.path.exists(dataset_yaml):
        print(f"错误：未找到数据集配置文件 {dataset_yaml}")
        print("请先运行 organize_dataset.py 组织数据集，并确保 dataset.yaml 存在")
        return
    
    # 加载预训练模型
    # 可选模型: yolo11n.pt (nano, 最小最快)
    #          yolo11s.pt (small)
    #          yolo11m.pt (medium)
    #          yolo11l.pt (large)
    #          yolo11x.pt (xlarge, 最大最准确)
    model_name = "yolo11n.pt"  # 根据你的需求修改
    
    # 如果本地有预训练模型，使用本地路径
    if os.path.exists(model_name):
        print(f"使用本地预训练模型: {model_name}")
        model = YOLO(model_name)
    else:
        print(f"下载并使用预训练模型: {model_name}")
        model = YOLO(model_name)  # 会自动下载
    
    # 训练参数
    results = model.train(
        data=dataset_yaml,      # 数据集配置文件路径
        epochs=100,              # 训练轮数（可根据需要调整）
        imgsz=640,               # 输入图片尺寸
        batch=32,                # 批次大小（根据GPU内存调整：8, 16, 32等）
        device=0,                # 设备：0为GPU，'cpu'为CPU，或指定GPU编号
        workers=8,               # 数据加载线程数
        project='runs/detect',   # 项目保存目录
        name='yolo11_cable_box',    # 实验名称
        exist_ok=True,           # 允许覆盖已存在的实验
        pretrained=True,         # 使用预训练权重
        optimizer='Adam',        # 优化器：SGD, Adam, AdamW, NAdam, RAdam, RMSProp
        verbose=True,            # 显示详细信息
        seed=0,                  # 随机种子
        deterministic=True,      # 确定性训练
        single_cls=False,        # 单类别模式（如果只有一个类别可以设为True）
        rect=False,              # 矩形训练
        cos_lr=False,           # 余弦学习率调度
        close_mosaic=10,         # 最后N个epoch关闭mosaic增强
        resume=False,            # 从上次检查点恢复训练
        amp=True,                # 自动混合精度训练
        fraction=1.0,           # 使用数据集的比例
        profile=False,           # 性能分析
        freeze=None,             # 冻结层数（None表示不冻结）
        lr0=0.01,                # 初始学习率
        lrf=0.01,                # 最终学习率 (lr0 * lrf)
        momentum=0.937,          # SGD动量
        weight_decay=0.0005,     # 权重衰减
        warmup_epochs=3.0,       # 预热轮数
        warmup_momentum=0.8,     # 预热动量
        warmup_bias_lr=0.1,      # 预热偏置学习率
        box=7.5,                 # 边界框损失权重
        cls=0.5,                 # 分类损失权重
        dfl=1.5,                 # DFL损失权重
        pose=12.0,               # 姿态损失权重（仅姿态任务）
        kobj=1.0,                # 关键点对象损失权重（仅姿态任务）
        label_smoothing=0.0,     # 标签平滑
        nbs=64,                  # 标称批次大小
        overlap_mask=True,       # 训练时掩码重叠（仅分割任务）
        mask_ratio=4,            # 掩码下采样比率（仅分割任务）
        dropout=0.0,             # Dropout（仅分类任务）
        val=True,                # 训练期间验证
    )
    
    print("\n训练完成！")
    print(f"最佳模型保存在: {results.save_dir}")
    print(f"可以使用以下代码加载模型进行推理：")
    print(f"  from ultralytics import YOLO")
    print(f"  model = YOLO('{results.save_dir}/weights/best.pt')")
    print(f"  results = model('path/to/image.jpg')")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train"], default="train", help="运行模式")
    _ = parser.parse_args()

    train_yolo11()

