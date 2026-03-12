#!/usr/bin/env python3
"""
PointRend 自定义数据集推理脚本
用于对 plug 数据集进行可视化推理
"""

import os
import sys
import cv2
import torch
import argparse
import numpy as np
from pathlib import Path

# 添加 detectron2 到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets.coco import register_coco_instances
from detectron2.projects.point_rend import add_pointrend_config


def register_plug_dataset():
    """注册 plug 数据集（用于获取元数据）"""
    dataset_root = "/home/users1/sjw/cursor/Yolo_pointrend/plug_train"
    json_file = os.path.join(dataset_root, "plug_train.json")
    image_root = dataset_root
    dataset_name = "plug_train"
    
    # 检查是否已注册，避免重复注册
    if dataset_name not in DatasetCatalog.list():
        register_coco_instances(
            dataset_name,
            {},
            json_file,
            image_root,
        )
        MetadataCatalog.get(dataset_name).set(
            thing_classes=["plug"],
            evaluator_type="coco",
        )
        print(f"✓ 已注册数据集: {dataset_name}")
    else:
        print(f"✓ 数据集已注册: {dataset_name}")
    
    return dataset_name


def main():
    parser = argparse.ArgumentParser(description="PointRend 推理脚本")
    parser.add_argument(
        "--config-file",
        default="configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco.yaml",
        help="配置文件路径",
    )
    parser.add_argument(
        "--weights",
        required=True,
        help="模型权重文件路径",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="输入图片路径或目录",
    )
    parser.add_argument(
        "--output",
        default="./output_inference",
        help="输出目录",
    )
    parser.add_argument(
        "--score-thresh",
        type=float,
        default=0.5,
        help="置信度阈值",
    )
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="设备 (cuda:0 或 cpu)",
    )
    parser.add_argument(
        "--show-masks-only",
        action="store_true",
        help="仅显示掩膜，不显示边界框",
    )
    parser.add_argument(
        "--save-binary-mask",
        action="store_true",
        help="同时保存二值掩码（单通道 PNG：前景=255，背景=0）",
    )
    parser.add_argument(
        "--mask-mode",
        choices=["union", "maxscore"],
        default="union",
        help="二值掩码生成方式：union=所有实例并集；maxscore=仅保留最高分实例",
    )
    
    args = parser.parse_args()
    
    # 注册数据集（获取类别信息）
    dataset_name = register_plug_dataset()
    
    # 1. 配置初始化
    cfg = get_cfg()
    add_pointrend_config(cfg)
    
    # 2. 加载配置文件
    cfg.merge_from_file(args.config_file)
    
    # 3. 设置模型权重
    cfg.MODEL.WEIGHTS = args.weights
    
    # 4. 设置推理参数
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.score_thresh
    cfg.MODEL.DEVICE = args.device
    
    # 5. 设置类别数量（根据数据集）
    # PointRend 要求 ROI_HEADS 和 POINT_HEAD 的类别数量必须一致
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # 只有 plug 类别
    cfg.MODEL.POINT_HEAD.NUM_CLASSES = 1  # PointRend 的 Point Head 也需要设置
    
    # 6. 创建推理器
    print(f"正在加载模型: {cfg.MODEL.WEIGHTS}")
    print(f"使用设备: {cfg.MODEL.DEVICE}")
    predictor = DefaultPredictor(cfg)
    print("✓ 模型加载成功!")
    
    # 7. 准备输入输出
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取元数据
    metadata = MetadataCatalog.get(dataset_name)
    
    # 8. 处理输入
    if input_path.is_file():
        # 单个文件
        image_paths = [input_path]
    elif input_path.is_dir():
        # 目录中的所有图片
        image_paths = list(input_path.glob("*.jpg")) + list(input_path.glob("*.png"))
    else:
        print(f"错误: 输入路径不存在: {input_path}")
        return
    
    print(f"找到 {len(image_paths)} 张图片")
    
    # 9. 对每张图片进行推理
    for img_path in image_paths:
        print(f"\n处理: {img_path.name}")
        
        # 读取图片
        im = cv2.imread(str(img_path))
        if im is None:
            print(f"  警告: 无法读取图片 {img_path}")
            continue
        
        # 执行推理
        outputs = predictor(im)
        
        # 获取实例预测结果
        instances = outputs["instances"].to("cpu")

        # 额外导出：二值掩码（H, W），保存为单通道 PNG
        if args.save_binary_mask:
            mask_out_path = output_dir / f"mask_{img_path.stem}.png"
            if instances.has("pred_masks") and len(instances) > 0:
                # pred_masks: (N, H, W) bool/uint8
                pred_masks = instances.pred_masks
                if hasattr(pred_masks, "numpy"):
                    pred_masks_np = pred_masks.numpy()
                else:
                    pred_masks_np = np.asarray(pred_masks)

                if pred_masks_np.ndim != 3:
                    raise ValueError(f"pred_masks 维度异常，期望 (N,H,W)，实际为 {pred_masks_np.shape}")

                if args.mask_mode == "maxscore":
                    # 选择最高分的实例
                    scores = instances.scores.numpy() if instances.has("scores") else None
                    best_idx = int(np.argmax(scores)) if scores is not None and scores.size > 0 else 0
                    binary_mask = pred_masks_np[best_idx]
                else:
                    # union：所有实例并集
                    binary_mask = np.any(pred_masks_np, axis=0)

                binary_mask_u8 = (binary_mask.astype(np.uint8) * 255)
            else:
                # 没有预测到实例时，输出全 0 mask
                h, w = im.shape[:2]
                binary_mask_u8 = np.zeros((h, w), dtype=np.uint8)

            cv2.imwrite(str(mask_out_path), binary_mask_u8)
            print(f"  ✓ 二值掩码已保存: {mask_out_path}")
        
        # 如果仅显示掩膜，移除边界框
        if args.show_masks_only and instances.has("pred_boxes"):
            instances.remove("pred_boxes")
        
        # 可视化结果
        v = Visualizer(
            im[:, :, ::-1],  # BGR -> RGB
            metadata=metadata,
            scale=1.2,
            instance_mode=ColorMode.IMAGE_BW if args.show_masks_only else ColorMode.IMAGE,
        )
        
        if instances.has("pred_masks"):
            if args.show_masks_only:
                out = v.overlay_instances(masks=instances.pred_masks)
            else:
                out = v.draw_instance_predictions(instances)
        else:
            out = v.draw_instance_predictions(instances)
        
        # 保存结果
        output_path = output_dir / f"pred_{img_path.name}"
        cv2.imwrite(str(output_path), out.get_image()[:, :, ::-1])  # RGB -> BGR
        print(f"  ✓ 结果已保存: {output_path}")
        
        # 打印检测信息
        if len(instances) > 0:
            scores = instances.scores.tolist()
            print(f"  检测到 {len(instances)} 个实例，置信度: {[f'{s:.2f}' for s in scores]}")
        else:
            print(f"  未检测到任何实例")
    
    print(f"\n✓ 所有图片处理完成! 结果保存在: {output_dir}")


if __name__ == "__main__":
    main()

