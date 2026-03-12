#!/usr/bin/env python3
"""
数据集组织脚本
将dataset_v1中的数据按照YOLO标准格式组织为train/val结构
"""
import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

def organize_dataset(source_dir="dataset_v1", output_dir="dataset", train_ratio=0.8, val_ratio=0.2):
    """
    组织数据集为YOLO标准格式
    
    Args:
        source_dir: 源数据目录
        output_dir: 输出数据集目录
        train_ratio: 训练集比例
        val_ratio: 验证集比例
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # 创建输出目录结构
    for split in ['train', 'val']:
        (output_path / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_path / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # 获取所有图片文件
    image_files = sorted([f for f in source_path.glob("*.png")])
    print(f"找到 {len(image_files)} 张图片")
    
    if len(image_files) == 0:
        print("错误：未找到图片文件！")
        return
    
    # 分割数据集
    train_files, val_files = train_test_split(
        image_files, 
        test_size=val_ratio, 
        random_state=42
    )
    
    print(f"训练集: {len(train_files)} 张")
    print(f"验证集: {len(val_files)} 张")
    
    # 复制训练集
    for img_file in train_files:
        # 复制图片
        shutil.copy2(img_file, output_path / 'images' / 'train' / img_file.name)
        
        # 复制对应的标签文件
        label_file = img_file.with_suffix('.txt')
        if label_file.exists():
            shutil.copy2(label_file, output_path / 'labels' / 'train' / label_file.name)
        else:
            print(f"警告：未找到标签文件 {label_file}")
    
    # 复制验证集
    for img_file in val_files:
        # 复制图片
        shutil.copy2(img_file, output_path / 'images' / 'val' / img_file.name)
        
        # 复制对应的标签文件
        label_file = img_file.with_suffix('.txt')
        if label_file.exists():
            shutil.copy2(label_file, output_path / 'labels' / 'val' / label_file.name)
        else:
            print(f"警告：未找到标签文件 {label_file}")
    
    print(f"\n数据集组织完成！")
    print(f"输出目录: {output_path.absolute()}")
    print(f"训练集: {output_path / 'images' / 'train'}")
    print(f"验证集: {output_path / 'images' / 'val'}")

if __name__ == "__main__":
    organize_dataset()

