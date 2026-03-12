#!/usr/bin/env python3
"""
快速测试脚本 - 验证数据集注册是否正常
"""

import os
import sys

# 添加 detectron2 到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import register_coco_instances


def test_dataset_registration():
    """测试数据集注册"""
    print("=" * 60)
    print("测试数据集注册")
    print("=" * 60)
    
    # 数据集路径
    dataset_root = "/home/users1/sjw/cursor/Yolo_pointrend/plug_train"
    json_file = os.path.join(dataset_root, "plug_train.json")
    image_root = dataset_root
    dataset_name = "plug_train"
    
    # 检查文件是否存在
    if not os.path.exists(json_file):
        print(f"❌ 错误: JSON 文件不存在: {json_file}")
        return False
    
    if not os.path.exists(image_root):
        print(f"❌ 错误: 图片目录不存在: {image_root}")
        return False
    
    print(f"✓ JSON 文件存在: {json_file}")
    print(f"✓ 图片目录存在: {image_root}")
    
    # 注册数据集
    try:
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
        
        print(f"✓ 数据集注册成功: {dataset_name}")
        
    except Exception as e:
        print(f"❌ 数据集注册失败: {e}")
        return False
    
    # 测试加载数据集
    try:
        print("\n正在加载数据集...")
        dataset_dicts = DatasetCatalog.get(dataset_name)
        print(f"✓ 数据集加载成功，包含 {len(dataset_dicts)} 个样本")
        
        # 显示第一个样本的信息
        if len(dataset_dicts) > 0:
            first_sample = dataset_dicts[0]
            print(f"\n第一个样本信息:")
            print(f"  图片文件: {first_sample.get('file_name', 'N/A')}")
            print(f"  图片ID: {first_sample.get('image_id', 'N/A')}")
            print(f"  高度: {first_sample.get('height', 'N/A')}")
            print(f"  宽度: {first_sample.get('width', 'N/A')}")
            print(f"  标注数量: {len(first_sample.get('annotations', []))}")
            
            if len(first_sample.get('annotations', [])) > 0:
                ann = first_sample['annotations'][0]
                print(f"  第一个标注:")
                print(f"    类别ID: {ann.get('category_id', 'N/A')}")
                print(f"    是否有分割: {'segmentation' in ann}")
                print(f"    是否有边界框: {'bbox' in ann}")
        
        # 显示元数据
        metadata = MetadataCatalog.get(dataset_name)
        print(f"\n数据集元数据:")
        print(f"  类别: {metadata.thing_classes}")
        print(f"  评估器类型: {metadata.evaluator_type}")
        
        print("\n" + "=" * 60)
        print("✓ 所有测试通过！数据集可以正常使用。")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"❌ 数据集加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_dataset_registration()
    sys.exit(0 if success else 1)

