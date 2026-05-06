#!/usr/bin/env python3
"""
PointRend 自定义数据集训练脚本
用于训练 plug 数据集的实例分割模型
"""

import os
import sys

# 添加 detectron2 到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json, register_coco_instances
from detectron2.projects.point_rend import add_pointrend_config
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator, DatasetEvaluators
import detectron2.data.transforms as T
from detectron2.data import DatasetMapper, build_detection_train_loader


def load_plug_json(json_file, image_root, dataset_name):
    """自定义加载函数，过滤掉 _background_ 和空类别。"""
    from detectron2.data.datasets.coco import load_coco_json
    import json
    
    # 先读取 JSON 文件获取类别信息
    with open(json_file, 'r') as f:
        coco_data = json.load(f)
    
    # 获取所有类别，过滤掉 _background_ 和空类别名
    all_categories = sorted(coco_data['categories'], key=lambda x: x['id'])
    valid_categories = [
        cat for cat in all_categories
        if str(cat.get('name', '')).strip() and cat.get('name') != '_background_'
    ]
    thing_classes = [cat['name'] for cat in valid_categories]
    
    # 创建类别 ID 映射（原始 ID -> 新的连续 ID，排除 _background_ 和空类别）
    thing_dataset_id_to_contiguous_id = {}
    new_idx = 0
    for cat in valid_categories:
        thing_dataset_id_to_contiguous_id[cat['id']] = new_idx
        new_idx += 1
    
    # 使用 dataset_name=None 调用 load_coco_json，这样它不会设置元数据
    dataset_dicts = load_coco_json(json_file, image_root, dataset_name=None)
    
    # 现在手动设置元数据
    metadata = MetadataCatalog.get(dataset_name)
    metadata.__dict__['thing_classes'] = thing_classes
    metadata.__dict__['thing_dataset_id_to_contiguous_id'] = thing_dataset_id_to_contiguous_id
    
    # 过滤数据集中的标注：移除无效类别标注，并更新类别 ID
    filtered_dicts = []
    for dataset_dict in dataset_dicts:
        filtered_anns = []
        for ann in dataset_dict.get("annotations", []):
            orig_cat_id = ann["category_id"]
            # 只保留在映射中的类别（即有效前景类别）
            if orig_cat_id in thing_dataset_id_to_contiguous_id:
                ann["category_id"] = thing_dataset_id_to_contiguous_id[orig_cat_id]
                filtered_anns.append(ann)
        dataset_dict["annotations"] = filtered_anns
        # 只保留有有效标注的图片
        if len(filtered_anns) > 0:
            filtered_dicts.append(dataset_dict)
    
    return filtered_dicts


def _subdirs_containing_json(root_dir, json_basename, max_show=25):
    """列出 root_dir 下哪些一级子目录中含有 json_basename（用于提示路径写错）。"""
    root_dir = os.path.abspath(root_dir)
    if not os.path.isdir(root_dir):
        return []
    out = []
    try:
        for name in sorted(os.listdir(root_dir)):
            sub = os.path.join(root_dir, name)
            if os.path.isdir(sub) and os.path.isfile(os.path.join(sub, json_basename)):
                out.append(sub)
    except OSError:
        return []
    return out[:max_show]


def register_plug_dataset(dataset_root=None, dataset_name="plug_train", json_filename="plug_train.json"):
    """注册 plug 数据集，返回 `(dataset_name, num_classes)`。"""
    if dataset_root is None:
        dataset_root = "/home/users1/sjw/cursor/Yolo_pointrend/plug_train"

    dataset_root = os.path.abspath(str(dataset_root))
    json_file = os.path.join(dataset_root, str(json_filename))
    image_root = dataset_root

    # 检查文件是否存在
    if not os.path.exists(json_file):
        msg = f"❌ 错误: 找不到 JSON 标注文件: {json_file}"
        candidates = _subdirs_containing_json(dataset_root, str(json_filename))
        if candidates:
            msg += (
                f"\n\n当前目录下没有 {json_filename}，但在其子目录中找到以下候选"
                f"（请把 --train-dataset-root / --val-dataset-root 指到其中**包含图片与 json 的那一层**）：\n"
            )
            msg += "\n".join(f"  - {p}" for p in candidates)
        raise FileNotFoundError(msg)
    if not os.path.exists(image_root):
        raise FileNotFoundError(f"❌ 错误: 找不到图片目录: {image_root}")
    
    # 如果已经注册过，先移除（防止重复运行出错）
    if dataset_name in DatasetCatalog.list():
        DatasetCatalog.remove(dataset_name)
    if dataset_name in MetadataCatalog.list():
        MetadataCatalog.remove(dataset_name)

    # ============================================
    # ✅ 使用自定义加载函数，过滤 _background_
    # ============================================
    print(f"正在注册数据集: {dataset_name}...")
    
    # 先注册一个临时数据集名称，让 load_coco_json 设置初始元数据
    temp_dataset_name = dataset_name + "_temp"
    register_coco_instances(
        temp_dataset_name,
        {},
        json_file,
        image_root,
    )
    
    # 触发一次加载，让 COCO 元数据初始化完成
    _ = MetadataCatalog.get(temp_dataset_name)
    _ = DatasetCatalog.get(temp_dataset_name)
    
    # 现在注册真正的数据集，使用自定义加载函数
    DatasetCatalog.register(dataset_name, lambda: load_plug_json(json_file, image_root, dataset_name))
    
    # 复制元数据（但会被 load_plug_json 更新）
    metadata = MetadataCatalog.get(dataset_name)
    metadata.set(
        json_file=json_file,
        image_root=image_root,
        evaluator_type="coco",
    )
    
    # 加载一次以触发元数据更新
    dataset_dicts = DatasetCatalog.get(dataset_name)
    
    # 获取更新后的元数据
    metadata = MetadataCatalog.get(dataset_name)
    num_classes = len(getattr(metadata, "thing_classes", []))
    num_images = len(dataset_dicts)
    num_annotations = sum(len(d.get("annotations", [])) for d in dataset_dicts)
    
    # 清理临时数据集
    if temp_dataset_name in DatasetCatalog.list():
        DatasetCatalog.remove(temp_dataset_name)
    if temp_dataset_name in MetadataCatalog.list():
        MetadataCatalog.remove(temp_dataset_name)
    
    print(f"✓ 数据集 {dataset_name} 注册成功")
    print(f"  - 类别: {metadata.thing_classes}")
    print(f"  - 类别数: {num_classes}")
    print(f"  - 图片数量: {num_images}")
    print(f"  - 标注数量: {num_annotations}")
    
    return dataset_name, num_classes


class Trainer(DefaultTrainer):
    """自定义训练器"""
    
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """构建评估器"""
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, output_dir=output_folder)
    
    @classmethod
    def build_train_loader(cls, cfg):
        """构建训练数据加载器"""
        mapper = DatasetMapper(
            cfg,
            is_train=True,
            augmentations=[
                T.ResizeShortestEdge(
                    cfg.INPUT.MIN_SIZE_TRAIN,
                    cfg.INPUT.MAX_SIZE_TRAIN,
                    cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
                ),
                T.RandomFlip(horizontal=True),
            ],
        )
        return build_detection_train_loader(cfg, mapper=mapper)


def setup(args):
    """创建配置并执行基本设置"""
    cfg = get_cfg()
    
    # 添加 PointRend 配置
    add_pointrend_config(cfg)
    
    # 加载配置文件
    cfg.merge_from_file(args.config_file)
    
    # 从命令行参数合并配置
    cfg.merge_from_list(args.opts)
    
    # ============================================
    # 📝 数据集配置
    # ============================================
    cfg.DATASETS.TRAIN = ("plug_train",)
    cfg.DATASETS.TEST = ("plug_train",) # 验证集
    
    # 设置输出目录
    if not hasattr(args, 'output_dir') or args.output_dir is None:
        cfg.OUTPUT_DIR = "./output/plug_pointrend"
    else:
        cfg.OUTPUT_DIR = args.output_dir
    
    # ============================================
    # 🔧 关键修正：确保类别数量正确
    # ============================================
    # PointRend 需要显式指定类别数（不含背景）
    # 您的数据集似乎只有 1 类 (plug)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1 
    cfg.MODEL.POINT_HEAD.NUM_CLASSES = 1 # PointRend 特有的头部也需要设置
    
    # 设置设备
    import torch
    if torch.cuda.is_available():
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if visible_devices:
            cfg.MODEL.DEVICE = "cuda:0"
        else:
            cfg.MODEL.DEVICE = "cuda"
    else:
        cfg.MODEL.DEVICE = "cpu"
    
    # 冻结配置
    cfg.freeze()
    
    default_setup(cfg, args)
    
    return cfg


def main(args):
    register_plug_dataset()
    
    cfg = setup(args)
    
    if args.eval_only:
        from detectron2.checkpoint import DetectionCheckpointer
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        return res
    
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=None,
        help="指定使用的 GPU ID"
    )
    
    args = parser.parse_args()
    
    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        print(f"✓ 已设置使用 GPU {args.gpu_id}")
    
    print("=" * 50)
    print(f"配置文件: {args.config_file}")
    print("=" * 50)
    
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )